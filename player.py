from __future__ import annotations

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import re
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import trimesh
from plyfile import PlyData
from PySide6 import QtCore, QtGui, QtWidgets


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DG_ROOT = REPO_ROOT / "dgmesh"
LOGO_PATH = REPO_ROOT / "imgs" / "4dgsplayer_logo_split_dual.svg"
if str(DG_ROOT) not in sys.path:
    sys.path.insert(0, str(DG_ROOT))

import nvdiffrast.torch as dr

from gaussian_renderer import render as gaussian_render
from player_ui import RenderView, setup_main_window_ui
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene.cameras import MiniCam
from utils.general_utils import build_covariance_from_scaling_rotation
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.renderer import render_mesh
from utils.graphics_utils import fov2focal
from nvdiffrast_utils import util as dr_util


FRAME_RE = re.compile(r"frame_(\d+)\.ply$", re.IGNORECASE)
LOG_DIR = REPO_ROOT / "logs"
LOG_PATH = LOG_DIR / "4dgsplayer.log"
LOG_ENABLED = "--enable-log" in sys.argv


def log_message(message: str) -> None:
    if not LOG_ENABLED:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def log_exception(prefix: str, exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_message(f"{prefix}\n{tb}")


def show_fatal_error(message: str, detail: str = "") -> None:
    try:
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("4DGS Player Error")
        box.setText(message)
        if detail:
            box.setDetailedText(detail)
        box.exec()
    except Exception:
        pass


@dataclass
class SequenceInfo:
    root: Path
    frames: Dict[int, Path]

    @property
    def frame_ids(self) -> List[int]:
        return sorted(self.frames.keys())

    def get(self, frame_id: int) -> Path:
        return self.frames[frame_id]


class OnlineGaussianBackend:
    def __init__(self, checkpoint_dir: Path, config: Dict[str, object], device: str = "cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.device = device
        self.white_background = bool(config.get("white_background", True))
        self.is_6dof = bool(config.get("is_6dof", False))
        bg_color = [1.0, 1.0, 1.0] if self.white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.gaussians = gaussian_model(
            int(config.get("sh_degree", 3)),
            grid_res=int(config.get("grid_res", 256)),
            density_thres=float(config.get("init_density_threshold", 0.05)),
            dpsr_sig=float(config.get("dpsr_sig", 0.5)),
        )
        self.deform = deform_model(
            is_blender=bool(config.get("is_blender", False)),
            is_6dof=self.is_6dof,
            model_name="deform",
        )
        self.gaussians.load_ply(str(checkpoint_dir), iteration=-1)
        self.deform.load_weights(str(checkpoint_dir), iteration=-1)

    @torch.no_grad()
    def render(self, camera: PlayerCamera, frame_id: int, frame_span: int, pipeline: StaticPipeline):
        span = max(1, int(frame_span))
        t = float(frame_id) / float(span)
        N = self.gaussians.get_xyz.shape[0]
        time_input = torch.ones(N, 1, device=self.device) * t
        d_xyz, d_rotation, d_scaling, _ = self.deform.step(self.gaussians.get_xyz, time_input)
        return gaussian_render(
            camera.mini_cam,
            self.gaussians,
            pipeline,
            self.background,
            d_xyz,
            d_rotation,
            d_scaling,
            self.is_6dof,
        )


class LRUCache:
    def __init__(self, max_items: int = 6):
        self.max_items = max_items
        self._items: OrderedDict[object, object] = OrderedDict()

    def get(self, key: object, loader: Callable[[], object]) -> object:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = loader()
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


class FrameGaussianModel:
    def __init__(
        self,
        xyz,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        sh_degree,
    ):
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self.opacity_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    def get_covariance(self, scaling_modifier=1.0):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)


class StaticPipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False


@dataclass
class OrbitCamera:
    target: np.ndarray
    distance: float
    yaw: float
    pitch: float
    fovy_deg: float = 50.0
    znear: float = 0.01
    zfar: float = 100.0

    def clamp(self):
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.distance = max(0.05, self.distance)

    def eye(self) -> np.ndarray:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        sy = math.sin(yaw)
        cy = math.cos(yaw)
        # DG-Mesh camera conventions are z-up; orbit in the x-y plane and lift along z.
        offset = np.array(
            [
                self.distance * cp * sy,
                -self.distance * cp * cy,
                self.distance * sp,
            ],
            dtype=np.float32,
        )
        return self.target.astype(np.float32) + offset

    def pan(self, dx: float, dy: float):
        eye = self.eye()
        forward = self.target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        scale = self.distance * 0.0015
        self.target = self.target - right * dx * scale + up * dy * scale

    def orbit(self, dx: float, dy: float):
        self.yaw -= dx * 0.35
        self.pitch += dy * 0.35
        self.clamp()

    def zoom(self, delta: float):
        factor = 1.0 - delta * 0.001
        if factor <= 0.1:
            factor = 0.1
        self.distance *= factor
        self.clamp()


class PlayerCamera:
    def __init__(
        self,
        width: int,
        height: int,
        orbit: OrbitCamera,
        device: str = "cuda",
    ):
        self.width = max(64, width)
        self.height = max(64, height)
        self.device = device
        self.orbit = orbit
        self.fovy = math.radians(orbit.fovy_deg)
        self.fovx = 2.0 * math.atan(math.tan(self.fovy / 2.0) * (self.width / self.height))
        self.fx = fov2focal(self.fovx, self.width)
        self.fy = fov2focal(self.fovy, self.height)
        self.znear = orbit.znear
        self.zfar = orbit.zfar

        eye = orbit.eye()
        at = orbit.target.astype(np.float32)
        look_at = at - eye
        look_at = look_at / (np.linalg.norm(look_at) + 1e-8)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(look_at, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(right, look_at)
        cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = cam_up
        c2w[:3, 2] = -look_at
        c2w[:3, 3] = eye
        self.c2w_blender = torch.tensor(c2w, dtype=torch.float32, device=device)

        c2w_opencv = self.c2w_blender.clone()
        c2w_opencv[:3, 1:3] *= -1
        world_view = torch.inverse(c2w_opencv).transpose(0, 1)
        projection = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy
        ).transpose(0, 1).to(device)
        full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

        self.mini_cam = MiniCam(
            self.width,
            self.height,
            self.fovy,
            self.fovx,
            self.znear,
            self.zfar,
            world_view,
            full_proj,
        )
        self.K = torch.tensor(
            [
                [self.fx, 0.0, self.width / 2.0],
                [0.0, self.fy, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

    def mesh_pose(self) -> torch.Tensor:
        return torch.inverse(self.c2w_blender)


def sorted_numeric(names: List[str], prefix: str) -> List[str]:
    def key_fn(name: str) -> int:
        return int(name.split(prefix, 1)[1])

    return sorted(names, key=key_fn)


def infer_sh_degree(extra_feature_count: int) -> int:
    coeffs = (extra_feature_count + 3) // 3
    return int(round(math.sqrt(coeffs))) - 1


def load_vertex_property_matrix(vertex, prefix: str) -> np.ndarray:
    names = sorted_numeric(
        [prop.name for prop in vertex.properties if prop.name.startswith(prefix)],
        prefix,
    )
    values = np.zeros((len(vertex["x"]), len(names)), dtype=np.float32)
    for idx, attr_name in enumerate(names):
        values[:, idx] = np.asarray(vertex[attr_name], dtype=np.float32)
    return values


def load_gaussian_frame(path: Path, device: str = "cuda") -> FrameGaussianModel:
    log_message(f"load_gaussian_frame: {path}")
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    opacities = np.asarray(vertex["opacity"], dtype=np.float32)[..., None]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
    features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
    features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"], dtype=np.float32)

    features_extra = load_vertex_property_matrix(vertex, "f_rest_")
    sh_degree = infer_sh_degree(features_extra.shape[1])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1))

    scales = load_vertex_property_matrix(vertex, "scale_")
    rots = load_vertex_property_matrix(vertex, "rot_")

    return FrameGaussianModel(
        xyz=torch.tensor(xyz, dtype=torch.float32, device=device),
        features_dc=torch.tensor(features_dc, dtype=torch.float32, device=device).transpose(1, 2).contiguous(),
        features_rest=torch.tensor(features_extra, dtype=torch.float32, device=device).transpose(1, 2).contiguous(),
        opacity=torch.tensor(opacities, dtype=torch.float32, device=device),
        scaling=torch.tensor(scales, dtype=torch.float32, device=device),
        rotation=torch.tensor(rots, dtype=torch.float32, device=device),
        sh_degree=sh_degree,
    )


def load_mesh_frame(path: Path, device: str = "cuda"):
    log_message(f"load_mesh_frame: {path}")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Trimesh):
        visual = mesh.visual
        vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
    else:
        # Handle Scene or other geometry types
        first_geometry = mesh.geometry[list(mesh.geometry.keys())[0]]
        visual = first_geometry.visual
        vertices = torch.tensor(np.asarray(first_geometry.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.asarray(first_geometry.faces), dtype=torch.int32, device=device)
    colors = np.asarray(visual.vertex_colors)
    if colors.shape[1] >= 3:
        colors = colors[:, :3].astype(np.float32) / 255.0
    else:
        colors = np.ones((vertices.shape[0], 3), dtype=np.float32)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    return vertices, faces, colors


def torch_image_to_qpixmap(image: torch.Tensor) -> QtGui.QPixmap:
    image_np = torch.clamp(image, 0.0, 1.0).detach().cpu().numpy()
    image_np = np.ascontiguousarray((image_np * 255).astype(np.uint8))
    h, w, _ = image_np.shape
    qimage = QtGui.QImage(image_np.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimage.copy())


def load_sequence_from_dir(directory: Path) -> Optional[SequenceInfo]:
    if not directory.exists() or not directory.is_dir():
        return None
    frame_paths = {}
    for child in sorted(directory.iterdir()):
        if not child.is_file():
            continue
        match = FRAME_RE.match(child.name)
        if match:
            frame_paths[int(match.group(1))] = child
    if not frame_paths:
        return None
    return SequenceInfo(root=directory, frames=frame_paths)


def load_json_dict(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log_exception(f"Failed to load JSON from {path}", exc)
        return None
    if not isinstance(data, dict):
        log_message(f"Expected a JSON object in {path}, got {type(data).__name__}")
        return None
    return data


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_dir: Path):
        super().__init__()
        log_message(f"MainWindow init start: start_dir={start_dir}")
        self.setWindowTitle("4DGS Player")
        if LOGO_PATH.exists():
            self.setWindowIcon(QtGui.QIcon(str(LOGO_PATH)))
        self.start_dir = start_dir
        self.path_dialog_root = start_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"Detected device={self.device}")
        if self.device != "cuda":
            log_message("CUDA unavailable; rendering may not work.")
            QtWidgets.QMessageBox.warning(
                self,
                "CUDA Required",
                "Current renderer path requires CUDA. The player can open, but rendering will be unavailable.",
            )
        self.pipeline = StaticPipeline()
        self.bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        self.glctx = dr.RasterizeCudaContext() if self.device == "cuda" else None
        self.gaussian_cache = LRUCache(max_items=4)
        self.mesh_cache = LRUCache(max_items=4)
        self.gaussian_sequences_by_path: Dict[str, SequenceInfo] = {}
        self.mesh_sequences_by_path: Dict[str, SequenceInfo] = {}
        self.mode = "Gaussian"
        self.gaussian_sequence: Optional[SequenceInfo] = None
        self.mesh_sequence: Optional[SequenceInfo] = None
        self.online_gaussian_checkpoint_dir: Optional[Path] = None
        self.online_gaussian_backend: Optional[OnlineGaussianBackend] = None
        self.online_default_frame_count = 200
        self.gaussian_frame_span = 1
        self.available_frame_ids: List[int] = []
        self.orbit = OrbitCamera(target=np.array([0.0, 0.0, 0.0], dtype=np.float32), distance=2.5, yaw=35.0, pitch=10.0)
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)

        self._build_ui()
        self._bind_events()
        self.on_mode_changed(self.mode_combo.currentText())
        self.recompute_frame_ids()
        self.render_current_frame()
        log_message("MainWindow init finished")

    def _build_ui(self):
        setup_main_window_ui(self)

    def _bind_events(self):
        self.play_button.clicked.connect(self.toggle_playback)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.gaussian_combo.currentIndexChanged.connect(self.on_sequence_changed)
        self.mesh_combo.currentIndexChanged.connect(self.on_sequence_changed)
        self.open_gaussian_button.clicked.connect(self.choose_gaussian_sequence_dir)
        self.open_mesh_button.clicked.connect(self.choose_mesh_sequence_dir)
        self.open_online_checkpoint_button.clicked.connect(self.choose_online_checkpoint_dir)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        self.frame_spin.valueChanged.connect(self.on_spin_changed)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        self.online_gaussian_check.toggled.connect(self.on_online_gaussian_toggled)
        self.online_scale_spin.valueChanged.connect(self.on_online_scale_changed)
        for view in (self.gaussian_view, self.mesh_view):
            view.orbit_changed.connect(self.on_orbit_changed)
            view.zoomed.connect(self.on_zoomed)
            view.resized.connect(self.render_current_frame)

    def dialog_directory(self, reference: Optional[Path] = None) -> str:
        if reference is not None and reference.exists():
            return str(reference)
        if self.path_dialog_root.exists():
            return str(self.path_dialog_root)
        return str(REPO_ROOT)

    def register_sequence(
        self,
        combo: QtWidgets.QComboBox,
        registry: Dict[str, SequenceInfo],
        directory: Path,
        kind: str,
    ) -> bool:
        sequence = load_sequence_from_dir(directory)
        if sequence is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Sequence",
                f"{directory} 中未找到 frame_XXXX.ply 序列。",
            )
            return False
        key = str(sequence.root)
        registry[key] = sequence
        if combo.findText(key) < 0:
            combo.addItem(key)
        combo.setCurrentText(key)
        self.path_dialog_root = directory
        log_message(f"Registered {kind} sequence: {directory}")
        return True

    def current_gaussian_sequence(self) -> Optional[SequenceInfo]:
        text = self.gaussian_combo.currentText()
        if not text:
            return None
        return self.gaussian_sequences_by_path.get(text)

    def current_mesh_sequence(self) -> Optional[SequenceInfo]:
        text = self.mesh_combo.currentText()
        if not text:
            return None
        return self.mesh_sequences_by_path.get(text)

    def update_gaussian_frame_span(self):
        if self.gaussian_sequence and self.gaussian_sequence.frame_ids:
            self.gaussian_frame_span = max(self.gaussian_sequence.frame_ids) + 1
        elif self.mesh_sequence and self.mesh_sequence.frame_ids:
            self.gaussian_frame_span = max(self.mesh_sequence.frame_ids) + 1
        else:
            self.gaussian_frame_span = self.online_default_frame_count

    def choose_gaussian_sequence_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Gaussian Sequence Directory",
            self.dialog_directory(self.gaussian_sequence.root if self.gaussian_sequence else None),
        )
        if directory:
            self.register_sequence(
                self.gaussian_combo,
                self.gaussian_sequences_by_path,
                Path(directory),
                "gaussian",
            )

    def choose_mesh_sequence_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Mesh Sequence Directory",
            self.dialog_directory(self.mesh_sequence.root if self.mesh_sequence else None),
        )
        if directory:
            self.register_sequence(
                self.mesh_combo,
                self.mesh_sequences_by_path,
                Path(directory),
                "mesh",
            )

    def choose_online_checkpoint_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Online Model Directory",
            self.dialog_directory(self.online_gaussian_checkpoint_dir),
        )
        if not directory:
            return
        checkpoint_dir = Path(directory)
        required_paths = [
            checkpoint_dir / "cfg_args.txt",
            checkpoint_dir / "point_cloud",
            checkpoint_dir / "deform",
        ]
        if not all(path.exists() for path in required_paths):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Online Model Directory",
                "请选择训练结果目录本身，而不是 output 总目录。\n\n"
                "目录内需要包含：\n"
                "- cfg_args.txt\n"
                "- point_cloud/\n"
                "- deform/",
            )
            return
        self.online_gaussian_checkpoint_dir = checkpoint_dir
        self.online_gaussian_backend = None
        self.path_dialog_root = checkpoint_dir
        self.update_gaussian_frame_span()
        log_message(f"Selected online checkpoint: {checkpoint_dir}")
        self.recompute_frame_ids()
        self.render_current_frame()

    def on_sequence_changed(self):
        log_message("on_sequence_changed start")
        self.gaussian_sequence = self.current_gaussian_sequence()
        self.mesh_sequence = self.current_mesh_sequence()
        self.online_gaussian_backend = None
        self.update_gaussian_frame_span()
        self.auto_frame_scene()
        self.recompute_frame_ids()
        self.render_current_frame()

    def on_online_gaussian_toggled(self, enabled: bool):
        if enabled and self.ensure_online_gaussian_backend() is None:
            self.online_gaussian_check.blockSignals(True)
            self.online_gaussian_check.setChecked(False)
            self.online_gaussian_check.blockSignals(False)
        self.recompute_frame_ids()
        self.render_current_frame()

    def on_online_scale_changed(self, _: float):
        if self.online_gaussian_check.isChecked():
            self.render_current_frame()

    def ensure_online_gaussian_backend(self) -> Optional[OnlineGaussianBackend]:
        if self.online_gaussian_backend is not None:
            return self.online_gaussian_backend
        checkpoint_dir = self.online_gaussian_checkpoint_dir
        if checkpoint_dir is None:
            return None
        cfg_path = checkpoint_dir / "cfg_args.txt"
        config = load_json_dict(cfg_path)
        if config is None:
            return None
        try:
            start = time.perf_counter()
            backend = OnlineGaussianBackend(checkpoint_dir, config, device=self.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.online_gaussian_backend = backend
            log_message(
                f"Online Gaussian backend ready from {checkpoint_dir} "
                f"in {elapsed_ms:.1f} ms"
            )
            return backend
        except Exception as exc:
            log_exception(f"Failed to initialize online Gaussian backend from {checkpoint_dir}", exc)
            return None

    def on_mode_changed(self, mode: str):
        self.mode = mode
        self.gaussian_view.setVisible(mode in {"Gaussian", "Split"})
        self.mesh_view.setVisible(mode in {"Mesh", "Split"})
        self.render_container.setVisible(True)
        self.frame_slider.setEnabled(True)
        self.frame_spin.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.recompute_frame_ids()
        self.render_current_frame()

    def recompute_frame_ids(self):
        has_online = (
            self.online_gaussian_check.isChecked()
            and self.online_gaussian_checkpoint_dir is not None
        )
        gaussian_ids = set(self.gaussian_sequence.frame_ids) if self.gaussian_sequence else set()
        mesh_ids = set(self.mesh_sequence.frame_ids) if self.mesh_sequence else set()
        if has_online:
            if gaussian_ids:
                online_ids = gaussian_ids
            elif mesh_ids:
                online_ids = mesh_ids
            else:
                online_ids = set(range(self.online_default_frame_count))
        else:
            online_ids = gaussian_ids
        if self.mode == "Gaussian":
            frame_ids = sorted(online_ids)
        elif self.mode == "Mesh":
            frame_ids = sorted(mesh_ids)
        else:
            frame_ids = sorted(online_ids & mesh_ids) if mesh_ids else sorted(online_ids)
        self.available_frame_ids = frame_ids
        max_index = max(0, len(frame_ids) - 1)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(0, max_index)
        self.frame_spin.setRange(0, max_index)
        self.frame_slider.setValue(min(self.frame_slider.value(), max_index))
        self.frame_spin.setValue(min(self.frame_spin.value(), max_index))
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        if frame_ids:
            self.status_label.setText(f"{self.mode}: {len(frame_ids)} frames")
        elif has_online:
            self.status_label.setText("已选择 online 模型，但当前模式没有可用帧")
        elif not self.gaussian_sequence and not self.mesh_sequence:
            self.status_label.setText("请先选择 Online Model Dir 或 Gaussian / Mesh 序列目录")
        else:
            self.status_label.setText("No overlapping frames for current mode")

    def on_slider_changed(self, value: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        self.render_current_frame()

    def on_spin_changed(self, value: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        self.render_current_frame()

    def on_fps_changed(self, value: int):
        if self.play_timer.isActive():
            self.play_timer.start(max(1, int(1000 / value)))

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.play_timer.start(max(1, int(1000 / self.fps_spin.value())))
            self.play_button.setText("Pause")

    def advance_frame(self):
        if not self.available_frame_ids:
            return
        next_index = (self.frame_slider.value() + 1) % len(self.available_frame_ids)
        self.frame_slider.setValue(next_index)

    def on_orbit_changed(self, dx: float, dy: float, mode: str):
        if mode == "pan":
            self.orbit.pan(dx, dy)
        else:
            self.orbit.orbit(dx, dy)
        self.render_current_frame()

    def on_zoomed(self, delta: float):
        self.orbit.zoom(delta)
        self.render_current_frame()

    def get_active_frame_id(self) -> Optional[int]:
        if not self.available_frame_ids:
            return None
        index = min(self.frame_slider.value(), len(self.available_frame_ids) - 1)
        return self.available_frame_ids[index]

    def load_gaussian(self, frame_id: int) -> Optional[FrameGaussianModel]:
        if not self.gaussian_sequence or frame_id not in self.gaussian_sequence.frames:
            return None
        cache_key = (str(self.gaussian_sequence.root), frame_id)
        return self.gaussian_cache.get(
            cache_key,
            lambda: load_gaussian_frame(self.gaussian_sequence.get(frame_id), device=self.device),
        )

    def load_mesh(self, frame_id: int):
        if not self.mesh_sequence or frame_id not in self.mesh_sequence.frames:
            return None
        cache_key = (str(self.mesh_sequence.root), frame_id)
        return self.mesh_cache.get(
            cache_key,
            lambda: load_mesh_frame(self.mesh_sequence.get(frame_id), device=self.device),
        )

    def auto_frame_scene(self):
        bounds_min = None
        bounds_max = None

        if self.gaussian_sequence and self.gaussian_sequence.frame_ids:
            first_frame = self.gaussian_sequence.frame_ids[0]
            gaussian = self.load_gaussian(first_frame)
            if gaussian is not None:
                xyz = gaussian.get_xyz.detach().cpu().numpy()
                bounds_min = xyz.min(axis=0)
                bounds_max = xyz.max(axis=0)

        if (bounds_min is None or bounds_max is None) and self.mesh_sequence and self.mesh_sequence.frame_ids:
            first_frame = self.mesh_sequence.frame_ids[0]
            mesh = self.load_mesh(first_frame)
            if mesh is not None:
                vertices = mesh[0].detach().cpu().numpy()
                bounds_min = vertices.min(axis=0)
                bounds_max = vertices.max(axis=0)

        if bounds_min is None or bounds_max is None:
            log_message("auto_frame_scene skipped: no geometry available")
            return

        center = (bounds_min + bounds_max) * 0.5
        extent = np.maximum(bounds_max - bounds_min, 1e-3)
        radius = float(np.linalg.norm(extent) * 0.5)
        distance = max(0.5, radius / math.tan(math.radians(self.orbit.fovy_deg) * 0.5) * 1.25)
        self.orbit.target = center.astype(np.float32)
        self.orbit.distance = distance
        self.orbit.clamp()
        log_message(
            f"auto_frame_scene center={self.orbit.target.tolist()} distance={self.orbit.distance:.4f}"
        )

    def render_current_frame(self):
        log_message(
            f"render_current_frame start: mode={self.mode} available={len(self.available_frame_ids)}"
        )
        frame_id = self.get_active_frame_id()
        if frame_id is None:
            self.gaussian_view.setText("No Gaussian frame")
            self.mesh_view.setText("No Mesh frame")
            return

        log_message(f"render_current_frame frame_id={frame_id}")

        if self.mode in {"Gaussian", "Split"}:
            try:
                if self.online_gaussian_check.isChecked():
                    if self.ensure_online_gaussian_backend() is not None:
                        self.render_online_gaussian_view(frame_id)
                    else:
                        self.gaussian_view.clear()
                        self.gaussian_view.setText("Online Gaussian backend unavailable")
                else:
                    gaussian = self.load_gaussian(frame_id)
                    if gaussian is not None:
                        self.render_gaussian_view(frame_id, gaussian)
                    else:
                        self.gaussian_view.clear()
                        self.gaussian_view.setText(f"Missing Gaussian frame {frame_id}")
            except Exception as exc:
                log_exception(f"Gaussian render failed for frame {frame_id}", exc)
                self.gaussian_view.clear()
                self.gaussian_view.setText(f"Gaussian render failed: {frame_id}")
        if self.mode in {"Mesh", "Split"}:
            try:
                mesh = self.load_mesh(frame_id)
                if mesh is not None:
                    self.render_mesh_view(frame_id, mesh)
                else:
                    self.mesh_view.clear()
                    self.mesh_view.setText(f"Missing Mesh frame {frame_id}")
            except Exception as exc:
                log_exception(f"Mesh render failed for frame {frame_id}", exc)
                self.mesh_view.clear()
                self.mesh_view.setText(f"Mesh render failed: {frame_id}")

    def gaussian_render_resolution(self, render_scale: float = 1.0) -> tuple[int, int, int, int]:
        view_width = max(64, self.gaussian_view.width())
        view_height = max(64, self.gaussian_view.height())
        scaled_width = max(64, int(round(view_width * render_scale)))
        scaled_height = max(64, int(round(view_height * render_scale)))
        return view_width, view_height, scaled_width, scaled_height

    def set_render_view_image(
        self,
        view: RenderView,
        image: torch.Tensor,
        tooltip: str,
    ) -> None:
        pixmap = torch_image_to_qpixmap(image)
        view.setPixmap(
            pixmap.scaled(
                view.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
        view.setToolTip(tooltip)

    def render_gaussian_view(self, frame_id: int, gaussian: FrameGaussianModel):
        log_message(f"render_gaussian_view frame_id={frame_id}")
        view_width, view_height, render_width, render_height = self.gaussian_render_resolution()
        camera = PlayerCamera(
            render_width,
            render_height,
            self.orbit,
            device=self.device,
        )
        zeros_xyz = torch.zeros_like(gaussian.get_xyz)
        zeros_rot = torch.zeros_like(gaussian.get_rotation)
        zeros_scale = torch.zeros_like(gaussian.get_scaling)
        render_pkg = gaussian_render(
            camera.mini_cam,
            gaussian,
            self.pipeline,
            self.bg_color,
            zeros_xyz,
            zeros_rot,
            zeros_scale,
            False,
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0).permute(1, 2, 0)
        visible = int(render_pkg["visibility_filter"].sum().item())
        radii = render_pkg["radii"]
        radii_max = float(radii.max().item()) if radii.numel() > 0 else 0.0
        log_message(
            "gaussian image stats "
            f"frame={frame_id} min={float(image.min()):.4f} "
            f"max={float(image.max()):.4f} mean={float(image.mean()):.4f} "
            f"visible={visible} radii_max={radii_max:.4f} "
            f"opacity_mean={float(gaussian.get_opacity.mean()):.4f} "
            f"scale_mean={float(gaussian.get_scaling.mean()):.4f} "
            f"fov=({math.degrees(camera.fovx):.2f},{math.degrees(camera.fovy):.2f}) "
            f"view_res={view_width}x{view_height} "
            f"render_res={render_width}x{render_height}"
        )
        self.set_render_view_image(self.gaussian_view, image, f"Gaussian frame {frame_id}")

    def render_online_gaussian_view(self, frame_id: int):
        log_message(f"render_online_gaussian_view frame_id={frame_id}")
        backend = self.ensure_online_gaussian_backend()
        if backend is None:
            self.gaussian_view.setText("Online Gaussian backend unavailable")
            return
        render_scale = float(self.online_scale_spin.value())
        view_width, view_height, render_width, render_height = self.gaussian_render_resolution(
            render_scale
        )
        camera = PlayerCamera(
            render_width,
            render_height,
            self.orbit,
            device=self.device,
        )
        start = time.perf_counter()
        render_pkg = backend.render(
            camera,
            frame_id=frame_id,
            frame_span=self.gaussian_frame_span,
            pipeline=self.pipeline,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        image = torch.clamp(render_pkg["render"], 0.0, 1.0).permute(1, 2, 0)
        visible = int(render_pkg["visibility_filter"].sum().item())
        radii = render_pkg["radii"]
        radii_max = float(radii.max().item()) if radii.numel() > 0 else 0.0
        log_message(
            "online gaussian image stats "
            f"frame={frame_id} min={float(image.min()):.4f} "
            f"max={float(image.max()):.4f} mean={float(image.mean()):.4f} "
            f"visible={visible} radii_max={radii_max:.4f} "
            f"fov=({math.degrees(camera.fovx):.2f},{math.degrees(camera.fovy):.2f}) "
            f"view_res={view_width}x{view_height} "
            f"render_res={render_width}x{render_height} "
            f"scale={render_scale:.2f} "
            f"dt_ms={elapsed_ms:.2f}"
        )
        self.set_render_view_image(self.gaussian_view, image, f"Online Gaussian frame {frame_id}")

    def render_mesh_view(self, frame_id: int, mesh):
        log_message(f"render_mesh_view frame_id={frame_id}")
        if self.glctx is None:
            self.mesh_view.setText("CUDA mesh rendering unavailable")
            return
        width = max(64, self.mesh_view.width())
        height = max(64, self.mesh_view.height())
        camera = PlayerCamera(width, height, self.orbit, device=self.device)
        vertices, faces, colors = mesh
        image = render_mesh(
            self.glctx,
            vertices,
            faces,
            colors,
            camera.mesh_pose(),
            camera.K,
            resolution=[height, width],
            whitebackground=True,
        ).permute(1, 2, 0)
        log_message(
            f"mesh image stats frame={frame_id} min={float(image.min()):.4f} max={float(image.max()):.4f}"
        )
        self.set_render_view_image(self.mesh_view, image, f"Mesh frame {frame_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4DGS Player")
    parser.add_argument(
        "--start-dir",
        default=str(REPO_ROOT / "output"),
        help="Initial directory used by file/folder pickers.",
    )
    parser.add_argument(
        "--enable-log",
        action="store_true",
        help="Enable writing startup/runtime logs to logs/4dgsplayer.log.",
    )
    return parser


def main():
    log_message("Application bootstrap start")
    try:
        app = QtWidgets.QApplication(sys.argv)
        log_message("QApplication created")
        log_message(
            f"Env: KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')} "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}"
        )

        def qt_excepthook(exc_type, exc_value, exc_tb):
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            log_message(f"Unhandled exception\n{tb}")
            show_fatal_error(
                "4DGS Player crashed. See logs/4dgsplayer.log for details.",
                tb,
            )

        sys.excepthook = qt_excepthook

        log_message("Building argparse parser")
        parser = build_parser()
        log_message("Parsing command line arguments")
        args = parser.parse_args(sys.argv[1:])
        start_dir = Path(args.start_dir)
        log_message(f"Arguments parsed: start_dir={start_dir}")
        window = MainWindow(start_dir=start_dir)
        window.resize(1600, 900)
        log_message("MainWindow resized")
        window.show()
        log_message("MainWindow shown")
        exit_code = app.exec()
        log_message(f"Application exit code={exit_code}")
        sys.exit(exit_code)
    except Exception as exc:
        log_exception("Fatal error during application startup", exc)
        show_fatal_error(
            "4DGS Player failed during startup. See logs/4dgsplayer.log for details.",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        raise


if __name__ == "__main__":
    main()
