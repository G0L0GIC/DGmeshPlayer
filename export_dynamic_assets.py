import os
import sys
import datetime
import json
import os.path as osp
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import imageio
import trimesh
import nvdiffrast.torch as dr
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene import DeformModelNormalSep as deform_model_sep
from scene import AppearanceModel as appearance_model
from gaussian_renderer import render
from utils.renderer import mesh_renderer
from utils.system_utils import load_config_from_file, mkdir_p


def load_checkpoint_config(checkpoint_dir):
    cfg_path = osp.join(checkpoint_dir, "cfg_args.txt")
    if not osp.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_export_args(base_config, cli_args):
    combined = dict(base_config)
    override_keys = [
        "config",
        "start_checkpoint",
        "output_dir",
        "frame_num",
        "iteration",
        "export_stride",
        "time_mode",
        "preview_count",
        "preview_seed",
        "export_gaussians",
        "export_mesh",
    ]
    for key in override_keys:
        if hasattr(cli_args, key):
            combined[key] = getattr(cli_args, key)
    return Namespace(**combined)


def load_export_models(dataset, opt, checkpoint, iteration):
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    deform = deform_model(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform",
    )
    deform_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_normal",
    )

    gaussians.load_ply(checkpoint, iteration=iteration)
    deform.load_weights(checkpoint, iteration=iteration)
    deform_normal.load_weights(checkpoint, iteration=iteration)
    return gaussians, deform, deform_normal


def load_mesh_export_models(dataset, checkpoint, iteration):
    deform_back = deform_model(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_back",
    )
    appearance = appearance_model(is_blender=dataset.is_blender)
    deform_back.load_weights(checkpoint, iteration=iteration)
    appearance.load_weights(checkpoint, iteration=iteration)
    return deform_back, appearance


def build_export_schedule(scene, time_mode, frame_num):
    if time_mode == "test":
        cameras = scene.getTestCameras().copy()
        frame_ids = list(range(len(cameras)))
        times = [float(cam.fid.item()) for cam in cameras]
        return frame_ids, times, cameras

    frame_num = int(frame_num)
    frame_ids = list(range(frame_num))
    times = [i / frame_num for i in frame_ids]
    return frame_ids, times, None


def sample_preview_indices(export_indices, preview_count, preview_seed):
    if preview_count <= 0 or not export_indices:
        return set()
    rng = np.random.default_rng(preview_seed)
    sampled = rng.choice(
        export_indices,
        size=min(preview_count, len(export_indices)),
        replace=False,
    )
    return {int(i) for i in sampled.tolist()}


def resolve_preview_camera(scene, preview_cameras, frame_id, time_value):
    if preview_cameras is not None:
        return preview_cameras[frame_id]
    viewpoint_cam = scene.getTestCameras()[0]
    viewpoint_cam.fid = torch.tensor([time_value], device="cuda")
    return viewpoint_cam


def resolve_output_dirs(output_dir, export_gaussians, export_mesh):
    gaussian_dir = None
    mesh_dir = None
    gaussian_preview_dir = None
    mesh_preview_dir = None
    if export_gaussians:
        gaussian_dir = osp.join(output_dir, "dynamic_gaussians_offline")
        gaussian_preview_dir = osp.join(gaussian_dir, "preview_renders")
    if export_mesh:
        mesh_dir = osp.join(output_dir, "dynamic_mesh_offline")
        mesh_preview_dir = osp.join(mesh_dir, "preview_renders")

    for directory in (
        output_dir,
        gaussian_dir,
        mesh_dir,
        gaussian_preview_dir,
        mesh_preview_dir,
    ):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
    return gaussian_dir, mesh_dir, gaussian_preview_dir, mesh_preview_dir


@torch.no_grad()
def save_dynamic_gaussian_ply(
    path,
    gaussians,
    deform,
    deform_normal=None,
    t=0.0,
    d_xyz=None,
    d_rotation=None,
    d_scaling=None,
    d_normal=None,
):
    """Export one deformed dynamic Gaussian frame to PLY.

    This function intentionally lives in the export stage instead of the
    Gaussian model class, so export responsibility stays inside the
    train / export / player split.
    """
    mkdir_p(os.path.dirname(path))

    N = gaussians.get_xyz.shape[0]
    t_val = float(t.view(-1)[0].item()) if torch.is_tensor(t) else float(t)

    need_recompute = (
        d_xyz is None
        or d_rotation is None
        or d_scaling is None
        or (torch.is_tensor(d_xyz) and d_xyz.shape[0] != N)
        or (torch.is_tensor(d_rotation) and d_rotation.shape[0] != N)
        or (torch.is_tensor(d_scaling) and d_scaling.shape[0] != N)
        or (
            d_normal is not None
            and torch.is_tensor(d_normal)
            and d_normal.shape[0] != N
        )
    )
    if need_recompute:
        time_input = torch.ones(N, 1, device="cuda") * t_val
        d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz, time_input)
        if deform_normal is not None:
            d_normal = deform_normal.step(gaussians.get_xyz, time_input)

    if not torch.is_tensor(d_xyz):
        d_xyz = torch.zeros_like(gaussians._xyz)
    if not torch.is_tensor(d_rotation):
        d_rotation = torch.zeros_like(gaussians._rotation)
    if not torch.is_tensor(d_scaling):
        d_scaling = torch.zeros_like(gaussians._scaling)
    if d_normal is None or not torch.is_tensor(d_normal):
        d_normal = torch.zeros_like(gaussians._normal)

    xyz = (gaussians._xyz + d_xyz).detach().cpu().numpy()
    normals = torch.nn.functional.normalize(gaussians._normal + d_normal, dim=1)
    normals = normals.detach().cpu().numpy()
    f_dc = (
        gaussians._features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        gaussians._features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = (gaussians.get_scaling + d_scaling).detach().cpu().numpy()
    rotation = (gaussians.get_rotation + d_rotation).detach().cpu().numpy()

    dtype_full = [
        (attribute, "f4")
        for attribute in gaussians.construct_list_of_attributes()
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    vertex_el = PlyElement.describe(elements, "vertex")

    density_thres_np = np.array([gaussians.density_thres_param.detach().cpu()])
    dens_thres = np.empty(1, dtype=[("density_thres", "f4")])
    dens_thres[:] = list(map(tuple, [density_thres_np]))

    gaussian_center_np = gaussians.gaussian_center.view(-1).detach().cpu().numpy()
    gaussian_center = np.empty(
        1,
        dtype=[
            ("gaussian_center_x", "f4"),
            ("gaussian_center_y", "f4"),
            ("gaussian_center_z", "f4"),
        ],
    )
    gaussian_center[:] = list(map(tuple, [gaussian_center_np]))

    gaussian_scale_np = np.array([gaussians.gaussian_scale.view(-1).detach().cpu().numpy()])
    gaussian_scale = np.empty(1, dtype=[("gaussian_scale", "f4")])
    gaussian_scale[:] = list(map(tuple, [gaussian_scale_np]))

    export_meta = np.empty(
        1,
        dtype=[
            ("export_version", "i4"),
            ("scale_in_storage_domain", "i4"),
            ("rotation_already_deformed", "i4"),
        ],
    )
    export_meta[:] = [(2, 0, 1)]

    PlyData(
        [
            vertex_el,
            PlyElement.describe(dens_thres, "density_thres"),
            PlyElement.describe(gaussian_center, "gaussian_center"),
            PlyElement.describe(gaussian_scale, "gaussian_scale"),
            PlyElement.describe(export_meta, "dynamic_export_meta"),
        ]
    ).write(path)


@torch.no_grad()
def export_dynamic_gaussians(
    dataset,
    opt,
    pipe,
    checkpoint,
    output_dir,
    frame_num,
    stride,
    time_mode,
    iteration,
    preview_count,
    preview_seed,
    export_gaussians,
    export_mesh,
):
    if not export_gaussians and not export_mesh:
        export_gaussians = True

    gaussians, deform, deform_normal = load_export_models(
        dataset,
        opt,
        checkpoint,
        iteration,
    )
    deform_back = None
    appearance = None
    if export_mesh:
        deform_back, appearance = load_mesh_export_models(dataset, checkpoint, iteration)
    scene = None

    os.makedirs(output_dir, exist_ok=True)
    (
        gaussian_output_dir,
        mesh_output_dir,
        gaussian_preview_dir,
        mesh_preview_dir,
    ) = resolve_output_dirs(
        output_dir,
        export_gaussians,
        export_mesh,
    )

    needs_scene = time_mode == "test" or preview_count > 0
    if needs_scene:
        try:
            scene = Scene(dataset, gaussians, shuffle=False)
        except Exception as exc:
            if time_mode == "test":
                raise
            print(
                "Warning: failed to initialize Scene for preview renders.\n"
                f"  source_path: {getattr(dataset, 'source_path', '<missing>')}\n"
                f"  data_type: {getattr(dataset, 'data_type', '<auto>')}\n"
                f"  reason: {exc}\n"
                "  This usually means cfg_args.txt points to a training-time dataset path "
                "that does not exist on the current machine, so scene type auto-detection fails.\n"
                "  Continuing with preview_count=0."
            )
            preview_count = 0

    frame_ids, times, preview_cameras = build_export_schedule(scene, time_mode, frame_num)

    stride = max(1, int(stride))
    export_indices = [idx for idx in frame_ids if idx % stride == 0]
    preview_indices = sample_preview_indices(export_indices, preview_count, preview_seed)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    mesh_preview_glctx = None
    if preview_count > 0 and mesh_output_dir is not None and scene is not None:
        try:
            mesh_preview_glctx = dr.RasterizeGLContext()
        except Exception as exc:
            print(f"Warning: failed to create mesh preview rasterizer: {exc}. Mesh previews will be skipped.")
    exported = 0
    gaussian_exported = 0
    mesh_exported = 0
    for idx, t in tqdm(list(zip(frame_ids, times)), desc="Exporting dynamic assets"):
        if idx % stride != 0:
            continue
        N = gaussians.get_xyz.shape[0]
        time_input = torch.ones(N, 1, device="cuda") * t
        d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz, time_input)
        d_normal = deform_normal.step(gaussians.get_xyz, time_input)
        if gaussian_output_dir is not None:
            save_dynamic_gaussian_ply(
                path=osp.join(gaussian_output_dir, f"frame_{idx:04d}.ply"),
                gaussians=gaussians,
                deform=deform,
                deform_normal=deform_normal,
                t=t,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
                d_normal=d_normal,
            )
            gaussian_exported += 1
        if mesh_output_dir is not None:
            frame_time = torch.tensor([t], device="cuda")
            verts, faces, vtx_color = mesh_renderer(
                glctx=None,
                gaussians=gaussians,
                d_xyz=d_xyz,
                d_normal=d_normal,
                fid=frame_time,
                deform_back=deform_back,
                appearance=appearance,
                freeze_pos=False,
                whitebackground=dataset.white_background,
            )
            vtx_color = torch.clamp(vtx_color, 0.0, 1.0) * 255
            save_mesh = trimesh.Trimesh(
                vertices=verts.detach().cpu().numpy(),
                faces=faces.detach().cpu().numpy(),
                vertex_colors=vtx_color.detach().cpu().numpy().astype(np.uint8),
            )
            save_mesh.export(osp.join(mesh_output_dir, f"frame_{idx:04d}.ply"))
            mesh_exported += 1
        if idx in preview_indices:
            viewpoint_cam = resolve_preview_camera(scene, preview_cameras, idx, t)

            if gaussian_preview_dir is not None:
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    dataset.is_6dof,
                )
                gs_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gs_image = gs_image.permute(1, 2, 0).detach().cpu().numpy()
                imageio.imwrite(
                    osp.join(gaussian_preview_dir, f"frame_{idx:04d}.png"),
                    (gs_image * 255).astype(np.uint8),
                )

            if mesh_preview_dir is not None and mesh_preview_glctx is not None:
                frame_time = torch.tensor([t], device="cuda")
                _, mesh_image, _, _, _ = mesh_renderer(
                    glctx=mesh_preview_glctx,
                    gaussians=gaussians,
                    d_xyz=d_xyz,
                    d_normal=d_normal,
                    fid=frame_time,
                    deform_back=deform_back,
                    appearance=appearance,
                    freeze_pos=False,
                    whitebackground=dataset.white_background,
                    viewpoint_cam=viewpoint_cam,
                )
                mesh_image = torch.clamp(mesh_image, 0.0, 1.0)
                mesh_image = mesh_image.permute(1, 2, 0).detach().cpu().numpy()
                imageio.imwrite(
                    osp.join(mesh_preview_dir, f"frame_{idx:04d}.png"),
                    (mesh_image * 255).astype(np.uint8),
                )
        exported += 1

    exported_items = []
    if gaussian_output_dir is not None:
        exported_items.append(f"gaussians={gaussian_exported} -> {gaussian_output_dir}")
    if mesh_output_dir is not None:
        exported_items.append(f"meshes={mesh_exported} -> {mesh_output_dir}")
    print("Done. " + ", ".join(exported_items))


def main():
    parser = ArgumentParser(description="Offline dynamic asset exporter")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_num", type=int, default=200)
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to export. -1 means latest.",
    )
    parser.add_argument(
        "--export_stride",
        type=int,
        default=1,
        choices=[1, 2, 5],
        help="Export every N frames (supported: 1, 2, 5).",
    )
    parser.add_argument(
        "--time_mode",
        type=str,
        default="uniform",
        choices=["uniform", "test"],
        help="uniform: t=i/frame_num, test: use test camera fid values.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=5,
        help="Number of random preview renders to save alongside exported PLYs.",
    )
    parser.add_argument(
        "--preview_seed",
        type=int,
        default=0,
        help="Random seed for preview frame sampling.",
    )
    parser.add_argument(
        "--export_gaussians",
        action="store_true",
        help="Export per-frame dynamic Gaussian PLYs.",
    )
    parser.add_argument(
        "--export_mesh",
        action="store_true",
        help="Export per-frame dynamic mesh PLYs.",
    )

    args = parser.parse_args(sys.argv[1:])

    checkpoint_config = load_checkpoint_config(args.start_checkpoint)
    if args.config and osp.exists(args.config):
        config_data = load_config_from_file(args.config)
        checkpoint_config.update(config_data)
    elif args.config:
        print(f"Configuration file {args.config} not found. Falling back to checkpoint cfg_args.txt if available.")

    if checkpoint_config:
        args = merge_export_args(checkpoint_config, args)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    if not args.export_gaussians and not args.export_mesh:
        args.export_gaussians = True
        args.export_mesh = True

    # In offline export mode, write outputs under checkpoint by default.
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"dynamic_assets_offline_{timestamp}"
        args.output_dir = osp.join(args.start_checkpoint, folder_name)

    export_dynamic_gaussians(
        dataset=lp,
        opt=op,
        pipe=pp,
        checkpoint=args.start_checkpoint,
        output_dir=args.output_dir,
        frame_num=args.frame_num,
        stride=args.export_stride,
        time_mode=args.time_mode,
        iteration=args.iteration,
        preview_count=args.preview_count,
        preview_seed=args.preview_seed,
        export_gaussians=args.export_gaussians,
        export_mesh=args.export_mesh,
    )


if __name__ == "__main__":
    main()
