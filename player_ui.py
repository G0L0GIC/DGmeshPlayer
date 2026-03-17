from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class RenderView(QtWidgets.QLabel):
    orbit_changed = QtCore.Signal(float, float, str)
    zoomed = QtCore.Signal(float)
    resized = QtCore.Signal()

    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
        self.setText(title)
        self._drag_pos: QtCore.QPoint | None = None
        self._drag_mode = "orbit"

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self._drag_pos = event.position().toPoint()
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._drag_mode = "pan"
        else:
            self._drag_mode = "orbit"
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_pos is None:
            return super().mouseMoveEvent(event)
        pos = event.position().toPoint()
        delta = pos - self._drag_pos
        self._drag_pos = pos
        self.orbit_changed.emit(float(delta.x()), float(delta.y()), self._drag_mode)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        self.zoomed.emit(float(event.angleDelta().y()))
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self.resized.emit()
        super().resizeEvent(event)


def setup_main_window_ui(window: QtWidgets.QMainWindow) -> None:
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)

    controls = QtWidgets.QGridLayout()
    layout.addLayout(controls)

    window.gaussian_combo = QtWidgets.QComboBox()
    window.mesh_combo = QtWidgets.QComboBox()
    window.open_gaussian_button = QtWidgets.QPushButton("Open GS Dir")
    window.open_mesh_button = QtWidgets.QPushButton("Open Mesh Dir")
    window.mode_combo = QtWidgets.QComboBox()
    window.mode_combo.addItems(["Gaussian", "Mesh", "Split"])
    window.mode_combo.setCurrentText("Gaussian")
    window.online_gaussian_check = QtWidgets.QCheckBox("Online GS")
    window.online_gaussian_check.setChecked(True)
    window.open_online_checkpoint_button = QtWidgets.QPushButton("Open Online Model Dir")
    window.online_scale_spin = QtWidgets.QDoubleSpinBox()
    window.online_scale_spin.setRange(0.25, 1.0)
    window.online_scale_spin.setSingleStep(0.05)
    window.online_scale_spin.setDecimals(2)
    window.online_scale_spin.setValue(1.0)
    window.online_scale_spin.setSuffix("x")
    window.play_button = QtWidgets.QPushButton("Play")
    window.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    window.frame_spin = QtWidgets.QSpinBox()
    window.fps_spin = QtWidgets.QSpinBox()
    window.fps_spin.setRange(1, 60)
    window.fps_spin.setValue(24)
    window.status_label = QtWidgets.QLabel("No sequence loaded")

    controls.addWidget(window.online_gaussian_check, 0, 0)
    controls.addWidget(window.open_online_checkpoint_button, 0, 1, 1, 3)
    controls.addWidget(QtWidgets.QLabel("GS Scale"), 0, 4)
    controls.addWidget(window.online_scale_spin, 0, 5)
    controls.addWidget(QtWidgets.QLabel("Mode"), 0, 6)
    controls.addWidget(window.mode_combo, 0, 7)
    controls.addWidget(window.play_button, 0, 8)
    controls.addWidget(QtWidgets.QLabel("FPS"), 0, 9)
    controls.addWidget(window.fps_spin, 0, 10)

    controls.addWidget(QtWidgets.QLabel("Gaussian"), 1, 0)
    controls.addWidget(window.gaussian_combo, 1, 1, 1, 2)
    controls.addWidget(window.open_gaussian_button, 1, 3)
    controls.addWidget(QtWidgets.QLabel("Mesh"), 1, 4)
    controls.addWidget(window.mesh_combo, 1, 5, 1, 2)
    controls.addWidget(window.open_mesh_button, 1, 7)

    controls.addWidget(QtWidgets.QLabel("Frame"), 3, 0)
    controls.addWidget(window.frame_slider, 3, 1, 1, 9)
    controls.addWidget(window.frame_spin, 3, 10)

    window.render_container = QtWidgets.QWidget()
    view_layout = QtWidgets.QHBoxLayout(window.render_container)
    view_layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(window.render_container, 1)

    window.gaussian_view = RenderView("Gaussian")
    window.mesh_view = RenderView("Mesh")
    view_layout.addWidget(window.gaussian_view, 1)
    view_layout.addWidget(window.mesh_view, 1)

    layout.addWidget(window.status_label)
