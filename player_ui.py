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
        self.setText(title)
        self._drag_pos: QtCore.QPoint | None = None
        self._drag_mode = "orbit"
        self.apply_theme("light")

    def apply_theme(self, theme: str):
        if theme == "light":
            self.setStyleSheet(
                "background:#f4f8ff; color:#35506c; border:1px solid rgba(85,120,168,0.28); border-radius:14px;"
            )
        else:
            self.setStyleSheet(
                "background:#0d1117; color:#d7dde8; border:1px solid rgba(112,145,193,0.18); border-radius:14px;"
            )

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


def _make_render_card(title: str, subtitle: str, view: RenderView) -> QtWidgets.QWidget:
    card = QtWidgets.QFrame()
    card.setObjectName("renderCard")
    layout = QtWidgets.QVBoxLayout(card)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    title_label = QtWidgets.QLabel(title)
    title_label.setObjectName("renderCardTitle")
    subtitle_label = QtWidgets.QLabel(subtitle)
    subtitle_label.setObjectName("renderCardSubtitle")

    header = QtWidgets.QWidget()
    header_layout = QtWidgets.QVBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(2)
    header_layout.addWidget(title_label)
    header_layout.addWidget(subtitle_label)

    layout.addWidget(header)
    layout.addWidget(view, 1)
    return card


def _build_hidden_legacy_controls(window: QtWidgets.QMainWindow) -> QtWidgets.QWidget:
    panel = QtWidgets.QFrame()
    panel.setObjectName("legacyControlsPanel")
    panel.setVisible(False)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(14)

    title = QtWidgets.QLabel("Legacy Controls")
    title.setObjectName("sidePanelTitle")
    helper = QtWidgets.QLabel("Fallback Qt widgets used when the React panel is unavailable.")
    helper.setWordWrap(True)
    helper.setObjectName("sidePanelSubtitle")
    layout.addWidget(title)
    layout.addWidget(helper)

    form = QtWidgets.QGridLayout()
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(10)
    layout.addLayout(form)

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
    window.status_label.setWordWrap(True)

    form.addWidget(window.online_gaussian_check, 0, 0)
    form.addWidget(window.open_online_checkpoint_button, 0, 1, 1, 2)
    form.addWidget(QtWidgets.QLabel("GS Scale"), 1, 0)
    form.addWidget(window.online_scale_spin, 1, 1)
    form.addWidget(QtWidgets.QLabel("Mode"), 2, 0)
    form.addWidget(window.mode_combo, 2, 1)
    form.addWidget(window.play_button, 2, 2)
    form.addWidget(QtWidgets.QLabel("FPS"), 3, 0)
    form.addWidget(window.fps_spin, 3, 1)
    form.addWidget(QtWidgets.QLabel("Gaussian"), 4, 0)
    form.addWidget(window.gaussian_combo, 4, 1)
    form.addWidget(window.open_gaussian_button, 4, 2)
    form.addWidget(QtWidgets.QLabel("Mesh"), 5, 0)
    form.addWidget(window.mesh_combo, 5, 1)
    form.addWidget(window.open_mesh_button, 5, 2)
    form.addWidget(QtWidgets.QLabel("Frame"), 6, 0)
    form.addWidget(window.frame_slider, 6, 1)
    form.addWidget(window.frame_spin, 6, 2)

    layout.addWidget(window.status_label)
    layout.addStretch(1)
    return panel


def apply_main_window_theme(window: QtWidgets.QMainWindow, theme: str) -> None:
    if theme == "light":
        stylesheet = """
        QMainWindow, QWidget {
            background: #eef4fb;
            color: #17324a;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 12px;
        }
        QFrame#renderCard, QFrame#frontendShell, QFrame#legacyControlsPanel {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(124, 151, 192, 0.32);
            border-radius: 18px;
        }
        QLabel#appTitle {
            font-size: 20px;
            font-weight: 700;
        }
        QLabel#appSubtitle, QLabel#sidePanelSubtitle, QLabel#viewportHint {
            color: #5f7b97;
        }
        QLabel#renderCardTitle, QLabel#sidePanelTitle {
            font-size: 14px;
            font-weight: 700;
        }
        QLabel#renderCardSubtitle {
            color: #6d86a1;
        }
        QLabel#frontendPlaceholder {
            color: #65809d;
            font-size: 13px;
        }
        QPushButton#sidebarHandleButton {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(124, 151, 192, 0.38);
            border-radius: 11px;
            padding: 0 2px;
            color: #35506c;
        }
        QSplitter::handle {
            background: transparent;
            width: 6px;
            height: 6px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QSlider {
            min-height: 30px;
        }
        """
    else:
        stylesheet = """
        QMainWindow, QWidget {
            background: #0b1020;
            color: #e6edf7;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 12px;
        }
        QFrame#renderCard, QFrame#frontendShell, QFrame#legacyControlsPanel {
            background: rgba(18, 26, 43, 0.88);
            border: 1px solid rgba(76, 101, 136, 0.25);
            border-radius: 18px;
        }
        QLabel#appTitle {
            font-size: 20px;
            font-weight: 700;
        }
        QLabel#appSubtitle, QLabel#sidePanelSubtitle, QLabel#viewportHint {
            color: #8ea3c5;
        }
        QLabel#renderCardTitle, QLabel#sidePanelTitle {
            font-size: 14px;
            font-weight: 700;
        }
        QLabel#renderCardSubtitle {
            color: #8ea3c5;
        }
        QLabel#frontendPlaceholder {
            color: #8ea3c5;
            font-size: 13px;
        }
        QPushButton#sidebarHandleButton {
            background: rgba(18, 26, 43, 0.94);
            border: 1px solid rgba(92, 124, 171, 0.34);
            border-radius: 11px;
            padding: 0 2px;
            color: #d8e5fb;
        }
        QSplitter::handle {
            background: transparent;
            width: 6px;
            height: 6px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QSlider {
            min-height: 30px;
        }
        """

    window.setStyleSheet(stylesheet)
    for view_name in ("gaussian_view", "mesh_view"):
        view = getattr(window, view_name, None)
        if view is not None and hasattr(view, "apply_theme"):
            view.apply_theme(theme)


def setup_main_window_ui(window: QtWidgets.QMainWindow) -> None:
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(10, 10, 10, 10)
    root.setSpacing(0)

    window.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    window.main_splitter.setChildrenCollapsible(False)
    root.addWidget(window.main_splitter, 1)

    left_panel = QtWidgets.QWidget()
    window.viewport_panel = left_panel
    left_layout = QtWidgets.QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(10)

    window.render_container = QtWidgets.QWidget()
    view_layout = QtWidgets.QHBoxLayout(window.render_container)
    view_layout.setContentsMargins(0, 0, 0, 0)
    view_layout.setSpacing(12)

    window.gaussian_view = RenderView("Gaussian")
    window.mesh_view = RenderView("Mesh")
    window.gaussian_card = _make_render_card(
        "Gaussian View", "Online GS / Gaussian sequence", window.gaussian_view
    )
    window.mesh_card = _make_render_card(
        "Mesh View", "Dynamic mesh sequence", window.mesh_view
    )
    view_layout.addWidget(window.gaussian_card, 1)
    view_layout.addWidget(window.mesh_card, 1)

    left_layout.addWidget(window.render_container, 1)

    right_panel = QtWidgets.QFrame()
    window.sidebar_panel = right_panel
    right_panel.setObjectName("frontendShell")
    right_panel.setMinimumWidth(360)
    right_layout = QtWidgets.QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    window.frontend_panel_host = QtWidgets.QFrame()
    host_layout = QtWidgets.QVBoxLayout(window.frontend_panel_host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.setSpacing(0)

    window.frontend_placeholder_label = QtWidgets.QLabel(
        "Initializing modern frontend…"
    )
    window.frontend_placeholder_label.setObjectName("frontendPlaceholder")
    window.frontend_placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    host_layout.addWidget(window.frontend_placeholder_label, 1)

    window.legacy_controls_container = _build_hidden_legacy_controls(window)
    host_layout.addWidget(window.legacy_controls_container, 1)
    right_layout.addWidget(window.frontend_panel_host, 1)

    window.main_splitter.addWidget(left_panel)
    window.main_splitter.addWidget(right_panel)
    window.main_splitter.setStretchFactor(0, 4)
    window.main_splitter.setStretchFactor(1, 2)
    window.main_splitter.setSizes([1120, 480])

    window.sidebar_handle_button = QtWidgets.QPushButton("❯", central)
    window.sidebar_handle_button.setObjectName("sidebarHandleButton")
    window.sidebar_handle_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    window.sidebar_handle_button.setFixedSize(20, 74)
    window.sidebar_handle_button.raise_()
    apply_main_window_theme(window, "light")
