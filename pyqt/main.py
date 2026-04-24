import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QFileDialog,
                             QLabel, QGroupBox, QProgressBar, QStatusBar, QComboBox)
from PyQt6.QtGui import QPixmap, QTransform, QFont, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent


class InferenceThread(QThread):
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        try:
            workshop = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main-workshop'))
            if workshop not in sys.path:
                sys.path.insert(0, workshop)
            from mymask import run as mask_run
            annotated, count, _class_counts = mask_run(self.model_path, self.image_path)
            self.finished.emit(annotated, count)
        except Exception as e:
            self.error.emit(str(e))


class SeedCountingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("种子计数智能分析系统 - YOLOv11-seg")
        self.resize(1400, 900)

        self.current_pixmap = None
        self.model_path = None
        self.image_path = None
        self._thread = None
        self._zoom_level = 1.0
        self._models_dir = os.path.join(os.path.dirname(__file__), 'models')

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- 左侧控制面板 ---
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(260)
        self.sidebar = QVBoxLayout(sidebar_widget)
        self.sidebar.setContentsMargins(16, 20, 16, 20)
        self.sidebar.setSpacing(20)

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        self.combo_model = QComboBox()
        self.combo_model.setPlaceholderText("请选择模型...")
        self.btn_refresh = QPushButton("刷新模型列表")
        self.btn_refresh.setObjectName("secondaryBtn")
        model_layout.addWidget(self.combo_model)
        model_layout.addWidget(self.btn_refresh)
        model_group.setLayout(model_layout)

        # 算法控制
        algo_group = QGroupBox("算法控制")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(10)
        self.btn_run = QPushButton("启动种子计数")
        self.btn_run.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.lbl_result = QLabel("检测结果: 待处理")
        self.lbl_result.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setObjectName("resultLabel")
        algo_layout.addWidget(self.btn_run)
        algo_layout.addWidget(self.progress)
        algo_layout.addWidget(self.lbl_result)
        algo_group.setLayout(algo_layout)

        # 图像工具
        img_group = QGroupBox("图像工具")
        img_layout = QVBoxLayout()
        img_layout.setSpacing(8)
        self.btn_open = QPushButton("打开种子图像")
        self.btn_fit = QPushButton("适应窗口")
        self.btn_100 = QPushButton("100% 原始大小")
        self.btn_rotate_l = QPushButton("左旋转 90°")
        self.btn_rotate_r = QPushButton("右旋转 90°")
        for btn in (self.btn_open, self.btn_fit, self.btn_100, self.btn_rotate_l, self.btn_rotate_r):
            img_layout.addWidget(btn)
        img_group.setLayout(img_layout)

        self.sidebar.addWidget(model_group)
        self.sidebar.addWidget(algo_group)
        self.sidebar.addWidget(img_group)
        self.sidebar.addStretch()

        # --- 右侧图像区 ---
        self.view = QGraphicsView()
        self.view.setObjectName("imageView")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.view, 1)

        # --- 底部状态栏 ---
        self.status_file = QLabel("未加载图像")
        self.status_zoom = QLabel("缩放: 100%")
        self.status_coord = QLabel("X: --  Y: --")
        for lbl in (self.status_file, self.status_zoom, self.status_coord):
            lbl.setObjectName("statusLabel")
        sb = QStatusBar()
        sb.setObjectName("statusBar")
        sb.addWidget(self.status_file, 2)
        sb.addWidget(self.status_zoom, 1)
        sb.addPermanentWidget(self.status_coord)
        self.setStatusBar(sb)

        self.apply_styles()
        self._refresh_models()

        self.combo_model.currentIndexChanged.connect(self._on_model_selected)
        self.btn_refresh.clicked.connect(self._refresh_models)
        self.btn_open.clicked.connect(self.load_image)
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_fit.clicked.connect(self.fit_view)
        self.btn_100.clicked.connect(self.reset_zoom)
        self.btn_rotate_l.clicked.connect(lambda: self.rotate_image(-90))
        self.btn_rotate_r.clicked.connect(lambda: self.rotate_image(90))

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QWidget#sidebar {
                background-color: #2a2a3e;
                border-right: 1px solid #44475a;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #a6adc8;
                border: 1px solid #44475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #5865f2;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 9px 14px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4752c4; }
            QPushButton:pressed { background-color: #3c45a5; }
            QPushButton:disabled { background-color: #44475a; color: #6c7086; }
            QPushButton#secondaryBtn {
                background-color: #313244;
                font-size: 12px;
                padding: 7px 14px;
            }
            QPushButton#secondaryBtn:hover { background-color: #45475a; }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #44475a;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 13px;
            }
            QComboBox:hover { border-color: #5865f2; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #5865f2;
                border: 1px solid #44475a;
            }
            QLabel#subLabel {
                font-size: 11px;
                color: #6c7086;
            }
            QLabel#resultLabel {
                color: #a6e3a1;
                padding: 6px 0;
            }
            QProgressBar {
                background-color: #44475a;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #5865f2;
                border-radius: 3px;
            }
            QGraphicsView#imageView {
                background-color: #11111b;
                border: none;
            }
            QStatusBar#statusBar {
                background-color: #181825;
                border-top: 1px solid #44475a;
                color: #6c7086;
                font-size: 12px;
            }
            QLabel#statusLabel {
                color: #6c7086;
                font-size: 12px;
                padding: 0 8px;
            }
        """)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.Type.MouseMove:
            if self.current_pixmap:
                scene_pos = self.view.mapToScene(event.pos())
                px = self.current_pixmap.pixmap()
                x, y = int(scene_pos.x()), int(scene_pos.y())
                if 0 <= x < px.width() and 0 <= y < px.height():
                    self.status_coord.setText(f"X: {x}  Y: {y}")
                else:
                    self.status_coord.setText("X: --  Y: --")
        return super().eventFilter(obj, event)

    def _refresh_models(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        pts = [f for f in os.listdir(self._models_dir) if f.endswith('.pt')] if os.path.isdir(self._models_dir) else []
        if pts:
            for name in sorted(pts):
                self.combo_model.addItem(name, os.path.join(self._models_dir, name))
            self.combo_model.setCurrentIndex(0)
            self.model_path = self.combo_model.currentData()
        else:
            self.model_path = None
        self.combo_model.blockSignals(False)
        self._update_run_button()

    def _on_model_selected(self, index):
        self.model_path = self.combo_model.itemData(index) if index >= 0 else None
        self._update_run_button()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择种子图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            px = QPixmap(path)
            self._show_pixmap(px)
            self.lbl_result.setText("检测结果: 待处理")
            self.status_file.setText(f"{os.path.basename(path)}  {px.width()} × {px.height()}")
            self._update_run_button()

    def _update_run_button(self):
        self.btn_run.setEnabled(bool(self.model_path and self.image_path))

    def run_inference(self):
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_result.setText("检测中，请稍候...")
        self._thread = InferenceThread(self.model_path, self.image_path)
        self._thread.finished.connect(self._on_inference_done)
        self._thread.error.connect(self._on_inference_error)
        self._thread.start()

    def _on_inference_done(self, annotated_bgr, count):
        h, w, ch = annotated_bgr.shape
        rgb = annotated_bgr[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qimg)
        self._show_pixmap(px)
        self.lbl_result.setText(f"检测结果: {count} 粒种子")
        self.status_file.setText(f"{os.path.basename(self.image_path)}  {px.width()} × {px.height()}  [检测完成]")
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

    def _on_inference_error(self, msg):
        self.lbl_result.setText(f"错误: {msg}")
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

    def _show_pixmap(self, pixmap):
        self.scene.clear()
        self.current_pixmap = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.current_pixmap)
        self.fit_view()

    def fit_view(self):
        if self.current_pixmap:
            self.view.resetTransform()
            self.view.fitInView(self.current_pixmap, Qt.AspectRatioMode.KeepAspectRatio)
            # 计算实际缩放比
            tr = self.view.transform()
            self._zoom_level = tr.m11()
            self.status_zoom.setText(f"缩放: {self._zoom_level * 100:.0f}%")

    def reset_zoom(self):
        if self.current_pixmap:
            self.view.resetTransform()
            self._zoom_level = 1.0
            self.status_zoom.setText("缩放: 100%")

    def rotate_image(self, angle):
        if self.current_pixmap:
            transform = QTransform().rotate(angle)
            new_pixmap = self.current_pixmap.pixmap().transformed(transform)
            self.current_pixmap.setPixmap(new_pixmap)
            self.view.fitInView(self.current_pixmap, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        new_zoom = self._zoom_level * factor
        if 0.05 <= new_zoom <= 20:
            self._zoom_level = new_zoom
            self.view.scale(factor, factor)
            self.status_zoom.setText(f"缩放: {self._zoom_level * 100:.0f}%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeedCountingUI()
    window.show()
    sys.exit(app.exec())
