import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QFileDialog,
                             QLabel, QGroupBox)
from PyQt6.QtGui import QPixmap, QTransform, QFont
from PyQt6.QtCore import Qt


class SeedCountingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("种子计数智能分析系统 - YOLOv11-seg")
        self.resize(1300, 900)

        # 核心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)  # 左右面板间距

        # --- 1. 左侧：控制面板区 ---
        self.sidebar = QVBoxLayout()
        self.sidebar.setContentsMargins(15, 20, 15, 20)
        self.sidebar.setSpacing(25)  # 增加控件间的垂直间距

        # 算法控制分组
        algo_group = QGroupBox("算法控制")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(15)

        self.btn_run = QPushButton("启动种子计数")
        self.lbl_result = QLabel("检测结果: 待处理")
        # 放大结果标签字体
        self.lbl_result.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        algo_layout.addWidget(self.btn_run)
        algo_layout.addWidget(self.lbl_result)
        algo_group.setLayout(algo_layout)

        # 图像操作分组
        img_group = QGroupBox("图像浏览工具")
        img_layout = QVBoxLayout()
        img_layout.setSpacing(15)

        self.btn_open = QPushButton("打开种子图像")
        self.btn_rotate_l = QPushButton("左旋转 90°")
        self.btn_rotate_r = QPushButton("右旋转 90°")

        img_layout.addWidget(self.btn_open)
        img_layout.addWidget(self.btn_rotate_l)
        img_layout.addWidget(self.btn_rotate_r)
        img_group.setLayout(img_layout)

        self.sidebar.addWidget(algo_group)
        self.sidebar.addWidget(img_group)
        self.sidebar.addStretch()

        # --- 2. 右侧：图像渲染区 ---
        self.view = QGraphicsView()
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.main_layout.addLayout(self.sidebar, 1)
        self.main_layout.addWidget(self.view, 4)

        # 样式应用
        self.apply_styles()

        # 信号连接
        self.btn_open.clicked.connect(self.load_image)
        self.btn_rotate_l.clicked.connect(lambda: self.rotate_image(-90))
        self.btn_rotate_r.clicked.connect(lambda: self.rotate_image(90))

        self.current_pixmap = None

    def apply_styles(self):
        # 统一设置按钮字体大小与内边距
        style = """
            QPushButton { 
                background-color: #3498db; 
                color: white; 
                border-radius: 6px; 
                padding: 12px 20px; 
                font-size: 15px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #2980b9; }
            QGroupBox { 
                font-size: 16px; 
                font-weight: bold; 
                border: 1px solid #bdc3c7; 
                border-radius: 8px; 
                margin-top: 15px; 
                padding-top: 20px;
            }
        """
        self.setStyleSheet(style)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择种子图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.scene.clear()
            self.current_pixmap = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.current_pixmap)
            self.view.fitInView(self.current_pixmap, Qt.AspectRatioMode.KeepAspectRatio)

    def rotate_image(self, angle):
        if self.current_pixmap:
            pixmap = self.current_pixmap.pixmap()
            transform = QTransform().rotate(angle)
            new_pixmap = pixmap.transformed(transform)
            self.current_pixmap.setPixmap(new_pixmap)
            self.view.fitInView(self.current_pixmap, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.view.scale(factor, factor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeedCountingUI()
    window.show()
    sys.exit(app.exec())