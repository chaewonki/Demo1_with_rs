from camera.realsense_manager import RealSenseManager
from serialComm.serial_comm import SerialComm
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

IMAGE_SIZE = (640, 480)
FONT_SIZE = 30 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMO1")
        self.setup_ui()
        self.serialComm = SerialComm()
        self.rs_manager = RealSenseManager()

        self.timerRGB = QTimer()
        self.timerRGB.timeout.connect(self.update_images)
        self.timerRGB.start(100)

        self.serialComm.kv_signal.connect(self.update_kV)
        self.serialComm.mA_signal.connect(self.update_mA)
        self.serialComm.mAs_signal.connect(self.update_mAs)
        self.serialComm.ms_signal.connect(self.update_ms)

    def timer_stop(self):
        self.timerRGB.stop()

    def update_images(self):
        rgb_image, depth_image, _, cropped_rgb_image, _, class_names = self.rs_manager.get_rgb_depth_images()

        height, width, _ = rgb_image.shape
        bytesPerLine = 3 * width

        qImg = QImage(rgb_image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        qImg = qImg.scaled(IMAGE_SIZE[0], IMAGE_SIZE[1])
        pixmap = QPixmap.fromImage(qImg)
        self.img_label.setPixmap(pixmap)

        bytesPerLine = 3 * width
        qImg = QImage(depth_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = qImg.scaled(IMAGE_SIZE[0], IMAGE_SIZE[1])
        pixmap = QPixmap.fromImage(qImg)
        self.img_label2.setPixmap(pixmap)

        if cropped_rgb_image is not None:
            cropped_rgb_image = np.ascontiguousarray(cropped_rgb_image)
            qCroppedImg = QImage(cropped_rgb_image.data,
                                 cropped_rgb_image.shape[1],
                                 cropped_rgb_image.shape[0],
                                 cropped_rgb_image.shape[1] * 3,
                                 QImage.Format_BGR888
                                )
            qCroppedImg = qCroppedImg.scaled(IMAGE_SIZE[0], IMAGE_SIZE[1])
            cropped_pixmap = QPixmap.fromImage(qCroppedImg)
            self.img_label3.setPixmap(cropped_pixmap)  
        
        self.position_value_label.setText(class_names)

    def update_kV(self, data):
        self.kv_value_label.setText(f"{data}")
    
    def update_mA(self, data):
        self.ma_value_label.setText(f"{data}")
    
    def update_mAs(self, data):
        self.mAs_ms_value_label.setText(f"{data:.1f}")
    
    def update_ms(self, data):
        self.mAs_ms_value_label.setText(f"{data:.1f}")

    def set_distance(self):
        _, _, milimeters, _, distance, _ = self.rs_manager.get_rgb_depth_images()
        self.distance_value_label.setText(f"{milimeters:.0f}")
        self.rs_manager.distance_to_table = milimeters
        self.rs_manager.distance = distance
    
    def get_thickness(self):
        _, _, milimeters, _, _, _ = self.rs_manager.get_rgb_depth_images()
        thickness = self.rs_manager.distance * 1000 - milimeters
        self.thickness_value_label.setText(f"{thickness:.0f}")
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        h_layout = QHBoxLayout()

        self.img_label = QLabel()
        self.img_label.setStyleSheet("background-color: white; border: 1px solid black;")
        self.img_label.setFixedSize(IMAGE_SIZE[0], IMAGE_SIZE[1])
        h_layout.addWidget(self.img_label)

        self.img_label2 = QLabel()
        self.img_label2.setStyleSheet("background-color: white; border: 1px solid black;")
        self.img_label2.setFixedSize(IMAGE_SIZE[0], IMAGE_SIZE[1])
        h_layout.addWidget(self.img_label2)

        self.img_label3 = QLabel()
        self.img_label3.setStyleSheet("background-color: white; border: 1px solid black;")
        self.img_label3.setFixedSize(IMAGE_SIZE[0], IMAGE_SIZE[1])
        h_layout.addWidget(self.img_label3)

        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #f0f0f0;")
        right_layout = QVBoxLayout(right_widget)

        title_label = QLabel("SYNICSCRAY DEMO APP")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"font-size: {FONT_SIZE}px; font-weight: bold;")

        distance_desc_label = QLabel("Distance(mm):")
        self.distance_value_label = QLabel("VALUE")
        thickness_desc_label = QLabel("Thickness(mm):")
        self.thickness_value_label = QLabel("VALUE")
        position_desc_label = QLabel("Position:")
        self.position_value_label = QLabel("VALUE")

        kv_desc_label = QLabel("kV:")
        self.kv_value_label = QLabel("00")
        ma_desc_label = QLabel("mA:")
        self.ma_value_label = QLabel("00")
        mAs_ms_desc_label = QLabel("mAs/ms")
        self.mAs_ms_value_label = QLabel("0.0")

        for label in [title_label, 
                      distance_desc_label, self.distance_value_label, 
                      thickness_desc_label, self.thickness_value_label,
                      position_desc_label, self.position_value_label,
                      kv_desc_label, self.kv_value_label, 
                      ma_desc_label, self.ma_value_label,
                      mAs_ms_desc_label, self.mAs_ms_value_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"font-size: {FONT_SIZE}px; font-weight: bold;")

        group1_box = QGroupBox("INFORMATION")
        group1_box.setStyleSheet(f"font-size: {FONT_SIZE}px; font-weight: bold;")
        group1_layout = QVBoxLayout()

        distance_layout = QHBoxLayout()
        distance_layout.addWidget(distance_desc_label)
        distance_layout.addWidget(self.distance_value_label)

        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(thickness_desc_label)
        thickness_layout.addWidget(self.thickness_value_label)

        position_layout = QHBoxLayout()
        position_layout.addWidget(position_desc_label)
        position_layout.addWidget(self.position_value_label)

        group1_layout.addLayout(distance_layout)
        group1_layout.addSpacing(10)
        group1_layout.addLayout(thickness_layout)
        group1_layout.addSpacing(10)
        group1_layout.addLayout(position_layout)
        group1_box.setLayout(group1_layout)

        group2_box = QGroupBox("Exposure Settings")
        group2_box.setStyleSheet(f"font-size: {FONT_SIZE}px; font-weight: bold;")
        group2_layout = QVBoxLayout()

        kv_layout = QHBoxLayout()
        kv_layout.addWidget(kv_desc_label)
        kv_layout.addWidget(self.kv_value_label)

        ma_layout = QHBoxLayout()
        ma_layout.addWidget(ma_desc_label)
        ma_layout.addWidget(self.ma_value_label)

        mAs_ms_layout = QHBoxLayout()
        mAs_ms_layout.addWidget(mAs_ms_desc_label)
        mAs_ms_layout.addWidget(self.mAs_ms_value_label)

        group2_layout.addLayout(kv_layout)
        group2_layout.addSpacing(10)
        group2_layout.addLayout(ma_layout)
        group2_layout.addSpacing(10)
        group2_layout.addLayout(mAs_ms_layout)
        group2_box.setLayout(group2_layout)

        right_layout.addWidget(title_label)
        right_layout.addSpacing(40)
        right_layout.addWidget(group1_box)
        right_layout.addSpacing(30)
        right_layout.addWidget(group2_box)
        right_layout.addSpacing(35)
        

        self.distance_button = QPushButton("SET DISTANCE")
        self.distance_button.setStyleSheet(f"font-size: {FONT_SIZE}px; padding: 10px;")
        self.distance_button.clicked.connect(self.set_distance)
        self.thicnkness_button = QPushButton("GET THICKNESS")
        self.thicnkness_button.setStyleSheet(f"font-size: {FONT_SIZE}px; padding: 10px;")
        self.thicnkness_button.clicked.connect(self.get_thickness)

        button_layout = QHBoxLayout()
        
        button_layout.addWidget(self.distance_button)
        button_layout.addWidget(self.thicnkness_button)

        right_layout.addLayout(button_layout)

        h_layout.addWidget(right_widget)
        main_layout.addLayout(h_layout)
        self.resize(1000, 800)
    