import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QGroupBox
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import pyrealsense2 as rs
import cv2
import numpy as np
import serial
import threading

STX = b'\x02'
READY = b'\x00\x00\x01'
STA = b'STA'
KV = b'\x00KV'
MA = b'\x00MA'
MAS = b'MAS'
MS = b'\x00MS'
APR = b'APR'
COL_X = b'COX'
COL_Y = b'COY'

IMAGE_SIZE = (640, 480)

FONT_SIZE = 30 

class SerialComm(QWidget):
    kv_signal = pyqtSignal(int)
    mA_signal = pyqtSignal(float)
    mAs_signal = pyqtSignal(float)
    ms_signal = pyqtSignal(float)
    projection_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.initSerial()

    def initSerial(self):
        self.serial_port = serial.Serial(
            port='COM3',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=None
        )
        
        serial_thread = threading.Thread(target=self.receive_packet)
        serial_thread.daemon = True
        serial_thread.start()
    
    def extract_direction_bits(self, data_bytes):
        last_byte = data_bytes[-1]
        thrid_fourth_bits = (last_byte >> 2) & 0b11
        thrid_fourth_bits = (thrid_fourth_bits<< 2) & 0b1111
        return thrid_fourth_bits

    def receive_packet(self):
        while True:
            byte_read = self.serial_port.read(1)
            if byte_read.startswith(STX):
                packet = byte_read + self.serial_port.read(9)
                id_bytes = packet[1:4] 
                data_bytes = packet[4:7]
                if id_bytes == KV:
                    kV_value = int.from_bytes(data_bytes)
                    self.kv_signal.emit(kV_value)
                elif id_bytes == MA:
                    mA_value = int.from_bytes(data_bytes) / 100
                    self.mA_signal.emit(mA_value)
                elif id_bytes == MAS:
                    mAs_ms_value = int.from_bytes(data_bytes) / 100
                    self.mAs_signal.emit(mAs_ms_value)
                elif id_bytes == MS:
                    ms_value = int.from_bytes(data_bytes) / 100
                    self.ms_signal.emit(ms_value)
                elif id_bytes == APR:
                    projection_bits = self.extract_direction_bits(data_bytes)
                    self.projection_signal.emit(projection_bits)
                elif id_bytes == COL_X:
                    cox_value = int.from_bytes(data_bytes)
                    print(f"COL_X: {cox_value}")
                elif id_bytes == COL_Y:
                    coy_value = int.from_bytes(data_bytes)
                    print(f"COL_Y: {coy_value}")
    
    def string_to_byte(self, key_string):
        return bytes.fromhex(key_string.replace(" ",""))

class RealSenseManager:
    distance = 0.0
    def __init__(self):
        self.init_realsense()
        self.setup_filters()
    
    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.exposure, 33000)
        depth_sensor.set_option(rs.option.gain, 16)
        depth_sensor.set_option(rs.option.laser_power, 150)  
        depth_sensor.set_option(rs.option.emitter_enabled, 1) 
        depth_sensor.set_option(rs.option.enable_auto_white_balance, 0)

        depth_sensor.set_option(rs.option.depth_units, 0.001)

        self.align = rs.align(rs.stream.color)
    
    def setup_filters(self):
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0)
        self.threshold_filter.set_option(rs.option.max_distance, 10)

        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)  # Default reasonable value
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)  # 0 = Jet
        self.colorizer.set_option(rs.option.histogram_equalization_enabled, 1)  # Checked

    def dynamic_crop(self, image, width, height):
        min_distance = 0.740  
        max_distance = 0.950  

        normalized_distance = 0.55 + (self.distance - min_distance) * (0.65 - 0.55) / (max_distance - min_distance)
        
        crop_size_factor = 1 - normalized_distance

        crop_width = int(width * crop_size_factor)
        crop_height = int(height * crop_size_factor)

        center_x, center_y = width // 2, height // 2
        x1 = max(0, center_x - crop_width // 2)
        x2 = min(width, center_x + crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        y2 = min(height, center_y + crop_height // 2)

        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    def crop_by_object(self, image, padding=20, draw_contours=True):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            image_with_contours = image.copy()
            
            if draw_contours:
                cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)
                cv2.rectangle(image_with_contours, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cropped_image = image[y1:y2, x1:x2]
            
            if draw_contours:
                cv2.imshow("Image with Contours", image_with_contours)
                cv2.waitKey(1)
            
            return cropped_image
        else:
            print("No object detected, using center crop.")
            width, height = image.shape[1], image.shape[0]
            crop_size = min(width, height) // 2
            center_x, center_y = width // 2, height // 2
            x1 = max(0, center_x - crop_size // 2)
            x2 = min(width, center_x + crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            y2 = min(height, center_y + crop_size // 2)
            return image[y1:y2, x1:x2]

    def get_rgb_depth_images(self):
        frames = self.pipeline.wait_for_frames()

        rgb_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(rgb_frame.get_data())

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame2 = frames.get_depth_frame()

        colorized_depth = self.colorizer.process(depth_frame)
        depth_image = np.asanyarray(colorized_depth.get_data())
        width, height = depth_frame2.get_width(), depth_frame2.get_height()
        distance = depth_frame2.get_distance(width // 2, height // 2)
        milimeters = distance * 1000    

        cropped_rgb_image = self.dynamic_crop(rgb_image, rgb_frame.get_width(), rgb_frame.get_height())

        cv2.imshow("Cropped RGB Image", cropped_rgb_image)  
        cv2.waitKey(1)  

        return rgb_image, depth_image, milimeters, cropped_rgb_image, distance
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEMO1")
        self.setup_ui()
        self.serialComm = SerialComm()
        self.rs_manager = RealSenseManager()

        self.timerRGB = QTimer()
        self.timerRGB.timeout.connect(self.update_images)
        self.timerRGB.start()

        self.serialComm.kv_signal.connect(self.update_kV)
        self.serialComm.mA_signal.connect(self.update_mA)
        self.serialComm.mAs_signal.connect(self.update_mAs)
        self.serialComm.ms_signal.connect(self.update_ms)

    def timer_stop(self):
        self.timerRGB.stop()

    def update_images(self):
        rgb_image, depth_image, _, cropped_rgb_image, _ = self.rs_manager.get_rgb_depth_images()

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
            cropped_height, cropped_width, _ = cropped_rgb_image.shape
            cropped_bytesPerLine = 3 * cropped_width
            qCroppedImg = QImage(cropped_rgb_image.tobytes(), cropped_width, cropped_height, cropped_bytesPerLine, QImage.Format_BGR888)
            qCroppedImg = qCroppedImg.scaled(IMAGE_SIZE[0], IMAGE_SIZE[1])
            cropped_pixmap = QPixmap.fromImage(qCroppedImg)
            self.img_label3.setPixmap(cropped_pixmap)  

    def update_kV(self, data):
        self.kv_value_label.setText(f"{data}")
    
    def update_mA(self, data):
        self.ma_value_label.setText(f"{data}")
    
    def update_mAs(self, data):
        self.mAs_ms_value_label.setText(f"{data:.1f}")
    
    def update_ms(self, data):
        self.mAs_ms_value_label.setText(f"{data:.1f}")

    def set_distance(self):
        _, _, milimeters, _, distance = self.rs_manager.get_rgb_depth_images()
        self.distance_value_label.setText(f"{milimeters:.0f}")
        self.rs_manager.distance_to_table = milimeters
        self.rs_manager.distance = distance
    
    def get_thickness(self):
        _, _, milimeters, _ = self.rs_manager.get_rgb_depth_images()
        thickness = self.rs_manager.distance - milimeters
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

        kv_desc_label = QLabel("kV:")
        self.kv_value_label = QLabel("00")
        ma_desc_label = QLabel("mA:")
        self.ma_value_label = QLabel("00")
        mAs_ms_desc_label = QLabel("mAs/ms")
        self.mAs_ms_value_label = QLabel("0.0")

        for label in [title_label, 
                      distance_desc_label, self.distance_value_label, 
                      thickness_desc_label, self.thickness_value_label,
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

        group1_layout.addLayout(distance_layout)
        group1_layout.addSpacing(10)
        group1_layout.addLayout(thickness_layout)
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
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())