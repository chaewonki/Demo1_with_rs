import pyrealsense2 as rs
import cv2
import numpy as np
from openvino.runtime import Core

CLASS_NAMES = {
    0: "LAT",
    1: "PA",
}

WIDTH_RGB = 1280
HEIGHT_RGB = 720

class RealSenseManager:
    def __init__(self):
        self.init_realsense()
        self.setup_filters()
        self.load_inference_model()
        self.distance = 0.0
    
    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, WIDTH_RGB, HEIGHT_RGB, rs.format.bgr8, 30)
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

    def load_inference_model(self):
        ie = Core()
        model_path = "app\\resources\\saved_model.xml"
        model = ie.read_model(model=model_path)
        self.compiled_model = ie.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0) 

    # from camera to table 955mm, (455x455)
    # from camera to table 720mm, (600x600)
    def generate_crop_size(self, x):
        return int(-0.617 * x + 1046)

    def dynamic_crop(self, image, width, height):
        crop_size = self.generate_crop_size(self.milimeters)
        center_x, center_y = width // 2, height // 2
        x1 = max(0, center_x - crop_size // 2)
        x2 = min(width, center_x + crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        y2 = min(height, center_y + crop_size // 2)

        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

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
        self.distance = depth_frame2.get_distance(width // 2, height // 2)
        self.milimeters = self.distance * 1000    

        cropped_rgb_image = self.dynamic_crop(rgb_image, WIDTH_RGB, HEIGHT_RGB)

        input_image = cv2.resize(cropped_rgb_image, (224, 224))  
        input_image = input_image.astype(np.float32) / 255.0 
        input_image = np.transpose(input_image, (2, 0, 1))  
        input_image = np.expand_dims(input_image, axis=0)  

        results = self.compiled_model([input_image])[self.output_layer]

        predicted_class = np.argmax(results)
        class_name = CLASS_NAMES.get(predicted_class, "Unknown")

        return rgb_image, depth_image, self.milimeters, cropped_rgb_image, self.distance, class_name
    