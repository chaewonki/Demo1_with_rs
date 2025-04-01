import pyrealsense2 as rs
import cv2
import numpy as np
from openvino.runtime import Core

CLASS_NAMES = {
    0: "LAT",
    1: "PA",
}

class RealSenseManager:
    distance = 0.0
    def __init__(self):
        self.init_realsense()
        self.setup_filters()
        self.load_inference_model()
    
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

    def load_inference_model(self):
        ie = Core()
        model_path = "app\\resources\\saved_model.xml"
        model = ie.read_model(model=model_path)
        self.compiled_model = ie.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0) 

    def dynamic_crop(self, image, width, height):
        min_distance = 0.740  
        max_distance = 0.950  

        normalized_distance = 0.55 + (self.distance - min_distance) * (0.65 - 0.55) / (max_distance - min_distance)
        normalized_distance = np.clip(normalized_distance, 0.1, 1.0)

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
            
            if draw_contours:
                image_with_contours = image.copy()
                cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)
                cv2.rectangle(image_with_contours, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cropped_image = image[y1:y2, x1:x2]
            
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

        input_image = cv2.resize(cropped_rgb_image, (224, 224))  
        input_image = input_image.astype(np.float32) / 255.0 
        input_image = np.transpose(input_image, (2, 0, 1))  
        input_image = np.expand_dims(input_image, axis=0)  

        results = self.compiled_model([input_image])[self.output_layer]

        predicted_class = np.argmax(results)
        class_name = CLASS_NAMES.get(predicted_class, "Unknown")

        return rgb_image, depth_image, milimeters, cropped_rgb_image, distance, class_name
    