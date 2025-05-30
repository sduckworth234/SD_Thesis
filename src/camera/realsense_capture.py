import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCapture:
    def __init__(self):
        # Configure the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def start(self):
        # Start the pipeline
        self.pipeline.start(self.config)

    def stop(self):
        # Stop the pipeline
        self.pipeline.stop()

    def capture_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def save_frame(self, color_image, depth_image, frame_id):
        # Save the color and depth images
        cv2.imwrite(f'color_frame_{frame_id}.png', color_image)
        cv2.imwrite(f'depth_frame_{frame_id}.png', depth_image)

if __name__ == "__main__":
    realsense = RealSenseCapture()
    realsense.start()

    try:
        frame_id = 0
        while True:
            color_image, depth_image = realsense.capture_frame()
            if color_image is not None:
                cv2.imshow('Color Frame', color_image)
                cv2.imshow('Depth Frame', depth_image)

                # Save frames every 10 iterations
                if frame_id % 10 == 0:
                    realsense.save_frame(color_image, depth_image, frame_id)

                frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        realsense.stop()
        cv2.destroyAllWindows()