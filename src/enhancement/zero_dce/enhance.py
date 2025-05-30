import cv2
import numpy as np
import torch

class ZeroDCE:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def enhance(self, low_light_image):
        with torch.no_grad():
            enhanced_image = self.model(low_light_image)
        return enhanced_image

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    model_path = "path/to/your/zerodce_model.pth"  # Update with your model path
    image_path = "path/to/your/low_light_image.jpg"  # Update with your image path
    output_path = "path/to/your/enhanced_image.jpg"  # Update with your output path

    low_light_image = load_image(image_path)
    enhancer = ZeroDCE(model_path)
    enhanced_image = enhancer.enhance(low_light_image)
    save_image(enhanced_image, output_path)