from PIL import Image, ImageOps, ImageFilter
import torch
import numpy as np

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class smooth_remove_black:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "threshold_tolerance": ("INT", {"default": 2, "min": 1, "max": 24, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_smooth_remove_black"

    CATEGORY = "image"

    def image_smooth_remove_black(self, images, threshold=127, threshold_tolerance=2):
        return (self.smooth_remove_black(images, threshold, threshold_tolerance), )

    def smooth_remove_black(self, image, threshold, threshold_tolerance):
        images = []
        image = [tensor2pil(img) for img in image]
        for img in image:
            grayscale_image = img.convert('L')
            blurred_image = grayscale_image.filter(
                ImageFilter.GaussianBlur(radius=threshold_tolerance))
            mask = blurred_image.point(
                lambda x: ((x / threshold) * 255) if x < threshold else 255)
            transparent_image = img.copy()
            transparent_image.putalpha(mask)
            images.append(pil2tensor(transparent_image))
        batch = torch.cat(images, dim=0)

        return batch

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Smooth Remove Black": smooth_remove_black
}
