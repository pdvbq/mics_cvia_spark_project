import numpy as np
import torch

def generate_masks(boxes, image_size):
    """
    Generate approximate binary masks from bounding boxes.
    Args:
        boxes (torch.Tensor): Bounding boxes, shape (num_objects, 4) in (x_min, y_min, x_max, y_max).
        image_size (tuple): Size of the image (height, width).
    Returns:
        torch.Tensor: Binary masks, shape (num_objects, height, width).
    """
    height, width = image_size
    masks = []
    for box in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        x_min, y_min, x_max, y_max = map(int, box)
        mask[y_min:y_max, x_min:x_max] = 1
        masks.append(mask)
    return torch.tensor(masks, dtype=torch.uint8)