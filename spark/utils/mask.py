import os
from glob import glob
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

def generate_masks(boxes, image_size):
    """
    Generate approximate binary masks from bounding boxes.
    Args:
        boxes (torch.Tensor): Bounding boxes, shape (num_objects, 4) in (x_min, y_min, x_max, y_max).
        image_size (tuple): Size of the image (height, width).
    Returns:
        torch.Tensor: Binary masks, shape (num_objects, height, width).
    """
    base_size = 1024
    height, width = image_size
    scale_h = height / base_size
    scale_w = width / base_size
    masks = []
    for box in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        x_min, y_min, x_max, y_max = map(int, box)
        x_min = int(x_min / scale_w)
        y_min = int(y_min / scale_h)
        x_max = int(x_max / scale_w)
        y_max = int(y_max / scale_h)
        mask[y_min:y_max, x_min:x_max] = 1
        masks.append(mask)
    return torch.tensor(np.array(masks), dtype=torch.uint8)

def generate_sam_masks(img_path):
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    # download the checkpoint from the repo
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam.to("cuda")
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)
    partition = "/mnt/lscratch/users/pchernakov"
    mask_path = img_path.replace("images", "masks").replace("../", "")
    os.makedirs(mask_path, exist_ok=True)
    for img_file in glob(os.path.join(img_path, "*.jpg")):
        basename = os.path.basename(img_file)
        dest = os.path.join(partition, mask_path, basename.replace(".jpg", ".npy"))
        if os.path.exists(dest):
            continue
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        stacked_masks = np.stack([mask["segmentation"] for mask in masks], axis=0)
        np.save(dest, stacked_masks)
        # cv2.imwrite(img.replace(".jpg", "_mask.jpg"), mask)

if __name__ == "__main__":
    generate_sam_masks("../datasets/stream1/images/train")