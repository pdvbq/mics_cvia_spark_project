import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Compose

class ImgTransform(Compose):
    def __init__(self, size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = F.to_pil_image(image)
        return super().__call__(image)

if __name__ == "__main__":
    transform = ImgTransform()
    img = torch.randn(3, 224, 224)
    img = transform(img)
    print(img.shape)
    print(img)