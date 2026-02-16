from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image
from torch import Tensor
from torchvision import transforms


def pil_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def show_image_from_url(url: str) -> None:
    img_pil = pil_from_url(url) # (H,W,C)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.show()


def make_batch_image_from_url(url: str, transform=None) -> Tensor:
    img_pil = pil_from_url(url) # (H,W,C)

    if transform is None:
        transform = transforms.ToTensor()

    img = transform(img_pil)          # (C,H,W)
    batch_img = img.unsqueeze(0)          # (1,C,H,W)

    return batch_img
