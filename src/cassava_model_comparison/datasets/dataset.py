# Standard library
from pathlib import Path
from typing import Callable, Tuple

# Third-party
import cv2
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


# CassavaDataset
class CassavaDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            transform: Callable[[Image.Image], Tensor]
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        img_bgr = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        image = self.transform(pil_img)

        return image, label




