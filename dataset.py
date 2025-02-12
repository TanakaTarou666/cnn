import os
import random
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

class MyDataset(Dataset):
    def __init__(self,image_dirs: List[str], transform: torchvision.transforms.Compose , num_samples: Optional[int] = None):
        self.transform = transform 
        self.loader = datasets.folder.default_loader
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: dict[str, int] = {dir: i for i, dir in enumerate(image_dirs)}

        # データセットの画像パスとラベルを取得
        for dir in self.class_to_idx:
            if not os.path.isdir(dir):
                continue
            images = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
            self.image_paths.extend(images)
            self.labels.extend([self.class_to_idx[dir]] * len(images))

        # サンプル数が指定されている場合は、その数だけランダムにサンプル
        if num_samples is not None and num_samples < len(self.image_paths):
            indices = random.sample(range(len(self.image_paths)), num_samples)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label:int = self.labels[idx]
        image: torch.Tensor = self.loader(self.image_paths[idx])
        if self.transform:
            image: torch.Tensor = self.transform(image)
        return image, label
