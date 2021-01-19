import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Callable, Tuple, List, Union

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from littissuesegmentation.augmenters import MyAugmenter
from littissuesegmentation.chunking import chunk_genrator
from numpy import ndarray

INPUT_SIZE = [3, 224, 224]
MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]


def preprocess_input(x: np.uint8):
    x = x / 255
    mean = np.array(MEANS)
    x = x - mean

    std = np.array(STDS)
    x = x / std

    return x


def norm_img(img) -> torch.Tensor:
    img = preprocess_input(img)
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().float()
    return img


def norm_mask(mask):
    mask = torch.from_numpy(mask.transpose((0, 1))).contiguous().float()
    return mask


def img_path_to_arr(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def mask_path_to_arr(filepath):
    # Note that the channels are
    mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    assert len(mask.shape) == 2, "Unexpected number of output layers"
    return mask


## greatly inspired by
## https://github.com/ternaus/cloths_segmentation/blob/59a2384bb270260603450729fa1ad52971ccea65/cloths_segmentation/dataloaders.py#L12
class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        readers: Tuple[Callable, Callable],
        normalizers: Tuple[Callable, Callable],
        augmenter: Callable = None,
    ) -> None:
        print(readers)
        self.samples = samples
        self.readers = readers
        self.length = len(self.samples)
        self.augmenter = augmenter
        self.normalizers = normalizers

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = [f(x) for x, f in zip(self.samples[index], self.readers)]

        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        image, mask = [f(x) for x, f in zip([image, mask], self.normalizers)]
        return image, mask


class LitSMPDatamodule(pl.LightningDataModule):
    def __init__(self, in_path: Union[Path, str], batch_size=32):
        super().__init__()

        print(
            f"Initializing datamoule in directory {in_path} and batch size of {batch_size}"
        )

        img_names = list((in_path / "img").glob("*.png"))
        input_names = [
            tuple(in_path / "img" / p.name, in_path / "mask" / p.name)
            for p in img_names
        ]
        print(f"Found {len(input_names)} Posible files")

        input_names = [tuple(x, y) for x, y in input_names if x.exists() and y.exists()]
        print(f"Found {len(input_names)} paired files")

        self.samples = input_names
        self.readers = [img_path_to_arr, mask_path_to_arr]
        self.normalizers = [norm_img, norm_mask]
        self.batch_size = batch_size
        self.augmenter = MyAugmenter()

    def setup(self, stage=None):
        random.shuffle(self.samples)
        val_len = int(len(self.samples) * 0.2)
        self.val_samples = self.samples[:val_len]
        self.train_samples = self.samples[val_len:]

        self.val_dataset = SegmentationDataset(
            self.val_samples, self.readers, self.normalizers
        )
        self.train_dataset = SegmentationDataset(
            self.train_samples, self.readers, self.normalizers, augmenter=self.augmenter
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=False,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--in_path",
            default=".",
            type=str,
            help="Parent directory with the data, should have 2 sub-directories, img and mask",
        )
        parser.add_argument(
            "--batch_size",
            default=32,
            type=int,
            help="Batch size to use",
        )

        return parser


class ChunkIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, img: ndarray, stride: int, chunk_size: int, verbose: bool=False) -> None:
        super(ChunkIterableDataset).__init__()
        self.chunk_gen = chunk_genrator(img, stride, chunk_size, verbose)

    def __iter__(self):
        return iter(self.chunk_gen)


class ImgChunkDataloader(torch.utils.data.DataLoader):
    def __init__(self, img: ndarray, stride: int, chunk_size: int, verbose: bool=False, batch_size: int=8) -> None:
        self.chunk_iterable = ChunkIterableDataset(img, stride, chunk_size, verbose)
        super().__init__(self.chunk_iterable, batch_size=batch_size, num_workers=0)
