import gc
from pathlib import Path
import os

import cv2
import numpy as np
from tqdm.auto import tqdm

data_dir = "/kaggle/input/hubmap-kidney-segmentation"
split = "train"  # Change this to use test
TILE_EXTENSION = "png"  # Change to jpg for smaller files

# Those folders will store our images
os.makedirs(f"{split}_tiles/images", exist_ok=True)
os.makedirs(f"{split}_tiles/masks", exist_ok=True)

# This list will contain information about all our images
meta_ls = []


# The break down starts here


def tile_img_pairs(img_path, mask_path, out_dir=".", tile_size=300, stride_length=300):
    gc.collect()

    img_id = Path(img_path).stem
    img = cv2.imread(str(img_path))
    print(f"{img.shape} Img Reading Done")
    mask = cv2.imread(str(mask_path))
    print(f"{img.shape} Mask Reading Done")

    out_dir = Path(out_dir)
    out_img_dir = out_dir / "img"
    out_mask_dir = out_dir / "mask"

    x_max, y_max = img.shape[:2]

    for x0 in tqdm(list(range(0, x_max, stride_length)), leave=False):
        x1 = min(x_max, x0 + tile_size)
        for y0 in tqdm(list(range(0, y_max, stride_length)), leave=False):
            y1 = min(y_max, y0 + tile_size)

            img_tile = img[x0:x1, y0:y1]
            mask_tile = mask[x0:x1, y0:y1]

            img_tile_path = (
                out_img_dir / f"{img_id}_{x0}-{x1}x_{y0}-{y1}y.{TILE_EXTENSION}"
            )
            mask_tile_path = (
                out_mask_dir / f"{img_id}_{x0}-{x1}x_{y0}-{y1}y.{TILE_EXTENSION}"
            )

            cv2.imwrite(str(img_tile_path), img_tile, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(str(mask_tile_path), mask_tile)

            meta_ls.append(
                [
                    img_id,
                    x0,
                    x1,
                    y0,
                    y1,
                    img_tile.min(),
                    img_tile.max(),
                    mask_tile.max(),
                    img_tile_path,
                    mask_tile_path,
                ]
            )
