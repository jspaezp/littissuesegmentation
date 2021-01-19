import numpy as np
from littissuesegmentation.augmenters import MyAugmenter


def test_augmenter():
    samp_img = np.random.randint(0, 255, size=(512, 512, 3)).astype("uint8")
    samp_mask = np.random.randint(0, 1, size=(512, 512, 1)).astype("uint8")

    aug = MyAugmenter()
    o_img, o_mask = aug(samp_img, samp_mask)
    print(f"Output img shape {o_img.shape}")
    print(f"Output mask shape {o_mask.shape}")


if __name__ == "__main__":
    test_augmenter()
