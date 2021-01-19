import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmenters.meta import Augmenter
from typing import Optional


class MyAugmenter:
    def __init__(self, augmenter: None = None) -> None:
        if augmenter is None:
            self.augmenter = iaa.Sequential(
                [
                    iaa.flip.Fliplr(p=0.5),
                    iaa.flip.Flipud(p=0.5),
                    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 0.2))),
                    iaa.Sometimes(0.5, iaa.MultiplyBrightness(mul=(0.8, 1.2))),
                    iaa.SomeOf(
                        n=2,
                        random_order=True,
                        children=[
                            iaa.color.MultiplyHue((0.0, 1.0)),
                            iaa.color.MultiplySaturation((0.5, 2.0)),
                            iaa.contrast.GammaContrast(
                                gamma=(0.5, 1.75), per_channel=True
                            ),
                            iaa.arithmetic.Dropout(0.05),
                            iaa.arithmetic.CoarseDropout(p=0.05, size_percent=0.3),
                            iaa.imgcorruptlike.Spatter(severity=(2, 5)),
                        ],
                    ),
                    # This looks like a really promising transform but it is SO SLOW
                    # iaa.Sometimes(0.2, iaa.PiecewiseAffine(scale=(0, 0.08)))
                    iaa.CropToFixedSize(width=224, height=224),
                ]
            )
        else:
            self.augmenter = augmenter

    def __call__(self, img_arr, mask, ret_imgaug=False):
        aug_out = self.augmenter(
            image=img_arr,
            segmentation_maps=SegmentationMapsOnImage(mask, shape=mask.shape),
        )

        if ret_imgaug:
            return aug_out
        return aug_out[0].copy(), aug_out[1].arr[..., 0].copy()
