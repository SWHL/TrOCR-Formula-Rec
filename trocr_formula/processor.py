# -*- encoding: utf-8 -*-
import albumentations as alb
import numpy as np

from trocr_formula.formula_processor_helper.nougat import Bitmap, Dilation, Erosion
from trocr_formula.formula_processor_helper.weather import (  # Snow,
    Fog,
    Frost,
    Rain,
    Shadow,
)


class TrainProcessor:
    def __init__(self):
        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=1,
                        ),
                        alb.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=0.5,
                        ),
                    ],
                    p=0.15,
                ),
                # alb.InvertImg(p=.15),
                alb.RGBShift(
                    r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3
                ),
                alb.GaussNoise(10, p=0.2),
                alb.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
                alb.ImageCompression(95, p=0.3),
                # alb.ToGray(always_apply=True),
                # alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                # ToTensorV2(),
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return image
        res = self.transform(image=image)
        image = res["image"]
        return image
