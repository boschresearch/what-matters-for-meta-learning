#   Copyright (c) 2022 Robert Bosch GmbH
#   Author: Ning Gao
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


class Augmenter(object):
    def __init__(self):
        # set global seed
        ia.seed(53)

        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential(
            [

                # crop some of the images by 0-10% of their height/width
                self.sometimes(iaa.CropAndPad(percent=(0, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255))),

                self.sometimes(iaa.GammaContrast((0.5, 2.0))),

                self.sometimes(iaa.AddToBrightness((-30, 30))),

                self.sometimes(iaa.AverageBlur(k=(1, 3))),

                self.sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                self.sometimes(
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.00, 0.05), size_percent=(0.02, 0.25),
                            per_channel=0.2
                        ),
                    ]),
                ),

            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def generate(self, images, segmaps=None):
        t, n, h, w, c = images.shape
        images = images.reshape(t * n, h, w, c)
        if segmaps is None:
            images = self.seq(images=(images*255).astype(np.uint8))
        else:
            segmaps = segmaps.reshape(t * n, h, w, 1)
            images, segmaps = self.seq(images=(images*255).astype(np.uint8), segmentation_maps=segmaps)
            segmaps = segmaps.reshape(t, n, h, w, 1)

        images = images.reshape(t, n, h, w, c)
        if segmaps is None:
            return (images/255.0).astype(np.float32)
        else:
            return (images/255.0).astype(np.float32), segmaps


class PascalAugmenter(object):
    def __init__(self):
        # set global seed
        ia.seed(53)

        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.

                # crop some of the images by 0-10% of their height/width
                self.sometimes(iaa.CropAndPad(percent=(0, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255))),

                self.sometimes(iaa.GammaContrast((0.5, 2.0))),

                # self.sometimes(iaa.AddToBrightness((-30, 30))),

                self.sometimes(iaa.AverageBlur(k=(1, 3))),

                self.sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                self.sometimes(
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.00, 0.05), size_percent=(0.02, 0.25),
                            per_channel=0.2
                        ),
                    ]),
                ),

            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def generate(self, images, segmaps=None):
        t, n, h, w, c = images.shape
        images = images.reshape(t * n, h, w, c)
        if segmaps is None:
            images = self.seq(images=(images*255).astype(np.uint8))
        else:
            segmaps = segmaps.reshape(t * n, h, w, 1)
            images, segmaps = self.seq(images=(images*255).astype(np.uint8), segmentation_maps=segmaps)
            segmaps = segmaps.reshape(t, n, h, w, 1)

        images = images.reshape(t, n, h, w, c)
        if segmaps is None:
            return (images/255.0).astype(np.float32)
        else:
            return (images/255.0).astype(np.float32), segmaps

