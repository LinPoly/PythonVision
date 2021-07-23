# This is a code segment running in scientific mode on PyCharm
# Requirements: python3.8, numpy 1.20, cv2

# %%
import cv2
import numpy as np
from pathlib import Path

PATH = Path('./first_assignment')
image_elain = cv2.imread(str(PATH/'elain1.bmp'), cv2.IMREAD_GRAYSCALE)
image_lena = cv2.imread(str(PATH/'lena.bmp'), cv2.IMREAD_GRAYSCALE)

# %%
# calculate the mean value and variance of lena
mean = np.mean(image_lena)
variance = np.var(image_lena)

# %%
class ProcessedImages:
    def __init__(self, input_image: np.ndarray, flag: bool = True):
        self.image = input_image
        self.flag = flag

    def grayscale_tune(self, to_scale):
        assert isinstance(to_scale, int)
        delta = self.image.max() - self.image.min()
        self.image = self.image-self.image.min()
        range_list = np.zeros([to_scale+2])
        range_list[-1] = delta
        for i in range(to_scale):
            range_list[i+1] = np.uint8((delta/(to_scale+1))*(i+1))
        tuned_img = np.zeros(self.image.shape, dtype=np.uint8)
        for i in range(to_scale+1):
            indices = np.where((self.image >= range_list[i]) &
                               (self.image < range_list[i + 1]))
            tuned_img[tuple(indices)] = range_list[i]*255/range_list[-2]
        indices = np.where(self.image == delta)
        tuned_img[tuple(indices)] = 255
        return ProcessedImages(tuned_img)

    def shear_image(self, shear_rate=1.5, shape=None):
        if shape == None:
            shape = self.image.shape
        self.shear_rate = shear_rate
        M = np.array([[1, 0, 0], [1, self.shear_rate, 0]])
        affine_image = cv2.warpAffine(self.image, M, shape)
        return ProcessedImages(affine_image)

    def rotate_image(self, angle = 30, shape=None):
        self.angle = angle
        if shape == None:
            shape = self.image.shape
        center = tuple([int(i/2) for i in shape])
        M = cv2.getRotationMatrix2D(center, self.angle, 1)
        rotated_image = cv2.warpAffine(self.image, M, shape)
        return ProcessedImages(rotated_image)

    def resize_image(self, inter, fx, fy):
        resized_image = cv2.resize(self.image, None, fx=fx, fy=fy, interpolation=inter)
        return ProcessedImages(resized_image)


Lena = ProcessedImages(input_image=image_lena)
Elain = ProcessedImages(input_image=image_elain)

# %%
def three_step_trans(image, angle, shear_rate, inter):
    sheared_img = image.shear_image(shear_rate)
    rotated_img = sheared_img.rotate_image(angle)
    resized_img = rotated_img.resize_image(inter, 4, 4)
    return resized_img.image
