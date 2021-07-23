# %%
import numpy as np
import cv2


def motion_blur(image, degree=24, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    # motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


img = cv2.imread('./assignments/sixth_assignment/lena.bmp', 0)
img_ = motion_blur(img)

# %%
import numpy as np
import cv2 as cv

def img_scale(ori_img: np.ndarray) -> np.ndarray:
    # This function can be modified since the scaling range
    # is not certain.
    factor = 255/(ori_img.max() - ori_img.min())
    scaled_img = factor*(ori_img - ori_img.min())
    scaled_img = scaled_img.astype(np.uint8)
    return scaled_img


# %%
def blur_flt(shape, a, b, T) -> np.ndarray:
    flt = np.zeros(shape, dtype=complex)
    it = np.nditer(flt, flags=['multi_index'], op_flags=['readwrite'])
    flt[0, 0] = T
    it.__next__()
    for x in it:
        f = np.pi * (a*it.multi_index[0]+b*it.multi_index[1])
        x[...] = T/f * np.sin(f) * np.exp(-1j*f)
    return flt


# %%
def blur_flt_1(shape, a, b, T) -> np.ndarray:
    # use this.
    flt = np.zeros(shape, dtype=complex)
    it = np.nditer(flt, flags=['multi_index'], op_flags=['readwrite'])
    center = [i // 2 for i in shape]
    for x in it:
        f = np.pi * (a*(it.multi_index[0]-center[0])+b*(it.multi_index[1]-center[1]))
        x[...] = T* np.sinc(f) * np.exp(-1j*f)
    return flt


# %%
def blur_flt_2(shape, a, b, T) -> np.ndarray:
    flt = np.zeros(shape, dtype=np.float32)
    it = np.nditer(flt, flags=['multi_index'], op_flags=['readwrite'])
    center = [i // 2 for i in shape]
    for x in it:
        f = np.pi * (a*(it.multi_index[0]-center[0])+b*(it.multi_index[1]-center[1]))
        x[...] = T* np.sinc(f) # * np.exp(-1j*f)
    flt -= flt.min()
    return flt


# %%
def img_flt(img: np.ndarray, flt: np.ndarray) -> np.ndarray:
    # use this
    assert img.shape == flt.shape
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F, axes=[0, 1])
    Fshift *= flt
    G = np.fft.ifftshift(Fshift)
    f = np.real(np.fft.ifft2(G))
    f = img_scale(f)
    return f


# %%
def img_flt_1(img: np.ndarray, flt: np.ndarray) -> np.ndarray:
    assert img.shape == flt.shape
    G = np.fft.fft2(img)
    F = G*flt
    f = np.real(np.fft.ifft2(F))
    f = img_scale(f)
    return f