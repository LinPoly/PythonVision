# %%
import numpy as np
import cv2 as cv


# %%
def img_scale(ori_img: np.ndarray) -> np.ndarray:
    # This function can be modified since the scaling range
    # is not certain.
    factor = 255/(ori_img.max() - ori_img.min())
    scaled_img = factor*(ori_img - ori_img.min())
    scaled_img = scaled_img.astype(np.uint8)
    return scaled_img


# %%
def gauss_noise(m, v, img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    noise = np.random.normal(m, v, img.shape)
    img = img + noise
    return img_scale(img)


# %%
def salt_pepper_noise(img: np.ndarray) -> np.ndarray:
    ori_img = img
    entry_num = ori_img.shape[0] * ori_img.shape[1]
    data_num = np.int32(0.2*entry_num)
    rdn_sq = np.random.randint(0, entry_num, data_num)
    row_sq = rdn_sq // ori_img.shape[1]
    col_sq = rdn_sq % ori_img.shape[1]
    ori_img[row_sq[: data_num // 2 + 1], col_sq[: data_num // 2 + 1]] = 0
    ori_img[row_sq[data_num // 2 + 1:], col_sq[data_num // 2 + 1:]] = 255
    return ori_img


# %%
img = cv.imread('./assignments/sixth_assignment/lena.bmp', 0)
gs_img = gauss_noise(30, 10, img)
sp_img = salt_pepper_noise(img)
cv.imwrite('./gs_img.png', gs_img)
cv.imwrite('./sp_img.png', sp_img)

# %%
def mean_flt(img: np.ndarray, ws: tuple, mode: str, Q=None) -> np.ndarray:
    assert mode in ('arth', 'geo', 'harmo', 'ctharmo')
    num = ws[0] * ws[1]
    ps = [k // 2 for k in ws]
    img = img.astype(np.float32)
    img = np.pad(img, tuple(ps), mode='reflect')
    prc_img = np.zeros(img.shape, dtype=np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ngb = img[i: i+2*ps[0]+1, j: j+2*ps[1]+1]
            if mode == 'arth':
                prc_img[i, j] = np.sum(ngb) // num
            elif mode == 'geo':
                prc_img[i, j] = np.prod(ngb) ** (1/num)
            elif mode == 'harmo':
                prc_img[i, j] = num // np.sum(1 / (ngb + 0.01))
            else:
                if Q == None:
                    Q = 1
                if Q >= 0:
                    prc_img[i, j] = np.sum(ngb ** (Q + 1)) // (np.sum(ngb ** Q) + 0.01)
                else:
                    prc_img[i, j] = np.sum((ngb + 0.01) ** (Q + 1)) // np.sum((ngb + 0.01) ** Q)

    prc_img = img_scale(prc_img)
    return  prc_img


def stat_flt(img: np.ndarray, ws: tuple, mode: str) -> np.ndarray:
    assert mode in ('median', 'min', 'max')
    ps = [k // 2 for k in ws]
    img = np.pad(img, tuple(ps), mode='reflect')
    prc_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ngb = img[i: i+2*ps[0]+1, j: j+2*ps[1]+1]
            if mode == 'median':
                prc_img[i, j] = np.median(ngb)
            elif mode == 'min':
                prc_img[i, j] = np.min(ngb)
            else:
                prc_img[i, j] = np.max(ngb)
    return prc_img


# %%
def gauss_flt(img: np.ndarray, R) -> np.ndarray:
    Hs = np.zeros(img.shape, dtype=np.float32)
    center = [i // 2 for i in Hs.shape]
    it = np.nditer(Hs, flags=['multi_index'], op_flags=['readwrite'])

    for x in it:
        d = (it.multi_index[0]-center[0]) ** 2 + (it.multi_index[1]-center[1]) ** 2
        x[...] = np.exp(-d / (2 * R**2))

    G = np.fft.fft2(img)
    Gshift = np.fft.fftshift(G)
    Gshift *= Hs
    F = np.fft.ifftshift(Gshift)
    f = np.real(np.fft.ifft2(F))
    f = img_scale(f)
    return f


# %%
def blur_flt(shape, a, b, T) -> np.ndarray:
    flt = np.zeros(shape, dtype=complex)
    it = np.nditer(flt, flags=['multi_index'], op_flags=['readwrite'])
    center = [i // 2 for i in shape]
    for x in it:
        f = np.pi * (a*(it.multi_index[0]-center[0])+b*(it.multi_index[1]-center[1]))
        x[...] = T* np.sinc(f) * np.exp(-1j*f)
    return flt


# %%
def img_flt(img: np.ndarray, flt: np.ndarray) -> np.ndarray:
    assert img.shape == flt.shape
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F, axes=[0, 1])
    Fshift *= flt
    G = np.fft.ifftshift(Fshift)
    f = np.real(np.fft.ifft2(G))
    f = img_scale(f)
    return f


# %%
def mmse_flt(img: np.ndarray, ori_flt: np.ndarray, K) -> np.ndarray:
    G = np.fft.fft2(img)
    Gshift = np.fft.fftshift(G)
    abs_flt = np.absolute(ori_flt) ** 2
    Fshift = 1/ori_flt * abs_flt/(abs_flt+K) * Gshift
    F = np.fft.ifftshift(Fshift)
    f = np.real(np.fft.ifft2(F))
    prc_img = img_scale(f)
    return prc_img


# %%
def cmmse_flt(img: np.ndarray, flt: np.ndarray, K) -> np.ndarray:
    L = np.zeros(img.shape, dtype=np.float32)
    center = [i//2 for i in img.shape]

    it = np.nditer(L, flags=['multi_index'], op_flags=['readwrite'])
    for x in it:
        D = (it.multi_index[0]-center[0]) ** 2 + (it.multi_index[1]-center[1]) ** 2
        x[...] = 4 * np.pi**2 * D

    G = np.fft.fft2(img)
    Gshift = np.fft.fftshift(G)
    Fshift = np.conj(flt)/(np.abs(flt)**2 + K*np.abs(L)**2) * Gshift
    F = np.fft.ifftshift(Fshift)
    f = np.real(np.fft.ifft2(F))
    prc_img = img_scale(f)
    return prc_img
