# %%
import numpy as np
from pathlib import Path
import cv2 as cv


# %%
def gsn_filter(sigma, size, img: np.ndarray):
    gsn_filter = np.zeros([size, size])
    center = int((size - 1) / 2)
    for i in range(size):
        for j in range(size):
            dist = (i-center) ** 2 + (j - center) ** 2
            gsn_filter[i, j] = np.exp(-dist / sigma ** 2)
    gsn_filter = gsn_filter / (2 * np.pi * sigma ** 2)
    pd_img = np.pad(img, center, mode='reflect')
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(center, img.shape[0] + center):
        for j in range(center, img.shape[1] + center):
            v = np.sum(pd_img[i-center:i+center+1, j-center:j+center+1]*gsn_filter)
            new_img[i-center, j-center] = v
    factor = 255/new_img.max()
    new_img = new_img*factor
    return new_img


# %%
def md_filter(size, img: np.ndarray):
    center = int((size - 1) / 2)
    new_img = np.zeros(img.shape, dtype=np.uint8)
    pd_img = np.pad(img, center, mode='reflect')
    for i in range(center, img.shape[0] + center):
        for j in range(center, img.shape[1] + center):
            new_img[i - center, j - center] = np.median(pd_img[i-center:i+center+1,
                                                        j-center:j+center+1])
    return new_img


# %%
def unsharp_masking(img: np.ndarray, k):
    # all-1 filter to blur the image
    pd_img = np.pad(img, 1, mode='constant')
    blr_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            blr_img[i-1, j-1] = np.sum(pd_img[i-1:i+2, j-1:j+2])
    mask_img = img-blr_img
    shp_img = img+k*mask_img.astype(np.uint8)
    return shp_img


def laplace(img: np.ndarray):
    pd_img = np.pad(img, 1, mode='constant')
    new_img = np.zeros(img.shape, dtype=np.uint8)
    lp_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.uint8)
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            new_img[i-1, j-1] = np.sum(pd_img[i-1:i+2, j-1:j+2]*lp_filter)
    return new_img


def sobel(img: np.ndarray):
    pd_img = np.pad(img, 1, mode='constant')
    new_img = np.zeros(img.shape, dtype=np.uint8)
    filter1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.uint8)
    filter2 = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint8)
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            neighbor = pd_img[i-1:i+2, j-1:j+2]
            new_img[i-1, j-1] = np.abs(np.sum(neighbor*filter1))+np.abs(np.sum(neighbor*filter2))
    return new_img


# %%
ImgPath = Path('./assignments/fourth_assignment')
test1_img = cv.imread(str(ImgPath/'test1.pgm'), cv.IMREAD_GRAYSCALE)
test2_img = cv.imread(str(ImgPath/'test2.tif'), cv.IMREAD_GRAYSCALE)
test3_img = cv.imread(str(ImgPath/'test3_corrupt.pgm'), cv.IMREAD_GRAYSCALE)
test4_img = cv.imread(str(ImgPath/'test4.tif'), cv.IMREAD_GRAYSCALE)
sizes = [3, 5, 7]
for size in sizes:
    test1_md_filtered = md_filter(size, test1_img)
    test2_md_filtered = md_filter(size, test2_img)
    test1_gs_filtered = gsn_filter(1.5, size, test1_img)
    test2_gs_filtered = gsn_filter(1.5, size, test2_img)
    cv.imwrite(f'./test1/spatial/test1_md_{size}_filtered.png', test1_md_filtered)
    cv.imwrite(f'./test2/spatial/test2_md_{size}_filtered.png', test2_md_filtered)
    cv.imwrite(f'./test1/spatial/test1_gs_{size}_filtered.png', test1_gs_filtered)
    cv.imwrite(f'./test2/spatial/test2_gs_{size}_filtered.png', test2_gs_filtered)

# %%
from functools import partial

canny = partial(cv.Canny, threshold1=100, threshold2=200)
unsharp = partial(unsharp_masking, k=2)
func_list = [laplace, sobel]
for func in func_list:
    ts3_prcs, ts4_prcs = map(func, [test3_img, test4_img])
    cv.imwrite(f'./test3/spatial/test3_{func.__name__}_filtered.png', ts3_prcs)
    cv.imwrite(f'./test4/spatial/test4_{func.__name__}_filtered.png', ts4_prcs)

# %%
ts3_prcs, ts4_prcs = map(unsharp, [test3_img, test4_img])
cv.imwrite('./test3/spatial/test3_unsharp_filtered.png', ts3_prcs)
cv.imwrite('./test4/spatial/test4_unsharp_filtered.png', ts4_prcs)
ts3_prcs, ts4_prcs = map(canny, [test3_img, test4_img])
cv.imwrite('./test3/spatial/test3_canny_filtered.png', ts3_prcs)
cv.imwrite('./test4/spatial/test4_canny_filtered.png', ts4_prcs)

# %%
def frq_filtering(img: np.ndarray, flt: np.ndarray):
    img = img.astype(np.float32)/255
    f_trans = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f_trans, axes=[0, 1])
    f_shift *= flt
    f_trans = np.fft.ifftshift(f_shift, axes=[0, 1])
    prcs_img = cv.idft(f_trans, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    return prcs_img


def filter_generate(shape: tuple, func):
    center = [int(k/2) for k in shape[:2]]
    flt = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist_sqr = ((i-center[0])**2+(j-center[1])**2)
            flt[i, j, :] = func(dist_sqr)
    return flt


def gs_generate(dist_sqr, cf_dist):
    return np.exp(-dist_sqr/(2*cf_dist**2))


def bw_generate(dist_sqr, cf_dist, n):
    return 1/(1+(dist_sqr/cf_dist**2)**n)


def id_generate(dist_sqr, cf_dist):
    if dist_sqr < cf_dist**2:
        return 1
    else:
        return 0


# %%
def lp_generate(dist_sqr):
    return -4*np.pi**2*dist_sqr


# %%
from functools import partial

ts1_img = cv.imread('./assignments/fifth_assignment/test1.pgm', 0)
ts2_img = cv.imread('./assignments/fifth_assignment/test2.tif', 0)
cf_list = [20, 50, 100, 200]


def img_scale(ori_img: np.ndarray, prc_img: np.ndarray):
    factor = (ori_img.max()-ori_img.min())/(prc_img.max()-prc_img.min())
    prc_img = (prc_img-prc_img.min())*factor+ori_img.min()
    prc_img = prc_img.astype(np.uint8)
    return prc_img


for cf_dist in cf_list:
    gs_func = partial(gs_generate, cf_dist=cf_dist)
    bw_func = partial(bw_generate, cf_dist=cf_dist, n=2)
    gs_flt_1 = filter_generate((256, 256, 2), gs_func)
    gs_flt_2 = filter_generate((512, 512, 2), gs_func)
    bw_flt_1 = filter_generate((256, 256, 2), bw_func)
    bw_flt_2 = filter_generate((512, 512, 2), bw_func)
    gs_ts1 = frq_filtering(ts1_img, gs_flt_1)
    bw_ts1 = frq_filtering(ts1_img, bw_flt_1)
    gs_ts2 = frq_filtering(ts2_img, gs_flt_2)
    bw_ts2 = frq_filtering(ts2_img, bw_flt_2)
    gs_ts1 = img_scale(ts1_img, gs_ts1)
    gs_ts2 = img_scale(ts2_img, gs_ts2)
    bw_ts1 = img_scale(ts1_img, bw_ts1)
    bw_ts2 = img_scale(ts2_img, bw_ts2)
    cv.imwrite(f'./test1/frequency/test1_gs_{cf_dist}.png', gs_ts1)
    cv.imwrite(f'./test1/frequency/test1_bw_{cf_dist}.png', bw_ts1)
    cv.imwrite(f'./test2/frequency/test2_gs_{cf_dist}.png', gs_ts2)
    cv.imwrite(f'./test2/frequency/test2_bw_{cf_dist}.png', bw_ts2)

# %%
ts3_img = cv.imread('./assignments/fifth_assignment/test3_corrupt.pgm', 0)
ts4_img = cv.imread('./assignments/fifth_assignment/test4.tif', 0)
cf_list = [20, 50, 100, 200]

# %%
for cf_dist in cf_list:
    gs_func = partial(gs_generate, cf_dist=cf_dist)
    bw_func = partial(bw_generate, cf_dist=cf_dist, n=2)
    gs_flt_1 = 1 - filter_generate((133, 134, 2), gs_func)
    gs_flt_2 = 1 - filter_generate((512, 512, 2), gs_func)
    bw_flt_1 = 1 - filter_generate((133, 134, 2), bw_func)
    bw_flt_2 = 1 - filter_generate((512, 512, 2), bw_func)
    gs_ts1 = frq_filtering(ts3_img, gs_flt_1)
    bw_ts1 = frq_filtering(ts3_img, bw_flt_1)
    gs_ts2 = frq_filtering(ts4_img, gs_flt_2)
    bw_ts2 = frq_filtering(ts4_img, bw_flt_2)
    gs_ts1 += ts3_img.astype(np.float32)/1000
    bw_ts1 += ts3_img.astype(np.float32)/1000
    gs_ts2 += ts4_img.astype(np.float32)/1000
    bw_ts2 += ts4_img.astype(np.float32)/1000
    gs_ts1 = img_scale(ts4_img, gs_ts1)
    gs_ts2 = img_scale(ts4_img, gs_ts2)
    bw_ts1 = img_scale(ts4_img, bw_ts1)
    bw_ts2 = img_scale(ts4_img, bw_ts2)
    cv.imwrite(f'./test3/frequency/test3_gs_{cf_dist}.png', gs_ts1)
    cv.imwrite(f'./test3/frequency/test3_bw_{cf_dist}.png', bw_ts1)
    cv.imwrite(f'./test4/frequency/test4_gs_{cf_dist}.png', gs_ts2)
    cv.imwrite(f'./test4/frequency/test4_bw_{cf_dist}.png', bw_ts2)

# %%
func = partial(gs_generate, cf_dist=50)
flt = filter_generate((512, 512, 2), func)
prcs_img = frq_filtering(ts4_img, flt)
ori_img = ts4_img.astype(np.float32)/255
shp_img = ori_img-prcs_img
shp_img = img_scale(ts4_img, shp_img)
# factor = (ts3_img.max()-ts3_img.min())/(shp_img.max()-shp_img.min())
# shp_img *= factor
# shp_img += (ts3_img.min()-shp_img.min())
# shp_img = shp_img.astype(np.uint8)
cv.imwrite('./test4/frequency/test4_unmask.png', shp_img)
