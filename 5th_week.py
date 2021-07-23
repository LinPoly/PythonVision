# %%
import numpy as np
from numpy.linalg import inv
import cv2 as cv

# %%
"""assigment 2"""
# choose coordinates manually
img_path_1 = './assignments/second_assignment/Image A.jpg'
img_path_2 = './assignments/second_assignment/Image B.jpg'
# the coordinate pairs are selected using windows paint 2D.
corr_coords = [((1211, 1445), (984, 1015)), ((2442, 2169), (1989, 2030)), ((1182, 1712), (885, 1265)),
               ((932, 2287), (500, 1757)), ((1768, 1204), (1584, 926)), ((1375, 1192), (1207, 812)),
               ((1373, 1845), (1037, 1447))]
corr_matrix = np.zeros([2, 3, 7], dtype=np.int32)
corr_matrix[:, 2, :] = 1
for i in range(7):
    corr_matrix[0, :-1, i], corr_matrix[1, :-1, i] = corr_coords[i]
Q, P = corr_matrix[0], corr_matrix[1]
H = P@Q.T@inv(Q@Q.T)
img_a = cv.imread(img_path_1)
img_b = cv.imread(img_path_2)
size = (img_b.shape[1], img_b.shape[0])
trans_img = cv.warpAffine(img_a, H[:-1], size)
cv.imwrite('registered_img.jpg', trans_img)

# %%
"""assignment 3"""
import glob
import matplotlib.pyplot as plt
from pathlib import Path


def get_histogram(img: np.ndarray, path:str):
    plt.hist(img.flatten(), 256, [0, 255])
    plt.savefig(path)
    plt.close()


SrcDirt = Path('./assignments/third_assignment')
HistDirt = Path('./histograms')
HistTransDirt = Path('./hist_trans')
filenames = glob.glob(str(SrcDirt / '*.bmp'))
imgs = list()
names = list()
for fname in filenames:
    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    name = fname.split('\\')[-1].split('.')[0]
    names.append(name)
    # path = str(HistDirt / f'{name}.png')
    # get_histogram(img, path)
    imgs.append(img)

# %%
for img,name in zip(imgs, names):
    eq_img = cv.equalizeHist(img)
    cmp_img = np.hstack((eq_img, img))
    cv.imwrite(str(HistTransDirt / f'{name}_histequal.bmp'), cmp_img)

# %%
from functools import partial


def hist_matching(ref_img: np.ndarray, src_img: np.ndarray):
    """set the histogram of original image as the target"""
    hist_func = partial(np.histogram, bins=np.arange(256), density=True)
    (ref_density, _), (src_density, _) = map(hist_func, (ref_img, src_img))
    def _map(density: np.ndarray):
        map_from = np.uint8(255*np.cumsum(density))
        map_list = list(zip(np.arange(256), map_from))
        return map_list
    ref_maps, src_maps = map(_map, (ref_density, src_density))
    merged_maps = list()
    for src_map in src_maps:
        start_pos = 0
        src_value = src_map[1]
        # lists are inherently sorted so only need to traverse one time,
        # use while instead of for to record a start position for
        # next search
        while start_pos < len(ref_maps):
            ref_value_1 = ref_maps[start_pos][1]
            if ref_value_1 >= src_value:
                break
            else:
                start_pos += 1
        ref_value_2 = ref_maps[start_pos - 1][1]
        if (ref_value_1 == src_value or
                ref_value_1 - src_value < ref_value_2 - src_value):
            pos = start_pos
        else:
            pos = start_pos - 1
        # start_pos == 0 needn't being treated as except,
        # cuz ref_maps[-1][1] is definitely greater than ref_maps[0][1]
        merged_maps.append((src_map[0], ref_maps[pos][0]))
    for map_from, map_to in merged_maps:
        indices = np.where(src_img == map_from)
        src_img[tuple(indices)] = map_to
    return src_img


HistMatchDirt = Path('./hist_matching')
# give positions of target images directly, or derive it by string matching.
ref_positions = [0, 3, 7, 11, 14]
for i in range(len(ref_positions)-1):
    this_pos = ref_positions[i]
    next_pos = ref_positions[i+1]
    for j in range(this_pos+1, next_pos):
        matched_img = hist_matching(imgs[7], imgs[j])
        cv.imwrite(str(HistMatchDirt/f'{names[j]}_histmatch.jpg'), matched_img)

# %%
from functools import partial


def img_enhance(img: np.ndarray, mean_para, var_para: tuple, enhance_para):
    assert mean_para > 0 and isinstance(mean_para, (int, float))
    ori_img = img
    window_size = 7
    pad_size = int((window_size-1)/2)
    glb_mean = np.mean(ori_img)
    glb_var = np.var(ori_img)
    x_start = y_start = pad_size
    x_stop, y_stop = ori_img.shape[0] + pad_size, ori_img.shape[1] + pad_size
    padded_img = np.pad(ori_img, (pad_size, pad_size), 'reflect')
    for i in range(x_start, x_stop):
        for j in range(y_start, y_stop):
            neighborhood = padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            ngh_mean = np.mean(neighborhood)
            ngh_var = np.var(neighborhood)
            if (var_para[0]*glb_var < ngh_var < var_para[1]*glb_var and
                    ngh_mean < mean_para*glb_mean):
                ori_img[i - pad_size, j - pad_size] *= enhance_para
    return ori_img


lena_img = cv.imread('./assignments/third_assignment/lena.bmp', cv.IMREAD_GRAYSCALE)
elain_img = cv.imread('./assignments/third_assignment/elain.bmp', cv.IMREAD_GRAYSCALE)
enh_func = partial(img_enhance, mean_para=0.5, var_para=(0.02, 0.8), enhance_para=5)
lena_enhance, elain_enhance = map(enh_func, (lena_img, elain_img))
cv.imwrite('./hist_enhance/lena_enh.bmp', lena_enhance)
cv.imwrite('./hist_enhance/elain_enh.bmp', elain_enhance)

# %%
def threshold_segment_binary(img:np.ndarray, threshold, tolerance):
    t = threshold
    while True:
        indices_bg = np.where(img < t)
        indices_obj = np.where(img >= t)
        bg_group, obj_group = img[tuple(indices_bg)], img[tuple(indices_obj)]
        mean_bg, mean_obj = map(np.mean, (bg_group, obj_group))
        new_t = (mean_bg+mean_obj)/2
        if np.abs(new_t-t) < tolerance:
            break
        t = new_t
    img[tuple(indices_bg)] = mean_bg
    img[tuple(indices_obj)] = mean_obj
    return img


lena_img = cv.imread('./assignments/third_assignment/lena.bmp', cv.IMREAD_GRAYSCALE)
women_img = cv.imread('./assignments/third_assignment/woman.BMP', cv.IMREAD_GRAYSCALE)
cv.imwrite('./hist_seg/lena_seg.bmp', threshold_segment_binary(lena_img, 70, 1))
cv.imwrite('./hist_seg/women_seg.bmp', threshold_segment_binary(women_img, 70, 1))
