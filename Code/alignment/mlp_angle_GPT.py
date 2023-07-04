import cv2
import numpy as np
import os
from auto_fill import puring

def norm_line(lines):
    lines = lines.reshape(lines.shape[0],-1)
    dist = lines[:, 0]
    angle = lines[:, 1]

    dist = np.array([dt if dt > 0 else np.abs(dt) for dt in dist]).reshape(-1, 1)
    angle = np.array([value if (value < np.pi / 2) else (np.pi - value) for value in angle]).reshape(-1, 1)
    rs = np.concatenate((dist, angle), axis=1)
    return rs

def sort_line(lines):
    variance_rho = 30
    mask = norm_line(lines)
    if mask.shape[0] == 1:
        return mask
    sm = mask[mask[:, 0].argsort()]
    groups = []
    current_group = []

    for i in range(sm.shape[0] - 1):
        current_group.append(sm[i])
        diff = sm[i + 1, 0] - sm[i, 0]
        if diff >= variance_rho:
            groups.append(np.array(current_group))
            current_group = []

    current_group.append(sm[-1])
    groups.append(np.array(current_group))

    averages = np.array([np.average(group, axis=0) for group in groups])

    return averages

def detect_angle(image):
    # cl_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=415)

    if lines is not None:
        mask = sort_line(lines)

        return mask

if __name__ == "__main__":
    path_img = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/U2Net/output/mcocr_private_145120pxmgi.jpg'
    anh = cv2.imread(path_img)
    name_image = os.path.splitext(os.path.basename(path_img))[0]
    rgb = cv2.imread(os.path.join('/home/dtan/Documents/GCN/GCN_Vietnam', 'Vietnam_invoice_data/mcocr2021_raw/test/test_images', name_image + ".jpg"))
    mask, cny = puring(anh)
    list_angle = detect_angle(cny)
