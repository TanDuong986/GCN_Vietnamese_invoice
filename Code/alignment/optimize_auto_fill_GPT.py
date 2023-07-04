import cv2
import numpy as np
import os
import argparse

def add_padding(img, padding_size=30, padding_color=(0, 0, 0)):
    top, bottom, left, right = padding_size, padding_size, padding_size, padding_size
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return pad_img

def dilate(img, padd=30):
    h, w = img.shape[0:2]
    img = img[padd:h-padd, padd:w-padd]
    return img

def puring(img):
    padd_size = 30
    threshold_area = 100000
    threshold_length = 1000
    thresh_canny = [50, 150]

    img = add_padding(img, padd_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cny = cv2.Canny(thresh, *thresh_canny)
    cny = cv2.dilate(cny, np.ones((3, 3)), iterations=1)
    contours, _ = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, False)
        if length > threshold_length and area > threshold_area:
            filtered_contours.append(contour)

    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, 0, 1, thickness=-1)

    tmp = mask * 255
    cny = cv2.Canny(tmp, *thresh_canny)
    cny = cv2.dilate(cny, np.ones((3, 3)), iterations=1)

    return dilate(mask), dilate(cny)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str
                        ,default=('/home/dtan/Documents/GCN/GCN_Vietnam/Code/U2Net/output/mcocr_private_145120pxmgi.jpg'),
                        help="source image")
    otp = parser.parse_args()

    img = cv2.imread(otp.source)
    name_image = os.path.splitext(os.path.basename(otp.source))[0]
    rgb = cv2.imread(os.path.join('Vietnam_invoice_data/mcocr2021_raw/test/test_images', name_image + ".jpg"))
    puring(img)
