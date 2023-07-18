import numpy as np
import cv2 

instruct = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013alybg.txt'

def draw(frame,img):
    ts = 2
    cl = (0,0,255)
    # Define the coordinates of the two points
    point1 = frame[0:2]
    point2 = frame[2:4]
    point3 = frame[4:6]
    point4 = frame[6:8]
    # Draw a line between the two points

    # cv2.line(img, point1, point2, cl, thickness=ts)
    # cv2.line(img, point2, point3, cl, thickness=ts)
    # cv2.line(img, point3, point4, cl, thickness=ts)
    # cv2.line(img, point4, point1, cl, thickness=ts)

# Read the text file
with open(instruct, 'r') as file:
    lines = file.readlines()

# Convert the data into a matrix
matrix = np.zeros((len(lines), 8), dtype=int)
for i, line in enumerate(lines):
    values = line.strip().split(',') # strip is not take /n, split is its name
    matrix[i] = [int(val) for val in values]

import os
here = os.path.dirname(os.path.abspath(__file__))
i=0
img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013alybg.jpg')
for m in matrix:
    draw(m,img)
    cv2.imwrite(f'{here}/ROI/sub_{str(i)}.jpg',img[m[1]:m[5],m[0]:m[4]])
    i+=1
# cv2.imshow("img",img)
# cv2.waitKey(0)