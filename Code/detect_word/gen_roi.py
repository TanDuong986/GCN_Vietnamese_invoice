'''Display boxes on invoice to see
Input: txt instruction 8 point
Output: image with boxes drawed'''
import numpy as np
import cv2 
import os

instruct = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013aukcu.txt'
name_img = os.path.basename(instruct).split('.')[0]+'.jpg'
here = os.path.dirname(os.path.abspath(__file__))

def draw(frame,img,i):
    ts = 2
    cl = (0,0,255)
    # Define the coordinates of the two points
    point1 = frame[0:2]
    point2 = frame[2:4]
    point3 = frame[4:6]
    point4 = frame[6:8]
    # Draw a line between the two points

    cv2.line(img, point1, point2, cl, thickness=ts)
    cv2.line(img, point2, point3, cl, thickness=ts)
    cv2.line(img, point3, point4, cl, thickness=ts)
    cv2.line(img, point4, point1, cl, thickness=ts)
    cv2.putText(img,str(i),point1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)

# Read the text file
with open(instruct, 'r') as file:
    lines = file.readlines()

# Convert the data into a matrix
matrix = np.zeros((len(lines), 8), dtype=int)
for i, line in enumerate(lines):
    values = line.strip().split(',') # strip is not take /n, split is its name
    matrix[i] = [int(val) for val in values]

matrix = matrix[matrix[:,1].argsort()]

i=0
img = cv2.imread(os.path.join(here,'result',name_img))
for m in matrix:
    draw(m,img,i)
    # cv2.imwrite(f'{here}/ROI/sub_{str(i)}.jpg',img[m[1]:m[5],m[0]:m[4]])
    i+=1
cv2.imshow("img",img)
cv2.waitKey(0)