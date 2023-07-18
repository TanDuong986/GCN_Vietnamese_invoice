import cv2
import numpy as np
from PIL import Image

def cv2image(image): #convert cut image into input of ocr
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw(frame,img):
    ts = 1
    cl = (0,0,255)
    # Define the coordinates of the two points
    point1 = tuple(frame[0:2])
    point2 = tuple(frame[2:4])
    point3 = tuple(frame[4:6])
    point4 = tuple(frame[6:8])
    print(point1,point2)
    # Draw a line between the two points

    cv2.line(img, point1, point2, cl, thickness=ts)
    cv2.line(img, point2, point3, cl, thickness=ts)
    cv2.line(img, point3, point4, cl, thickness=ts)
    cv2.line(img, point4, point1, cl, thickness=ts)

def stretch_ROI(roi,src):
    '''
    Input : roi is coordinate size (8*1)
            src : source image
            
    Output : aligmented image'''
    offset = 5 # pixels

    old_coor = roi.reshape(-1,2)
    x_ = old_coor[:,0]
    y_ = old_coor[:,1]

    xmin = min(x_)
    xmax = max(x_)
    ymin = min(y_)
    ymax = max(y_)
    small_image = src[ymin-offset:ymax+offset,xmin-offset:xmax+offset]

    size = src.shape[0:2] # h,w
    new_coor = np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]).reshape(-1,2)
    #convert matrix 
    cvm = cv2.getPerspectiveTransform(old_coor.astype(np.float32),new_coor.astype(np.float32))
    result = cv2.warpPerspective(src, cvm, size[::-1])


instruct = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013alybg.txt'
# Read the text file
with open(instruct, 'r') as file:
    lines = file.readlines()

# Convert the data into a matrix
matrix = np.zeros((len(lines), 8), dtype=int)
for i, line in enumerate(lines):
    values = line.strip().split(',') # strip is not take /n, split is its name
    matrix[i] = [int(val) for val in values]

csd = matrix[26].reshape(-1,2)
x_ = csd[:,0]
y_ = csd[:,1]

xmin = min(x_)
xmax = max(x_)
ymin = min(y_)
ymax = max(y_)
new_point = np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]).reshape(-1,2)
img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/result/res_mcocr_public_145013alybg.jpg')
size = img.shape[0:2]
cvm = cv2.getPerspectiveTransform(csd.astype(np.float32),new_point.astype(np.float32))

per_img = cv2.warpPerspective(img,cvm,size[::-1])

# draw(new_point.reshape(8,),img)
draw(new_point.reshape(8,),per_img)
# cv2.imshow("per",per_img[ymin:ymax,xmin:xmax])
# imgg = np.hstack((img,per_img))
print(new_point)
cv2.imshow("raw",per_img)
cv2.waitKey(0)