import numpy as np
from gcn.graphVN import Grapher
import cv2
import timeit

# def cook_x(chain):
    # chinku = chain[chain[:,0].argsor()]
    # print(chinku)

def find_same_line_bboxes(poly, threshold):
    matrix = np.zeros((len(poly), 8), dtype=np.uint64)
    for i, polyy in enumerate(poly):
        component = polyy.strip().split(",")
        matrix[i] = [int(inter) for inter in component]

    matrix = matrix[matrix[:, 1].argsort()]
    prev_y = matrix[0, 1]
    line_idx = 1
    mask = [1]

    for y in matrix[1:, 1]:
        gap_y = np.round(((y - prev_y) / h) * 100, 5)
        if gap_y > threshold:
            line_idx += 1
        mask.append(line_idx)
        prev_y = y

    return mask

def create_mask_y(matrix, y_threshold):
    # Sort by y-coordinate
    matrix = matrix[matrix[:, 1].argsort()]
    
    prev_y = matrix[0, 1]
    line_idx = 1
    mask = [1]

    for y in matrix[1:, 1]:
        gap_y = np.round(((y - prev_y) / h) * 100, 5)
        if gap_y > y_threshold:
            line_idx += 1
        mask.append(line_idx)
        prev_y = y
    return matrix,np.array(mask)

def group_boxes(matrix,threshold = 1.0):
    n = len(matrix)
    grouped_boxes = []
    current_group = [matrix[0]]

    for i in range(1, n):
        distance = abs((matrix[i][0] - matrix[i-1][2])*100 / w)
        # print(f'{matrix[i][0]}- {matrix[i-1][2]} = {distance}')
        if distance > threshold:
            grouped_boxes.append(current_group)
            current_group = [matrix[i]]
        else:
            current_group.append(matrix[i])

    grouped_boxes.append(current_group)  # Add the last group

    return grouped_boxes

def sort_x(chain):
    if chain.shape[0] == 1:
        return [chain]
    else:
        chain = chain[chain[:,0].argsort()]
        return(group_boxes(chain))

def sentence_cluster(matrix,mask,x_thresh):
    out_put = []
    flag = mask[-1]
    for inter in range(flag):
        work = np.where(mask == inter+1)
        # matrix[work[0]] = sort_x(matrix[work[0]])
        out_put.append(sort_x(matrix[work[0]]))
    return out_put
        


path = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_46056287dfaa0cf455bb.txt'
with open(path,'r') as f:
    poly = f.readlines()
img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_46056287dfaa0cf455bb.jpg')
h,w,_ = img.shape
matrix = np.zeros((len(poly), 8), dtype=np.int64)
for i, polyy in enumerate(poly):
    component = polyy.strip().split(",")
    matrix[i] = [int(inter) for inter in component]
matrix, mask = create_mask_y(matrix,1.0)
cluster = sentence_cluster(matrix,mask,1.0)

for clus in cluster:
    for cl in clus:
        for c in cl:
            # print(c)
            print(type(c))




#### input is matrix and thresh of y, x
#### ouput is tuple of matrix that contains rs[cluster[polygon[],[],[]],[],[]]