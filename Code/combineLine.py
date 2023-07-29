import numpy as np
from gcn.graphVN import Grapher
import cv2
import timeit

from ocr.ocr import ocr_csd


def find_same_line_bboxes(poly, threshold,h):
    matrix = np.zeros((len(poly), 8), dtype=np.uint32)
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

def create_mask_y(matrix, y_threshold,h):
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

def group_boxes(matrix,w,threshold ): # cang tang cang noi cac tu xa nhau
    n = len(matrix)
    grouped_boxes = []
    current_group = [matrix[0]]

    for i in range(1, n):
        distance = abs((matrix[i][0] - matrix[i-1][2])*100 / w)
        # print(distance)
        # print()
        # print(f'{matrix[i][0]}- {matrix[i-1][2]} = {distance}')
        if distance > threshold:
            grouped_boxes.append(current_group)
            current_group = [matrix[i]]
        else:
            current_group.append(matrix[i])

    grouped_boxes.append(current_group)  # Add the last group

    return grouped_boxes

def sort_x(chain,w,thresh):
    if chain.shape[0] == 1:
        return [chain]
    else:
        chain = chain[chain[:,0].argsort()]
        return(group_boxes(chain,w,threshold=thresh))

def sentence_cluster(matrix,mask,w,thresh):
    out_put = []
    flag = mask[-1]
    for inter in range(flag):
        work = np.where(mask == inter+1)
        # matrix[work[0]] = sort_x(matrix[work[0]])
        out_put.append(sort_x(matrix[work[0]],w,thresh))
    return out_put

def combination(chyong):
    chyong = np.array(chyong)
    xmin = int(np.min(chyong[:,[0,2,4,6]]))
    ymin = int(np.min(chyong[:,[1,3,5,7]]))
    xmax = int(np.max(chyong[:,[0,2,4,6]]))
    ymax = int(np.max(chyong[:,[1,3,5,7]]))
    return xmin,ymin,xmax,ymax

def end2end(matrix,img,model):
    h,w,_ = img.shape
    # matrix = np.zeros((len(polyy), 8), dtype=np.int32)
    matrix, mask = create_mask_y(matrix,1.0,h)
    cluster = sentence_cluster(matrix,mask,w,1.5)
    box = []
    texts = []
    for clus in cluster:
        for cl in clus:
            xmin,ymin,xmax,ymax = combination(cl)
            box.append([xmin,ymin,xmax,ymax])
            texts.append(ocr_csd([xmin,ymin,xmax,ymax],img,model))
    return box,texts

if __name__ == "__main__":
    path = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013aukcu.txt'
    with open(path,'r') as f:
        poly = f.readlines()

    img = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/result/res_mcocr_public_145013aukcu.jpg')
    h,w,_ = img.shape
    matrix = np.zeros((len(poly), 8), dtype=np.int64)
    for i, polyy in enumerate(poly):
        component = polyy.strip().split(",")
        matrix[i] = [int(inter) for inter in component]

    cluster = end2end(matrix,h,w)

    canvas = cv2.imread('/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/images/mcocr_public_145013aukcu.jpg')
    color = (0, 0, 255)  # Green color (BGR format)
    thickness = 2  # Thickness of the box's border

    for clus in cluster:
        for cl in clus:
            xmin, ymin, xmax, ymax =combination(cl)
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, thickness)

    cv2.imshow("ve thoi",cv2.resize(canvas,(600,800)))
    cv2.imwrite("done_combination.jpg",canvas)
    cv2.waitKey(0)
            
            




