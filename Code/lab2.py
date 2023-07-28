def group_boxes(matrix, threshold):
    n = len(matrix)
    grouped_boxes = []
    current_group = [matrix[0]]

    for i in range(1, n):
        
        distance = matrix[i][0] - matrix[i-1][2]

        if distance > threshold:
            grouped_boxes.append(current_group)
            current_group = [matrix[i]]
        else:
            current_group.append(matrix[i])

    grouped_boxes.append(current_group)  # Add the last group

    return grouped_boxes
import numpy as np

# Example matrix
matrix = [[206, 125 ,225, 125 ,225 ,145, 206 ,145],
 [225, 126, 249, 126 ,249 ,146 ,225, 146],
 [249, 126, 278, 126, 278 ,146, 249, 146],
 [277 ,128 ,305, 128, 305, 146, 277, 146],
 [308, 129, 317, 129, 317 ,145, 308, 145],
 [321, 128 ,350, 128, 350, 146, 321, 146],
 [351 ,126 ,386, 132, 382, 152, 347, 145]]

# threshold_distance = 10  # Set your desired threshold distance here

# result = group_boxes(matrix, threshold_distance)
# for rs in result:
#     print(rs)
matrix = np.array(matrix)

# Find the minimum value among columns 1, 3, 5, and 7
min_value = np.min(matrix[:, [0, 2, 4, 6]])
print(min_value)