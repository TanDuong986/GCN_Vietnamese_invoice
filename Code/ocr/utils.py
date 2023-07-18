import numpy as np 

def read_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    # Convert the data into a matrix
    matrix = np.zeros((len(lines), 8), dtype=int)
    for i, line in enumerate(lines):
        values = line.strip().split(',') # strip is not take /n, split is its name
        matrix[i] = [int(val) for val in values]
    return matrix