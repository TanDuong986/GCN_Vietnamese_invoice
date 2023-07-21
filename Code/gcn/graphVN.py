import os
import cv2
import pandas as pd
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

class Grapher:
    """
    Description:
            This class is used to generate:
                    1) the graph (in dictionary form) { source_node: [destination_node1, destination_node2]}
                    2) the dataframe with relative_distances 

    Inputs: The class consists of a pandas dataframe consisting of cordinates for bounding boxe and the image of the invoice/receipt. 

    """
    def __init__(self, filename, data_fd):
        self.filename = filename
        self.data_fd = data_fd

        # tim path den file label box
        file_path = os.path.join(self.data_fd, "label_mcocr2021", 'mcocr_public_'+filename + '.jpg.csv') 
        
        #path cua image roi - concat file name -> absolute path of image
        image_path = os.path.join(self.data_fd, "images",  'mcocr_public_'+filename + '.jpg')

        #read box file not have label
        with open(file_path,'r') as f:
            self.df = pd.read_csv(f)
        # self.df.drop(["cells index","cate_id","group_id"], axis=1, inplace=True)
        
        #read that image
        self.image = cv2.imread(image_path)

    def convert_to_list(self,input_string):
        return [int(num) for num in input_string[1:-1].split(', ')]

    def graph_formation(self, export_graph = False):
        df, image = self.df, self.image
        """
        preprocessing the raw csv files to favorable df 
        """
        df['poly'] = df['poly'].apply(self.convert_to_list)

        # Create separate columns for each element in the lists
        df_split = pd.DataFrame(df['poly'].to_list(), columns=[f'pos{i+1}' for i in range(8)])

        # Concatenate the original DataFrame with the new split columns
        df = pd.concat([ df_split,df], axis=1)

        # Drop the original column with the string representations of lists
        df.drop(['poly','pos3','pos4','pos7','pos8'], axis=1, inplace=True)
        new_name_and_order = {'pos1':'xmin','pos2':'ymin','pos5':'xmax','pos6':'ymax','cate_id':'label_id','cate_text':'label_text','vietocr_text':'content'}
        # column = ['label_id','label_text','content','xmin','ymin','xmax','ymax']
        df = df[list(new_name_and_order.keys())].rename(columns=new_name_and_order)
        

        return df