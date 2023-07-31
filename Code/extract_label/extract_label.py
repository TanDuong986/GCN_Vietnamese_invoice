import json
import csv
import pandas as pd
import ast

def convert_to_list_of_integers(input_string):
    return ast.literal_eval(input_string)

def convert_to_list(input_string):
    # Remove leading '[' and trailing ']' from the string and split by comma
    return [int(num) for num in input_string[1:-1].split(', ')]

col = ['cells', 'poly', 'cate_id', 'cate_text', 'vietocr_text', 'group_id']
with open('/home/dtan/Documents/GCN/GCN_Vietnam/Vietnam_invoice_data/preprocessed_data/mcocr_labels.json','r') as f:
    data = json.load(f)
for i in data.keys(): #name of all image
    employee_data = data[i]
    df = pd.DataFrame(employee_data["cells"],columns=["poly", "cate_id", "cate_text", "vietocr_text", "group_id"])
    df.drop(["group_id"], axis=1, inplace=True)
    df.to_csv('/home/dtan/Documents/GCN/GCN_Vietnam/label_mcocr2021/'+str(i) + '.csv',index=False) 