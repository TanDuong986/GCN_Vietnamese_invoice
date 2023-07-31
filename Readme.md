# :vietnam: Key Infomation Extraction on Vietnamese invoice

Structure of project:
```
GCN_Vietnamese
|__Code
|    |__aligment (preprocess raw image folder)
|    | |__angle_mlp.py (main)
|    |__detect_word 
|    | |__inference.py
|    | |__perspective.py
|    |__extract_label (explore raw label file into small file )
|    | |
|    | |__extract_label.py
|    |__gcn (create instance of Graph and insists of inference file)
|    | |__use
|    | |__graph.py (class of Graph)
|    | |__prepare (create instance and preprocessing data)
|    | |__train.py
|    | |__model.py
|    | |__gen_dataVN.py
|    | |__test_single.py
|    | |__inference.py
|    |__material (prepare data file for train GCN)
|    | |__test_data
|    | |__train_data
|    |__ocr (material for read text)
|    | |__ocr.py
|    |__U2Net (Segmentation background)
|    | |__u2net_test.py
|    |__Vietnam_invoice_data
|    |__mcocr2021_raw (raw data with background - not use) - 1934 images all
|    | |__mcocr_train_data
|    | |__mcocr_val_data
|    | |__test
|    |__preprocessed_data (preprocess data (rotated, delete back) use this) - 1090 images
|    | |__images (.jpg)
|    | |__label_mcocr2021 (.csv)
|__combineLine.py
|__fullGCN.py
```


File main is [fullGCN](/Code/fullGCN.py), This file combination of process have 5 parts. 

* Detect word with **CRAFT** call to [detect](/Code/detect_word/inference.py)
* **Combination** the output of CRAFT into each sentences (cause the sentence detect mode of CRAFT is not accurate despite of adjust param so I write a combine scripts in [combineLine](/Code/combineLine.py)) 
* OCR with **Vietocr** (create model in main file and pass it into [end2end](/Code/combineLine.py) function to read sentences after combine line)
* Embedding graph with Graph instance (create feature, embedding sentence by [vietnamese-sbert](https://huggingface.co/keepitreal/vietnamese-sbert))
* Inference with GCN and save image output.

___
### Ouput of each component
- **<font color='Blue' >1: U2net Segmentation</font>**
  
![](/Code/U2Net/output/mcocr_public_145013chgcz.jpg)

<br><br>

- **<font color='orange' >2: Text Detection CRAFT</font>**
  
![](/Code/detect_word/result/res_mcocr_public_145013aedmq.jpg)

<br><br>

- **<font color='green' >3: Combination into sentence</font>**
  
![](/Code/detect_word/result/done_combination.jpg)

<br><br>
- **<font color='purple' >4: GCN</font>**
  
![](/Code/gcn/output_result/mcocr_public_145013cxgot_result_.png)
