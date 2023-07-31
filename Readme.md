# :vietnam: Key Infomation Extraction on Vietnamese invoice

Structure of project:
```
GCN_Vietnamese
|_Code
|_aligment (preprocess raw image folder)
| |_angle_mlp.py (main)
|_detect_word 
| |_inference.py
| |_perspective.py
|_extract_label (explore raw label file into small file )
| |_extract_label.py
|_gcn (create instance of Graph and insists of inference file)
| |_use
| |_graph.py (class of Graph)
| |_prepare (create instance and preprocessing data)
| |_train.py
| |_ model.py
| |_gen_dataVN.py
| |_test_single.py
| |_inference.py
|_material (prepare data file for train GCN)
| |_test_data
| |_train_data
|_ocr (material for read text)
| |_ocr.py
|_U2Net (Segmentation background)
| |_u2net_test.py
|_Vietnam_invoice_data
|_mcocr2021_raw (raw data with background - not use) - 1934 images all
| |_mcocr_train_data
| |_mcocr_val_data
| |_test
|_preprocessed_data (preprocess data (rotated, delete back) use this) - 1090 images
|
| |_images (.jpg)
| |_label_mcocr2021 (.csv)
|_combineLine.py
|_fullGCN.py
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
![](/done_combination.jpg)

<br><br>
- **<font color='purple' >4: GCN</font>**
![](/Code/gcn/output_result/mcocr_public_145013cxgot_result_.png)
