# License 
- This is a dataset of VNDAG for research purposes only. 
You need to sign an user agreement form [1] and send to vndag@vietnlp.com to register before using any data from VNDAG.
For other purposes, please contact us. We have different agreements for partners.

[1] https://drive.google.com/file/d/139OkQZno0TVc3Ifzk3jr0ebv2X_pR8SG/view?usp=sharing

# Data format:
- label_dict.json: contains the mapping from field-id to field-name. 
- `./mcocr_train_data/train_images` contains raw receipts of training data.
- `./mcocr_val_data/val_images` contains raw receipts of validation (i.e., public testing) data.
- File `mcocr_train_data/mcocr_train_df.csv` contains annotations as described in [2].
- File `mcocr_val_data/mcocr_val_sample_df.csv` contains list of testing receipts and predicted info (quality score and ocred text). Before submission, please rename this file to `results.csv` and `zip` it.

[2] https://rivf2021-mc-ocr.vietnlp.com/dataset
