# MFTEP
MFTEP is a multimodal fusion method to predict TCR-epitope interactions by fusing the sequence features, molecular graph features, and 3D structure features of TCRs and epitopes.

# Dependencies
MFTEP is writen in Python based on Pytorch. The required software dependencies are listed below:

```
torch
Numpy
pandas
scikit-learn
torch_geometric
torch_cluster
torch_scatter
timm
pytoda
Bio
```

# Data
All the data used in the paper were collected from public databases. We also uploaded the processed data in the data package.

# Usage of MFTEP
Obtain 3D structure:
```
python data/get_3D_structure.py
```
Training MFTEP:
```
python train_model.py --device "cuda" --epoch 50 --batch_size 64 --train_dataset "./data/strict_split/fold0/train.csv" --test_dataset "./data/strict_split/fold0/test.csv"
```
Predict TCR-epitope pairs:
```
python train_model.py --device "cuda" --epoch 50 --batch_size 64 --test_dataset "./data/strict_split/fold0/test.csv" --only_test True --save_model "./checkpoint.pt"
```
