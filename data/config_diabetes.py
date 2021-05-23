from data.dataset import *
import numpy as np
import pickle

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#hyperparameters
batch_size = 8
validate_every_no_of_batches = 80
epochs = 100000
input_size = 10
output_size = 1
hidden_shapes = [16]
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/diabetes_log.txt"

#diabetes dataset
diabetes_dataset = load_diabetes()

X = diabetes_dataset['data']

data = dataset(X, diabetes_dataset['target'], batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.6, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
