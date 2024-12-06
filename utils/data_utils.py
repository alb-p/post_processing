import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler



def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """
    Saves data to a CSV file.
    """
    data.to_csv(file_path, index=False)

def split_data(data, train_ratio=0.7):
    """
    Splits data into training and testing sets.
    """
    train_size = int(len(data) * train_ratio)
    return data[:train_size], data[train_size:]


def preprocess_dataset(dataset):
    dataset.replace('?', np.nan, inplace=True)
    dataset.dropna(inplace=True)
    return dataset

def generate_bld(dataset, target_variable, sensible_attribute):
    """
    Generates a binary dataset from the original dataset.
    """
    bin = BinaryLabelDataset(
    df=dataset.copy(),
    label_names=[target_variable],
    favorable_label=1,
    unfavorable_label=0,
    protected_attribute_names=[sensible_attribute])
    return bin

def X_y_train(dataset):
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset.features)
    y_train = dataset.labels.ravel()
    return scale_orig, X_train, y_train