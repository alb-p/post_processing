import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler

from utils.gen_utils import sns_bar_plotting



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

# def train_test_distribution_plot(dataset_orig, dataset_orig_train, dataset_orig_test, sensible_attribute, target_variable, plots_dir, dataset_name): 
#     df_orig = dataset_orig.convert_to_dataframe()[0]
#     df_train = dataset_orig_train.convert_to_dataframe()[0]
#     df_test = dataset_orig_test.convert_to_dataframe()[0]
#      # Group by gender and income, and normalize counts
#     income_gender_counts_orig = df_orig.groupby([sensible_attribute, target_variable]).size() / len(df_orig)
#     income_gender_counts_train = df_train.groupby([sensible_attribute, target_variable]).size() / len(df_train)
#     income_gender_counts_test = df_test.groupby([sensible_attribute, target_variable]).size() / len(df_test)
    
#     # Create a DataFrame for plotting
#     df_comparison = pd.DataFrame({
#         'Origin': income_gender_counts_orig,
#         'Training': income_gender_counts_train,
#         'Testing': income_gender_counts_test
#     }).stack().reset_index()
    
#     # Rename columns for clarity
#     df_comparison.columns = [sensible_attribute, target_variable, 'Dataset', 'Proportion']
#     sns_bar_plotting(
#         df=df_comparison, 
#         x='Dataset', 
#         y='Proportion', 
#         hue=f'{sensible_attribute}({target_variable})', 
#         filepath=f'{plots_dir}/{dataset_name}/train_test_distribution.png'
#     )
def train_test_distribution_plot(dataset_orig, dataset_orig_train, dataset_orig_test, sensible_attribute, target_variable, plots_dir, dataset_name): 
    df_orig = dataset_orig.convert_to_dataframe()[0]
    df_train = dataset_orig_train.convert_to_dataframe()[0]
    df_test = dataset_orig_test.convert_to_dataframe()[0]
    
    # Group by sensible_attribute and target_variable, normalize counts
    income_gender_counts_orig = df_orig.groupby([sensible_attribute, target_variable]).size().unstack()
    income_gender_counts_train = df_train.groupby([sensible_attribute, target_variable]).size().unstack()
    income_gender_counts_test = df_test.groupby([sensible_attribute, target_variable]).size().unstack()


    orig_size = len(df_orig)
    train_size = len(df_train)
    test_size = len(df_test)
    df_comparison_train_test = pd.DataFrame({
        'Male(<=50K)': [income_gender_counts_orig.loc[1, 0]/orig_size, income_gender_counts_train.loc[1, 0]/train_size, income_gender_counts_test.loc[1, 0]/test_size],
        'Male(>50K)': [income_gender_counts_orig.loc[1, 1]/orig_size, income_gender_counts_train.loc[1, 1]/train_size, income_gender_counts_test.loc[1, 1]/test_size],
        'Female(<=50K)': [income_gender_counts_orig.loc[0, 0]/orig_size, income_gender_counts_train.loc[0, 0]/train_size, income_gender_counts_test.loc[0, 0]/test_size],
        'Female(>50K)': [income_gender_counts_orig.loc[0, 1]/orig_size, income_gender_counts_train.loc[0, 1]/train_size, income_gender_counts_test.loc[0, 1]/test_size],
    }, index=['Origin', 'Training', 'Testing'])


    var_name = sensible_attribute+'_'+target_variable
    value_name='Count'
    df_comparison_train_test_reset = df_comparison_train_test.reset_index().melt(id_vars='index', var_name=var_name, value_name=value_name)
    df_comparison_train_test_reset.rename(columns={'index': 'Dataset_Type'}, inplace=True)
    sns_bar_plotting(
        df=df_comparison_train_test_reset, 
        x='Dataset_Type', 
        y=value_name, 
        hue= var_name,
        title='Train-test distributions',
        filepath=f'{plots_dir}/{dataset_name}/train_test_distribution.png'
    )



# def train_test_distribution_plot(dataset_orig, dataset_orig_train, dataset_orig_test, sensible_attribute, target_variable, plots_dir, dataset_name): 
#     # Convert datasets to DataFrames
#     df_orig = dataset_orig.convert_to_dataframe()[0]
#     df_train = dataset_orig_train.convert_to_dataframe()[0]
#     df_test = dataset_orig_test.convert_to_dataframe()[0]
    
#     # Group by sensible_attribute and target_variable, normalize counts
#     income_gender_counts_orig = df_orig.groupby([sensible_attribute, target_variable]).size() / len(df_orig)
#     income_gender_counts_train = df_train.groupby([sensible_attribute, target_variable]).size() / len(df_train)
#     income_gender_counts_test = df_test.groupby([sensible_attribute, target_variable]).size() / len(df_test)
    
#     # Create a DataFrame for plotting
#     df_comparison = pd.DataFrame({
#         'Origin': income_gender_counts_orig,
#         'Training': income_gender_counts_train,
#         'Testing': income_gender_counts_test
#     }).stack().reset_index()
    
#     # Rename columns for clarity
#     df_comparison.columns = [sensible_attribute, target_variable, 'Dataset', 'Proportion']
    
#     # Create a combined column for hue
#     df_comparison['Attribute(Target)'] = df_comparison[sensible_attribute].astype(str) + '(' + df_comparison[target_variable].astype(str) + ')'
    
#     # Call the plotting function
#     sns_bar_plotting(
#         df=df_comparison, 
#         x='Dataset', 
#         y='Proportion', 
#         hue='Attribute(Target)',  # Use the combined column name
#         title='Train-test distributions',
#         filepath=f'{plots_dir}/{dataset_name}/train_test_distribution.png'
#     )
