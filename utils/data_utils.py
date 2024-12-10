import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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

def train_test_distribution_plot(dataset_orig, dataset_orig_train, dataset_orig_test, sensible_attribute, target_variable, plots_dir, dataset_name): 
    income_gender_counts_orig, orig_size = extract_target_sensible_attributes(dataset_orig, sensible_attribute, target_variable)
    income_gender_counts_train, train_size = extract_target_sensible_attributes(dataset_orig_train, sensible_attribute, target_variable)
    income_gender_counts_test, test_size = extract_target_sensible_attributes(dataset_orig_test, sensible_attribute, target_variable)

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
        filepath=f'{plots_dir}/{dataset_name}/{dataset_name}_train_test_distribution.png'
    )

def extract_target_sensible_attributes(dataset, sensible_attribute, target_variable):
    if  isinstance(dataset, BinaryLabelDataset):
        dataset = dataset.convert_to_dataframe()[0]
    return dataset.groupby([sensible_attribute, target_variable]).size().unstack(), len(dataset)

def stages_distribution_plot(dataset_orig, dataset_pred, dataset_transf, sensible_attribute, target_variable, plots_dir, dataset_name, technique_name, model_name): 
    income_gender_counts_orig, _ = extract_target_sensible_attributes(dataset_orig, sensible_attribute, target_variable)
    income_gender_counts_pred, _ = extract_target_sensible_attributes(dataset_pred, sensible_attribute, target_variable)
    income_gender_counts_transf, _ = extract_target_sensible_attributes(dataset_transf, sensible_attribute, target_variable)
    df_comparison_target = pd.DataFrame({
        'Male(<=50K)': [income_gender_counts_orig.loc[1, 0], income_gender_counts_pred.loc[1, 0], income_gender_counts_transf.loc[1,0]],
        'Male(>50K)': [income_gender_counts_orig.loc[1, 1], income_gender_counts_pred.loc[1, 1], income_gender_counts_transf.loc[1,1]],
        'Female(<=50K)': [income_gender_counts_orig.loc[0, 0], income_gender_counts_pred.loc[0, 0], income_gender_counts_transf.loc[0,0]],
        'Female(>50K)': [income_gender_counts_orig.loc[0, 1], income_gender_counts_pred.loc[0, 1], income_gender_counts_transf.loc[0,1]],
    },index = ['Original', 'Predicted', 'Transformed'])

    
    # var_name = sensible_attribute+'_'+target_variable
    # value_name='Count'
    # df_comparison_target_reset = df_comparison_target.reset_index().melt(id_vars='index', var_name=var_name, value_name=value_name)
    # df_comparison_target_reset.rename(columns={'index': 'Dataset_Type'}, inplace=True)
    

    # FIXME: delete this hard coding from old code
    df_comparison_target_reset = df_comparison_target[['Male(<=50K)', 'Male(>50K)', 'Female(<=50K)', 'Female(>50K)']].reset_index().melt(id_vars='index', var_name='Gender_Income', value_name='Count')
    df_comparison_target_reset.rename(columns={'index': 'Dataset_Type'}, inplace=True)

    
    # sns_bar_plotting(
    #     df=df_comparison_target_reset, 
    #     x='Dataset_Type', 
    #     y=value_name, 
    #     hue= var_name,
    #     title=f'{technique_name} - {model_name}: Target distributions',
    #     filepath=f'{plots_dir}/{dataset_name}/{technique_name}_{model_name}_target_distribution.png'
    # )
    filepath=f'{plots_dir}/{dataset_name}/{technique_name}_{model_name}_target_distribution.png'

    # Plotting with Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_comparison_target_reset, x='Dataset_Type', y='Count', hue='Gender_Income', palette='rocket')
    plt.title("Income by Gender Across Original, Predicted, and Transformed Datasets")
    plt.xlabel("Dataset Type")
    plt.ylabel("Count of Individuals")
    plt.xticks(rotation=0)
    plt.legend(title="Income and Gender")

    # Save and show the plot
    plt.savefig(filepath)


# def stages_distribution_plot(dataset_orig, dataset_pred, dataset_transf, sensible_attribute, target_variable, plots_dir, dataset_name, technique_name, model_name):
#     # Extract and verify target distributions
#     income_gender_counts_orig, _ = extract_target_sensible_attributes(dataset_orig, sensible_attribute, target_variable)
#     income_gender_counts_pred, _ = extract_target_sensible_attributes(dataset_pred, sensible_attribute, target_variable)
#     income_gender_counts_transf, _ = extract_target_sensible_attributes(dataset_transf, sensible_attribute, target_variable)
    
#     # Construct DataFrame for target distribution only
#     df_comparison_target = pd.DataFrame({
#         'Male(<=50K)': [income_gender_counts_orig.loc[1, 0], income_gender_counts_pred.loc[1, 0], income_gender_counts_transf.loc[1, 0]],
#         'Male(>50K)': [income_gender_counts_orig.loc[1, 1], income_gender_counts_pred.loc[1, 1], income_gender_counts_transf.loc[1, 1]],
#         'Female(<=50K)': [income_gender_counts_orig.loc[0, 0], income_gender_counts_pred.loc[0, 0], income_gender_counts_transf.loc[0, 0]],
#         'Female(>50K)': [income_gender_counts_orig.loc[0, 1], income_gender_counts_pred.loc[0, 1], income_gender_counts_transf.loc[0, 1]],
#     }, index=['Original', 'Predicted', 'Transformed'])
    
#     # Debugging: Verify contents
#     print(df_comparison_target)

#     # Reshape for plotting
#     df_comparison_reset = df_comparison_target.reset_index().melt(id_vars='index', var_name=f'{sensible_attribute}_{target_variable}', value_name='Count')
#     df_comparison_reset.rename(columns={'index': 'Dataset_Type'}, inplace=True)

#     # Debugging: Verify reshaped data
#     print(df_comparison_reset)

#     # Plot the target distribution
#     sns_bar_plotting(
#         df=df_comparison_reset,
#         x='Dataset_Type',
#         y='Count',
#         hue=f'{sensible_attribute}_{target_variable}',
#         title=f'{technique_name} - {model_name}: Target distributions',
#         filepath=f'{plots_dir}/{dataset_name}/{technique_name}_{model_name}_target_distribution.png'
#     )

