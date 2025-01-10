from matplotlib import pyplot as plt
import pandas as pd
from utils.association_rules_utils import compute_df_transactions
from utils.data_utils import get_safe_value
from utils.gen_utils import sns_line_plotting
import seaborn as sns


# TODO: male/female -> privileged/unprivileged

def compute_accuracy(df_orig_test, df_orig_test_pred, df_transf_test_pred, target_variable, sensible_attribute, filepath, technique_name, model_name):
    '''
    Return the accuracy of the model after the transformation
    '''
    income_gender_counts_test = df_orig_test.groupby([sensible_attribute, target_variable]).size().unstack(fill_value=0)
    income_gender_counts_test_pred = df_orig_test_pred.groupby([sensible_attribute, target_variable]).size().unstack(fill_value=0)
    income_gender_counts_transf_test_pred = df_transf_test_pred.groupby([sensible_attribute, target_variable]).size().unstack(fill_value=0)
    total_test_len = df_orig_test.shape[0]


    male_actual = get_safe_value(income_gender_counts_test, 1, 1)
    male_pred_before = get_safe_value(income_gender_counts_test_pred, 1, 1)
    male_pred_after = get_safe_value(income_gender_counts_transf_test_pred, 1, 1)

    female_actual = get_safe_value(income_gender_counts_test, 0, 0)
    female_pred_before = get_safe_value(income_gender_counts_test_pred, 0, 0)
    female_pred_after = get_safe_value(income_gender_counts_transf_test_pred, 0, 0)

    # male_accuracy_before = round((1 - abs(male_pred_before - male_actual) / total_test_len), 3)
    # female_accuracy_before = round((1 - abs(female_pred_before - female_actual) / total_test_len), 3)
    # total_accuracy_before = round((1 - abs(abs(male_pred_before - male_actual) + abs(female_pred_before - female_actual)) / total_test_len), 3)

  
    male_accuracy_after = round(1-(abs(male_pred_after - male_actual) / total_test_len), 3)
    female_accuracy_after = round(1-(abs(female_pred_after - female_actual) / total_test_len), 3)
    print(f"overall accuracy: male_actual {male_actual}, male_pred_after {male_pred_after}, female_actul {female_actual}, female_pred_after {female_pred_after}, total_test_len {total_test_len}")
    total_accuracy_after = round(1-(abs(abs(male_pred_after - male_actual) + abs(female_pred_after - female_actual)) / total_test_len), 3)

    # df_accuracy = pd.DataFrame({
    #     'Metrics': ['Male Accuracy', 'Female Accuracy', 'Overall Accuracy'],
    #     'Before': [male_accuracy_before, female_accuracy_before, total_accuracy_before],
    #     'After':[male_accuracy_after, female_accuracy_after, total_accuracy_after] 
    # })   
    # return round(male_accuracy_after-male_accuracy_before,3), round(female_accuracy_after-female_accuracy_before,3), round(total_accuracy_after-total_accuracy_before,3)
    return male_accuracy_after, female_accuracy_after, total_accuracy_after

def plot_accuracy_list(accuracy_list, plots_dir, dataset_name, technique_name):
    columns = ["Dataset", "Technique", "Model", "priv_accuracy", "unpriv_accuracy", "total_accuracy"]
    accuracy_df = pd.DataFrame(accuracy_list, columns=columns)

    # Drop privileged and unprivileged accuracy columns as they are not needed for this plot
    accuracy_df = accuracy_df[["Model", "total_accuracy"]]
    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=accuracy_df,
        x="Model",
        y="total_accuracy",
        palette="rocket"
    )

    # Add plot labels and title
    plt.title(f"Accuracy Delta Across Models - {technique_name}")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    filepath = f"{plots_dir}/{technique_name}_{dataset_name}_accuracy2.png"


    # Save and show the plot
    plt.savefig(filepath)
    plt.close()
    

def compute_consistency(dataset_orig_test, dataset_transf_test_pred, orig_asso_rules, dataset_name):
    changed_tuples = dataset_orig_test.labels != dataset_transf_test_pred.labels
    changed_df = dataset_orig_test.convert_to_dataframe()[0][changed_tuples]
    changed_transactions= compute_df_transactions(changed_df, dataset_name)
    # a row respects rules if meeting all the antecedents, it meets also the consequent.
    # Only the label-changed rows are analyzed 
    rows_not_respecting_rules = 0
    rows_violating_rules = changed_transactions.apply(lambda row: violates_any_rule(row, orig_asso_rules), axis=1)
    rows_not_respecting_rules = rows_violating_rules.sum()
    # Calculate the delta
    total_rows = dataset_transf_test_pred.features.shape[0]
    delta = (rows_not_respecting_rules / total_rows)
    return round(1-delta,3)
    

def plot_consistency(df, dataset_name, filepath):
    #  Create a DataFrame for plotting directly from the input DataFrame
    df_consistency = df[['technique_name', 'model_name', 'consistency']].rename(
        columns={'technique_name': 'Technique', 'model_name': 'Model', 'consistency': 'Value'}
    )
    
    # Call sns_line_plotting with the prepared DataFrame
    sns_line_c_plotting(
        df=df_consistency,
        x='Model',
        y='Value',
        hue='Technique',
        title=f'{dataset_name}: Consistency',
        axhline=1,
        filepath=filepath
    )

def sns_line_c_plotting(df, x='Stage', y='Value', hue='Metrics', title="Metric Performance: Before vs After", grid=True, axhline=-1, filepath='output/plots/sns_line_plotting_noname.png'):
    # Plot the line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")

    # Set the plot title and labels
    plt.title(title)
    if axhline != -1:
        plt.axhline(axhline, color='red', linestyle='--')
        if axhline == 0:
            plt.ylim(-1, 1)
        elif axhline == 1:
            plt.ylim(0.5, 1.1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=hue)
    plt.grid(grid)
    plt.savefig(filepath)
    plt.close()



def plot_consistency_list(consistency_list, plots_dir, dataset_name, technique_name):
    #FIXME: move the plotting function to gen_utils
    columns = ["Dataset", "Technique", "Model", "Final Consistency"]
    consistency_df = pd.DataFrame(consistency_list, columns=columns)
    
    # Add an initial consistency value of 1 for all models
    consistency_df["Initial Consistency"] = 1.0

    # Melt the DataFrame to long format for plotting
    melted_df = pd.melt(
        consistency_df,
        id_vars=["Model"],
        value_vars=["Initial Consistency", "Final Consistency"],
        var_name="Stage",
        value_name="Consistency Value"
    )
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=melted_df,
        x="Stage",
        y="Consistency Value",
        hue="Model",
        marker="o",
        errorbar=None
    )

    # Add a horizontal line for ideal consistency (1.0)
    plt.axhline(y=1.0, color="red", linestyle="--", label="Ideal Consistency")

    # Set plot labels, title, and legend
    plt.title(f"Consistency Metric Across Models - {technique_name}")
    plt.xlabel("Consistency Stage")
    plt.ylabel("Consistency Value")
    plt.ylim(0.93, 1.01)
    plt.legend(title="Model", loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    # Set up file path
    filepath = f"{plots_dir}/{technique_name}_{dataset_name}_consistency4.png"
    plt.savefig(filepath)
    plt.close()

# def plot_consistency_list(consistency_list, plots_dir, dataset_name, technique_name):
#     columns = ["dataset_name", "technique_name", "model_name", "consistency"]
#     consistency_df = pd.DataFrame(consistency_list, columns=columns)
    
#     # Filter the DataFrame for the current dataset and technique
#     consistency_df = consistency_df[
#         (consistency_df["dataset_name"] == dataset_name) & 
#         (consistency_df["technique_name"] == technique_name)
#     ]
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         consistency_df["model_name"], 
#         consistency_df["consistency"], 
#         marker="o", linestyle="-", label=f"Technique: {technique_name}"
#     )
#     plt.title(f"Consistency Scores for {technique_name} on {dataset_name}")
#     plt.xlabel("Model")
#     plt.ylabel("Consistency")
#     plt.axhline(1, color='red', linestyle='--')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     # Set up file path
#     filepath = f"{plots_dir}/{technique_name}_{dataset_name}_consistency3.png"
#     plt.savefig(filepath)



def violates_any_rule(row, rules):
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        if not satisfies_rule(row, antecedents, consequents):
            return True  # Row violates this rule
    return False  # Row satisfies all rules

def satisfies_rule(row, antecedents, consequents):
    antecedent_check = all(row[item] for item in antecedents)
    consequent_check = all(row[item] for item in consequents)
    return not antecedent_check or ( antecedent_check and consequent_check)
