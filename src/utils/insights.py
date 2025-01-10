import pandas as pd
import json

config_path = "config/config.json"

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path)



def compute_analysis(table_path, metrics, title):
    config = load_config(config_path)
    tables_dir = config["tables_dir"]
    analysis_dir = config["analysis_dir"]

    df = load_data(f"{tables_dir}/{table_path}")
    datasets = df["dataset_name"].unique()
    techniques = df["technique_name"].unique()
    models = df["model_name"].unique()

    # Group by dataset and technique, then calculate the mean of metrics
    technique_avg_by_dataset = df.groupby(["dataset_name", "technique_name"])[metrics].mean()
    # Combine results into a single DataFrame with the dataset name as a column
    combined_data = technique_avg_by_dataset.reset_index()
    # Round the metrics for better readability
    combined_data[metrics] = combined_data[metrics].round(3)
    # Save the combined results to a single CSV file
    output_file = f"{analysis_dir}/{title}.csv"
    combined_data.to_csv(output_file, index=False)

    # # Create a boolean DataFrame indicating where metric values are > 0
    # positive_counts = df[metrics] > 0

    # # Add the dataset_name and technique_name columns back to the boolean DataFrame
    # positive_counts = pd.concat([df[["dataset_name", "technique_name"]], positive_counts], axis=1)

    # # Group by dataset and technique, then count the rows where metrics > 0
    # count_positive = positive_counts.groupby(["dataset_name", "technique_name"])[metrics].sum()

    # # Convert the results into a combined DataFrame with dataset names as a column
    # combined_positive_counts = count_positive.reset_index()

    # # Save to a CSV file
    # output_file = f"{tables_dir}/positive_metrics_counts.csv"
    # combined_positive_counts.to_csv(output_file, index=False)

fairness_metrics = ["GroupFairness", "PredictiveParity", "PredictiveEquality", "EqualOpportunity", "EqualizedOdds"]
compute_analysis("fairness_results.csv", fairness_metrics, "technique_fairness_analysis")
quality_metrics = ["accuracy", "consistency"]
compute_analysis("quality_results.csv", quality_metrics, "technique_quality_analysis")
performance_metrics = ["Accuracy", "Recall", "Precision", "F1"]
compute_analysis("performance_results.csv", performance_metrics, "technique_performance_analysis")