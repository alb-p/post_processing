import pandas as pd
import json

import matplotlib.pyplot as plt
import numpy as np

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


def compute_count(table_path, metrics, title):
    config = load_config(config_path)
    tables_dir = config["tables_dir"]
    analysis_dir = config["analysis_dir"]

    df = load_data(f"{tables_dir}/{table_path}")    
    # Group by dataset and technique, count True values for each metric
    # technique_counts = df.groupby(["dataset_name", "model_name"])[metrics].sum()
    technique_counts = df.groupby(["dataset_name", "technique_name"])[metrics].sum()
    
    # Reset the index to convert it to a standard table format
    combined_data = technique_counts.reset_index()
    
    # Save the table to a CSV file
    output_file = f"{analysis_dir}/{title}.csv"
    combined_data.to_csv(output_file, index=False)


def model_printing_fairness(df_to_plot, metrics, axhline=-1, title="Fairness Metrics Comparison", filepath="output/plots"):
    config = load_config(config_path)
    plot = config["tables_dir"]
    filepath = f"{plot}/"
    analysis_dir = config["analysis_dir"]
    datasets = df_to_plot["dataset_name"].unique()
    models = df_to_plot["model_name"].unique()
    
    for dataset in datasets:
        dataset_data = df_to_plot[df_to_plot["dataset_name"] == dataset]
        for model in models:
            model_data = dataset_data[dataset_data["model_name"] == model]
            techniques = model_data["technique_name"].unique()

            x = np.arange(len(metrics) // 2)  # Base x positions for metrics
            width = 0.8  # Base bar width

            fig, ax = plt.subplots(figsize=(12, 6))

            # Generate paired colors
            colormap = plt.cm.Set1
            num_colors = len(techniques) * 2  # Each technique has 'Before' and 'After'
            colors = [colormap(i / num_colors) for i in range(num_colors)]

            for i, technique in enumerate(techniques):
                technique_data = model_data[model_data["technique_name"] == technique]
                before_values = technique_data[metrics[::2]].values.flatten()
                after_values = technique_data[metrics[1::2]].values.flatten()

                # Assign colors from the paired colormap
                before_color = colors[2 * i]
                after_color = colors[2 * i + 1]

                ax.bar(
                    x + i * width / len(techniques),
                    before_values,
                    width=0.8 * width / len(techniques),
                    label=f"{technique} (Before)",
                    alpha=0.5,
                    color=before_color
                )
                ax.bar(
                    x + i * width / len(techniques) + 0.2 * width / len(techniques),
                    after_values,
                    width=0.8 * width / len(techniques),
                    label=f"{technique} (After)",
                    alpha=0.8,
                    color=after_color
                )

            # Add gridlines, labels, and title
            ax.set_ylabel("Metric Values")
            ax.set_xlabel("Fairness Metrics")
            ax.set_title(f"{model} - {title} ({dataset})")
            ax.set_xticks(x + (len(techniques) - 1) * width / (2 * len(techniques)))
            ax.set_xticklabels([metric.replace("_after", "") for metric in metrics[1::2]])
            ax.grid(axis="y")

            # Add horizontal line if specified
            if axhline == -2:
                ax.axhline(1.0, color="red", linestyle="dashed", label="Max Improvement")
                ax.axhline(-1.0, color="red", linestyle="dashed", label="Max Decline")
            elif axhline != -1:
                ax.axhline(axhline, color="red", linestyle="--", label="Ideal Value")
                ax.set_ylim(min(-1, axhline - 0.1), max(1.1, axhline + 0.1))

            ax.legend(title="Techniques")
            plt.xticks(rotation=30)
            plt.tight_layout()

            # Save the plot
            plot_filename = f"{filepath}/{dataset}_{model}_{title.replace(' ', '_')}.png"
            plt.savefig(plot_filename)
            plt.close()


fairness_metrics = ["dataset_name", "model_name", "technique_name", "GroupFairness", "GroupFairness_after", "PredictiveParity", "PredictiveParity_after", "PredictiveEquality", "PredictiveEquality_after", "EqualOpportunity", "EqualOpportunity_after", "EqualizedOdds", "EqualizedOdds_after"]
fairness_df = pd.DataFrame(fairness_list, columns=fairness_df_columns)
model_printing_fairness(
    df_to_plot=fairness_df,
    metrics=fairness_metrics,
    axhline=0,
    title="Fairness Metrics Behavior")
fairness_df.to_csv(f"{tables_dir}/fairness_results.csv", index=False)



fairness_metrics = ["GroupFairness", "PredictiveParity", "PredictiveEquality", "EqualOpportunity", "EqualizedOdds"]
compute_analysis("fairness_results.csv", fairness_metrics, "technique_fairness_analysis")
quality_metrics = ["accuracy", "consistency"]
compute_analysis("quality_results.csv", quality_metrics, "technique_quality_analysis")
performance_metrics = ["Accuracy", "Recall", "Precision", "F1"]
compute_analysis("performance_results.csv", performance_metrics, "technique_performance_analysis")
fairness_star_metrics = ["GroupFairness_star","PredictiveParity_star","PredictiveEquality_star","EqualOpportunity_star","EqualizedOdds_star"]
compute_count("fairness_star_results.csv", fairness_star_metrics, "technique_fairness_counts")

# fairness_metrics = ["GroupFairness", "PredictiveParity", "PredictiveEquality", "EqualOpportunity", "EqualizedOdds"]
# compute_analysis("fairness_results.csv", fairness_metrics, "model_fairness_analysis")
# quality_metrics = ["accuracy", "consistency"]
# compute_analysis("quality_results.csv", quality_metrics, "model_quality_analysis")
# performance_metrics = ["Accuracy", "Recall", "Precision", "F1"]
# compute_analysis("performance_results.csv", performance_metrics, "model_performance_analysis")
# fairness_star_metrics = ["GroupFairness_star","PredictiveParity_star","PredictiveEquality_star","EqualOpportunity_star","EqualizedOdds_star"]
# compute_count("fairness_star_results.csv", fairness_star_metrics, "model_fairness_counts")