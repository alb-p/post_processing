import pandas as pd
import ast
import json
import re
import os

### CONFIGURATION AND DATA LOADING ###

def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

### CLEANING FUNCTIONS ###

def force_remove_frozenset(value):
    """Remove 'frozenset()' wrapper and return a clean, comma-separated list."""
    if isinstance(value, str) and "frozenset" in value:
        try:
            evaluated = ast.literal_eval(value)  # Convert string to Python object
            if isinstance(evaluated, frozenset):
                return ", ".join(evaluated)  # Convert set to comma-separated string
        except (ValueError, SyntaxError):
            return value.replace("frozenset(", "").replace(")", "").replace("{", "").replace("}", "").replace("'", "")
    return value  # Return unchanged if not a frozenset

def clean_data(df):
    """Cleans dataset by converting frozenset representations."""
    return df.applymap(force_remove_frozenset)

def clean_asso_rules(config_path):
    """
    Cleans association rules from datasets, saves them in multiple formats.
    """
    config = load_config(config_path)
    tables_dir = config["tables_dir"]
    analysis_dir = config["analysis_dir"]
    datasets = config["datasets"]
    models = config["models"]
    
    for dataset in datasets:
        for model in models:
            table_path = f"{dataset['name']}/{model['name']}_diff_asso_rules.csv"
            full_table_path = f"{tables_dir}/{table_path}"

            if not os.path.exists(full_table_path):
                print(f"Skipping {full_table_path} (file not found)")
                continue

            print(f"Processing: {table_path}")

            df = load_data(full_table_path)
            cleaned_df = clean_data(df)

            # Define output file paths
            cleaned_csv_path = f"{analysis_dir}/{dataset['name']}/{model['name']}_cleaned_asso_rules.csv"
            cleaned_latex_path = f"{analysis_dir}/{dataset['name']}/{model['name']}_cleaned_asso_rules.tex"

            # Save cleaned CSV and LaTeX
            cleaned_df.to_csv(cleaned_csv_path, index=False)
            cleaned_df.to_latex(cleaned_latex_path, index=False, escape=True, na_rep="-", float_format="%.2f")

            print(f"Saved cleaned data to: {cleaned_csv_path}")

### LATEX PROCESSING FUNCTIONS ###

def extract_relevant_rules(file_path):
    """Extracts content between \midrule and \bottomrule from LaTeX files."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    match = re.search(r"\\midrule(.*?)\\bottomrule", content, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return ""

def process_latex_tables(config_path):
    """
    Processes LaTeX tables for each dataset and model,
    extracting relevant content and saving a new file.
    """
    config = load_config(config_path)
    analysis_dir = config["analysis_dir"]
    datasets = config["datasets"]
    models = config["models"]

    begin_file = f"{analysis_dir}/begin_table.txt"

    for dataset in datasets:
        for model in models:
            latex_file = f"{analysis_dir}/{dataset['name']}/{model['name']}_cleaned_asso_rules.tex"
            output_file = f"{analysis_dir}/{dataset['name']}/{model['name']}_final_asso_rules.tex"

            if not os.path.exists(latex_file):
                print(f"Skipping {latex_file} (file not found)")
                continue

            # Read table header
            with open(begin_file, "r", encoding="utf-8") as f:
                begin_content = f.read()

            # Extract relevant rules
            rules_content = extract_relevant_rules(latex_file)

            dataset_name = dataset["name"].capitalize() + " Dataset"
            model_name = model["name"] + " Model"

            # Properly formatted end content
            end_content = f"""        \\hline
        \\end{{tabular}}
        }}
        \\\\[10pt]
        \\caption{{Association Rule for {dataset_name} - {model_name}}}
        \\label{{table:asso_rules_{dataset['name']}_{model['name']}}}
        \\end{{table}}
        """

            # Merge all components
            final_content = begin_content + rules_content + end_content

            # Save processed LaTeX file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_content)

            print(f"Processed LaTeX table saved as {output_file}")
### MAIN EXECUTION ###

if __name__ == "__main__":
    # Load config
    config_path = "config/config.json"
    
    # Step 1: Clean Association Rules
    clean_asso_rules(config_path)

    # Step 2: Process LaTeX tables separately for each dataset and model
    process_latex_tables(config_path)
