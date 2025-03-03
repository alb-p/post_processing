import pandas as pd
import ast
import json

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path)

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
    """
    Cleans the dataset by:
    - Removing frozenset representations and converting them into readable lists.
    """
    return df.applymap(force_remove_frozenset)

def clean_asso_rules(config_path):
    """
    Cleans association rules from datasets and models and saves in multiple formats.
    """
    config = load_config(config_path)
    tables_dir = config["tables_dir"]
    analysis_dir = config["analysis_dir"]
    datasets = config["datasets"]
    models = config["models"]
    
    for dataset in datasets:
        for model in models:
            table_path = f"{dataset['name']}/{model['name']}_diff_asso_rules.csv"
            print(f"Processing: {table_path}")
            
            df = load_data(f"{tables_dir}/{table_path}")
            cleaned_df = clean_data(df)
            
            # Save cleaned file in multiple formats
            cleaned_csv_path = f"{analysis_dir}/{dataset['name']}/{model['name']}_cleaned_asso_rules.csv"
            cleaned_latex_path = f"{analysis_dir}/{dataset['name']}/{model['name']}_cleaned_asso_rules.tex"

            
            cleaned_df.to_csv(cleaned_csv_path, index=False)
            cleaned_df.to_latex(cleaned_latex_path, index=False, escape=True, na_rep="-",float_format="%.2f")
            
            print(f"Saved cleaned data to: {cleaned_csv_path}")
    
    df_1 = load_data(f"{analysis_dir}/adult/Decision Tree_cleaned_asso_rules.csv")
    print(df_1.head())

# Run the cleaning function
config_path = "config/config.json"
clean_asso_rules(config_path)