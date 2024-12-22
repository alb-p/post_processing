import pandas as pd
from data.adult_utils import prepare_adult_asso_rules
from data.compas_utils import prepare_compas_asso_rules
from mlxtend.frequent_patterns import association_rules, fpgrowth, apriori


def compute_association_rules(dataset, dataset_name, target_variable, support, confidence):
    df_transactions = compute_df_transactions(dataset, dataset_name)
    return compute_taget_rules(df_transactions, target_variable, support, confidence)
    

def compute_taget_rules(df_transactions, target_variable, min_support, min_confidence):
    # Apply FP-growth algorithm to find frequent itemsets
    frequent_itemsets = fpgrowth(df_transactions, min_support=min_support, use_colnames=True)
    # Extract association rules with min confidence
    res = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    res = res.sort_values(by='confidence', ascending=False)
    res_target = res[(res['consequents'] == {target_variable[0]}) | (res['consequents'] == {target_variable[1]})]
    res_target = res_target[["antecedents", "consequents", "support", "confidence"]]
    res_target = round(res_target, 3)
    return res_target


def compute_df_transactions(dataset, dataset_name):
    if "adult" in dataset_name:
        data_asso_rules = prepare_adult_asso_rules(dataset)
    if "compas" in dataset_name:
        data_asso_rules = prepare_compas_asso_rules(dataset)
    # Create a binary representation of transactions
    df_transactions = data_asso_rules.map(lambda x: True if x > 0 else False)
    return df_transactions

def compute_diff_association_rules(association_rules_technique, transf_asso_rules, model_name):
    support_col = f"{model_name}_support"
    confidence_col = f"{model_name}_confidence"
    association_rules_technique[support_col] = None
    association_rules_technique[confidence_col] = None

    # Keep track of rules to append
    rules_to_append = []

    # Iterate over rows in transf_asso_rules
    for _, rule in transf_asso_rules.iterrows():
        matched = False
        for index, rule_orig in association_rules_technique.iterrows():
            if rule['antecedents'] == rule_orig['antecedents'] and rule['consequents'] == rule_orig['consequents']:
                # Update the matching row in association_rules_technique
                association_rules_technique.at[index, support_col] = rule['support']
                association_rules_technique.at[index, confidence_col] = rule['confidence']
                matched = True
                break
        
        if not matched:
            # Add the unmatched rule to the list for appending
            new_rule = rule.copy()
            new_rule[support_col] = rule['support']
            new_rule[confidence_col] = rule['confidence']
            rules_to_append.append(new_rule)

    # Append unmatched rules to association_rules_technique
    if rules_to_append:
        association_rules_technique = pd.concat(
            [association_rules_technique, pd.DataFrame(rules_to_append)],
            ignore_index=True
        )
    return association_rules_technique
    
    
def export_association_rules(rules, tables_dir, dataset_name, filename):
    """
    Exports association rules to a CSV file.

    Args:
        rules (pd.DataFrame): DataFrame containing association rules.
        file_path (str): Path to save the CSV file.
    """
    filepath = f"{tables_dir}/{dataset_name}/{filename}"
    # os.makedirs(filepath, exist_ok=True)
    export_rules = []
    for _, rule in rules.iterrows():
        export_rules.append({
            "Dataset": dataset_name,
            "Rule": f"{set(rule['antecedents'])} -> {set(rule['consequents'])}".replace("frozenset({", "").replace("})", ""),
            "support": round(rule["support"],3),
            "confidence": round(rule["confidence"],3)
        })
    export_df = pd.DataFrame(export_rules, columns=["Dataset", "Rule", "support", "confidence"])
    export_df.to_csv(filepath, index=False)
    
def clean_association_rules(rules):
    clean_rules = []
    for _, rule in rules.iterrows():
        clean_rules.append({
            "antecedents": rule["antecedents"],
            "consequents": rule["consequents"],
            "support": round(rule["support"],3),
            "confidence": round(rule["confidence"],3)
        })
    # questo passaggio rimette indici da 0
    return pd.DataFrame(clean_rules, columns=["antecedents", "consequents", "support", "confidence"])
