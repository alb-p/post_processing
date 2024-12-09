from utils.association_rules_utils import compute_df_transactions


def compute_accuracy(df_orig_test, df_transf_test_pred, unprivileged_groups, privileged_groups, target_variable, sensible_attribute):
    income_gender_counts_test = df_orig_test.groupby([sensible_attribute, target_variable]).size().unstack()
    income_gender_counts_transf_test_pred = df_transf_test_pred.groupby([sensible_attribute, target_variable]).size().unstack()
    total_test_len = df_orig_test.shape[0]
    male_accuracy = round((1 - abs(income_gender_counts_transf_test_pred.loc[1, 1] - income_gender_counts_test.loc[1, 1]) / total_test_len), 3)
    female_accuracy = round((1 - abs(income_gender_counts_transf_test_pred.loc[0, 0] - income_gender_counts_test.loc[0, 0]) / total_test_len), 3)
    total_accuracy = round((1 - abs(abs(income_gender_counts_transf_test_pred.loc[1, 1] - income_gender_counts_test.loc[1, 1])+abs(income_gender_counts_transf_test_pred.loc[0, 0] - income_gender_counts_test.loc[0, 0])) / total_test_len), 3)
    return male_accuracy, female_accuracy, total_accuracy

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
    delta = 1 - (rows_not_respecting_rules / total_rows)
    print(f"Consistency: {delta}")
    
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
