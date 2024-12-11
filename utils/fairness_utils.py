import pandas as pd
from sklearn.metrics import confusion_matrix

from utils.fairness_metrics import fairness_metrics_division
from utils.gen_utils import sns_line_plotting

def compute_fairness_metrics(df_orig_test, df_orig_test_pred, df_transf_test_pred, unprivileged_groups, privileged_groups, target_variable, sensible_attribute, filepath, technique_name, model_name):
    
    TP_discr, TN_discr, FN_discr, FP_discr, len_discr, TP_priv, TN_priv, FN_priv, FP_priv, len_priv = preliminary(df_orig_test, df_orig_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, target_variable=target_variable, sensible_attribute=sensible_attribute)
    TP_discr_pp, TN_discr_pp, FN_discr_pp, FP_discr_pp, len_discr_pp, TP_priv_pp, TN_priv_pp, FN_priv_pp, FP_priv_pp, len_priv_pp = preliminary(df_orig_test, df_transf_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, target_variable=target_variable, sensible_attribute=sensible_attribute)
    
    metrics_before = fairness_metrics_division(TP_discr, TN_discr, FP_discr, FN_discr, len_discr, TP_priv, TN_priv, FP_priv, FN_priv, len_priv, threshold=0.15)
    metrics_after = fairness_metrics_division(TP_discr_pp, TN_discr_pp, FP_discr_pp, FN_discr_pp, len_discr_pp, TP_priv_pp, TN_priv_pp, FP_priv_pp, FN_priv_pp, len_priv_pp, threshold=0.15)
    
    fairness_metrics_selection = ['GroupFairness', 'PredictiveParity', 'EqualOpportunity']
    before_values = [metrics_before[metric]['Value'] for metric in fairness_metrics_selection]
    after_values = [metrics_after[metric]['Value'] for metric in fairness_metrics_selection]

    df_fairness = pd.DataFrame({
        'Metrics': ['GroupFairness', 'PredictiveParity', 'EqualOpportunity'],
        'Before': before_values,
        'After': after_values
    })
    filepath = filepath + "/"+technique_name+"_"+model_name+"_fairness_metrics.png"
    sns_line_plotting(df=df_fairness, axhline=0, filepath=filepath, title=f'{technique_name} - {model_name}: Fairness metrics')
    return round(abs(metrics_before['GroupFairness']['Value'])-abs(metrics_after['GroupFairness']['Value']),3),round(abs(metrics_before['PredictiveParity']['Value'])- abs(metrics_after['PredictiveParity']['Value']),3),round(abs(metrics_before['EqualOpportunity']['Value'])- abs(metrics_after['EqualOpportunity']['Value']),3)

def preliminary(df_before, df_after, unprivileged_groups, privileged_groups, target_variable, sensible_attribute):
    y_before_privileged, y_before_discriminated = get_test_groups(df_before, unprivileged_groups, privileged_groups, target_variable, sensible_attribute)
    y_after_privileged, y_after_discriminated= get_test_groups(df_after, unprivileged_groups, privileged_groups, target_variable, sensible_attribute)
    assert(len(y_before_privileged) == len(y_after_privileged))
    assert(len(y_before_discriminated) == len(y_after_discriminated))
    TP_discr, TN_discr, FN_discr, FP_discr, len_discr = calculate_confusion_matrix(y_before_discriminated, y_after_discriminated)
    TP_priv, TN_priv, FN_priv, FP_priv, len_priv = calculate_confusion_matrix(y_before_privileged, y_after_privileged)
    return TP_discr, TN_discr, FN_discr, FP_discr, len_discr, TP_priv, TN_priv, FN_priv, FP_priv, len_priv

def calculate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) 
    TP = cm[1][1]
    TN = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]
    total = TP + TN + FN + FP
    
    return TP, TN, FN, FP, total

def get_test_groups(df_test, unprivileged_groups, privileged_groups, target_variable, sensible_attribute):
    test_discriminated = df_test.loc[df_test[sensible_attribute] == unprivileged_groups[0][sensible_attribute]]
    test_privileged = df_test.loc[df_test[sensible_attribute] == privileged_groups[0][sensible_attribute]]
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Privileged group size: {len(test_privileged)}, Discriminated group size: {len(test_discriminated)}')
    y_test_discriminated = test_discriminated[target_variable].astype(int)
    y_test_privileged = test_privileged[target_variable].astype(int)
    return y_test_privileged, y_test_discriminated
