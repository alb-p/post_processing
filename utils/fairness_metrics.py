from sklearn.metrics import confusion_matrix

def fairness_metrics_division(TP_discr, TN_discr, FP_discr, FN_discr, len_discr,
                              TP_priv, TN_priv, FP_priv, FN_priv, len_priv, threshold):
    metrics = {}

    # Compute each metric
    metrics['GroupFairness'] = compute_group_fairness(TP_discr, FP_discr, len_discr,
                                                      TP_priv, FP_priv, len_priv)
    metrics['PredictiveParity'] = compute_predictive_parity(TP_discr, FP_discr,
                                                            TP_priv, FP_priv)
    metrics['PredictiveEquality'] = compute_predictive_equality(FP_discr, TN_discr,
                                                                FP_priv, TN_priv)
    metrics['EqualOpportunity'] = compute_equal_opportunity(FN_discr, TP_discr,
                                                            FN_priv, TP_priv)
    metrics['EqualizedOdds'] = compute_equalized_odds(TP_discr, FN_discr, TN_discr, FP_discr,
                                                      TP_priv, FN_priv, TN_priv, FP_priv, threshold)
    metrics['ConditionalUseAccuracyEquality'] = compute_conditional_use_accuracy(TP_discr, FP_discr, TN_discr, FN_discr,
                                                                                  TP_priv, FP_priv, TN_priv, FN_priv, threshold)
    metrics['OverallAccuracyEquality'] = compute_overall_accuracy(TP_discr, TN_discr, TP_priv, TN_priv, threshold)
    metrics['TreatmentEquality'] = compute_treatment_equality(FP_discr, FN_discr, FP_priv, FN_priv)
    metrics['FORParity'] = compute_for_parity(FN_discr, TN_discr, FN_priv, TN_priv)
    metrics['FN'] = compute_fn_parity(FN_discr, len_discr, FN_priv, len_priv)
    metrics['FP'] = compute_fp_parity(FP_discr, len_discr, FP_priv, len_priv)

    # Standardize and scale the metrics
    for key, value in metrics.items():
        metrics[key] = {
            'Value': standardize_and_rescale(value[0]),
            'Discr_group': value[1],
            'Priv_group': value[2]
        }

    return metrics

# Helper functions
def compute_group_fairness(TP_discr, FP_discr, len_discr, TP_priv, FP_priv, len_priv):
    group_fairness_discr = (TP_discr + FP_discr) / len_discr if len_discr > 0 else 0
    group_fairness_priv = (TP_priv + FP_priv) / len_priv if len_priv > 0 else 0
    group_fairness = (group_fairness_discr / group_fairness_priv) if group_fairness_priv > 0 else 2
    return group_fairness, group_fairness_discr, group_fairness_priv

def compute_predictive_parity(TP_discr, FP_discr, TP_priv, FP_priv):
    discr_ratio = TP_discr / (TP_discr + FP_discr) if TP_discr + FP_discr > 0 else 0
    priv_ratio = TP_priv / (TP_priv + FP_priv) if TP_priv + FP_priv > 0 else 0
    predictive_parity = (discr_ratio / priv_ratio) if priv_ratio > 0 else 2
    return predictive_parity, discr_ratio, priv_ratio

def compute_predictive_equality(FP_discr, TN_discr, FP_priv, TN_priv):
    discr_ratio = FP_discr / (TN_discr + FP_discr) if TN_discr + FP_discr > 0 else 0
    priv_ratio = FP_priv / (TN_priv + FP_priv) if TN_priv + FP_priv > 0 else 0
    predictive_equality = (discr_ratio / priv_ratio) if priv_ratio > 0 else 2
    return predictive_equality, discr_ratio, priv_ratio

def compute_equal_opportunity(FN_discr, TP_discr, FN_priv, TP_priv):
    discr_ratio = FN_discr / (TP_discr + FN_discr) if TP_discr + FN_discr > 0 else 0
    priv_ratio = FN_priv / (TP_priv + FN_priv) if TP_priv + FN_priv > 0 else 0
    equal_opportunity = (priv_ratio / discr_ratio) if discr_ratio > 0 else 2
    return equal_opportunity, discr_ratio, priv_ratio

def compute_equalized_odds(TP_discr, FN_discr, TN_discr, FP_discr,
                           TP_priv, FN_priv, TN_priv, FP_priv, threshold):
    odds1 = compute_equal_opportunity(FN_discr, TP_discr, FN_priv, TP_priv)[0]
    odds2 = compute_predictive_equality(FP_discr, TN_discr, FP_priv, TN_priv)[0]
    if odds1 != 0 and odds2 != 0:
        equalized_odds = and_function(odds1, odds2, threshold)
    else:
        equalized_odds = 2
    return equalized_odds, odds1, odds2

def compute_conditional_use_accuracy(TP_discr, FP_discr, TN_discr, FN_discr,
                                     TP_priv, FP_priv, TN_priv, FN_priv, threshold):
    accuracy1 = compute_predictive_parity(TP_discr, FP_discr, TP_priv, FP_priv)[0]
    accuracy2 = compute_predictive_equality(TN_discr, FN_discr, TN_priv, FN_priv)[0]
    if accuracy1 != 0 and accuracy2 != 0:
        conditional_accuracy = and_function(accuracy1, accuracy2, threshold)
    else:
        conditional_accuracy = 2
    return conditional_accuracy, accuracy1, accuracy2

def compute_overall_accuracy(TP_discr, TN_discr, TP_priv, TN_priv, threshold):
    oae1 = TP_discr / TP_priv if TP_priv > 0 else 2
    oae2 = TN_discr / TN_priv if TN_priv > 0 else 2
    if oae1 != 0 and oae2 != 0:
        overall_accuracy = and_function(oae1, oae2, threshold)
    else:
        overall_accuracy = 2
    return overall_accuracy, oae1, oae2

def compute_treatment_equality(FP_discr, FN_discr, FP_priv, FN_priv):
    priv_ratio = FN_priv / FP_priv if FP_priv > 0 else 2
    discr_ratio = FN_discr / FP_discr if FP_discr > 0 else 0
    treatment_equality = priv_ratio / discr_ratio if discr_ratio > 0 else 0
    return treatment_equality, discr_ratio, priv_ratio

def compute_for_parity(FN_discr, TN_discr, FN_priv, TN_priv):
    discr_ratio = FN_discr / (TN_discr + FN_discr) if TN_discr + FN_discr > 0 else 0
    priv_ratio = FN_priv / (TN_priv + FN_priv) if TN_priv + FN_priv > 0 else 0
    for_parity = priv_ratio / discr_ratio if discr_ratio > 0 else 0
    return for_parity, discr_ratio, priv_ratio

def compute_fn_parity(FN_discr, len_discr, FN_priv, len_priv):
    fn_discr = FN_discr / len_discr if len_discr > 0 else 0
    fn_priv = FN_priv / len_priv if len_priv > 0 else 0
    fn_metric = fn_priv / fn_discr if fn_discr > 0 else 2
    return fn_metric, fn_discr, fn_priv

def compute_fp_parity(FP_discr, len_discr, FP_priv, len_priv):
    fp_discr = FP_discr / len_discr if len_discr > 0 else 0
    fp_priv = FP_priv / len_priv if len_priv > 0 else 0
    fp_metric = fp_discr / fp_priv if fp_priv > 0 else 0
    return fp_metric, fp_discr, fp_priv

def and_function(m1, m2, th):
    if m1 > 1 + th and m2 > 1 + th:
        return max(m1, m2)
    elif m1 < 1 - th and m2 < 1 - th:
        return min(m1, m2)
    elif (1 - th <= m1 <= 1 + th) and (1 - th <= m2 <= 1 + th):
        return max(m1, m2)
    elif (1 - th <= m1 <= 1 + th or 1 - th <= m2 <= 1 + th) and (m1 > 1 + th or m2 > 1 + th):
        return max(m1, m2)
    elif (1 - th <= m1 <= 1 + th or 1 - th <= m2 <= 1 + th) and (m1 < 1 - th or m2 < 1 - th):
        return min(m1, m2)
    else:
        return max(m1, m2)

def standardize_and_rescale(metric):
    metric = metric - 1
    return max(-1, min(metric, 1))
