from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import DeterministicReranking

from fairlearn.postprocessing import ThresholdOptimizer


def apply_pp_techinque(technique,best_class_thresh, dataset, dataset_pred, unprivileged_groups, privileged_groups):
    technique_name = technique["name"].lower()
    technique_params = technique["params"]
    if "reject option" in technique_name:
        metric_name = technique_params["metric_name"]
        low_class_thresh = technique_params["low_class_thresh"]
        high_class_thresh = technique_params["high_class_thresh"]
        num_class_thresh = technique_params["num_class_thresh"]
        num_ROC_margin = technique_params["num_ROC_margin"]
        metric_ub = technique_params["metric_ub"]
        metric_lb = technique_params["metric_lb"]
        allowed_metrics = ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]
        if metric_name not in allowed_metrics:
            raise ValueError("Metric name should be one of allowed metrics")
        technique_instance = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups,
                                                        low_class_thresh=low_class_thresh,
                                                        high_class_thresh=high_class_thresh,
                                                        num_class_thresh=num_class_thresh,
                                                        num_ROC_margin=num_ROC_margin,
                                                        metric_name=metric_name,
                                                        metric_ub=metric_ub,
                                                        metric_lb=metric_lb)
    elif "calibrated" in technique_name:
        cost_constraint=technique_params["constraint"]
        randseed = technique_params["randseed"]
        technique_instance = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups,
                                                             cost_constraint=cost_constraint,
                                                             seed=randseed)
    elif "equalized odds" in technique_name:
        randseed = technique_params["randseed"]
        technique_instance = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                                unprivileged_groups = unprivileged_groups,
                                                seed = randseed)
    elif "reranking" in technique_name:
        pass
    elif "threshold" in technique_name:
        pass
    else:
        raise ValueError(f"Technique {technique_name} not supported.")
    
    technique_instance = technique_instance.fit(dataset, dataset_pred)


    fav_inds = dataset_pred.scores > best_class_thresh
    dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
    dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label
    dataset_pred = technique_instance.predict(dataset_pred)
    return dataset_pred