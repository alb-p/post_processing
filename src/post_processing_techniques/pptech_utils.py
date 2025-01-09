from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import DeterministicReranking

from fairlearn.postprocessing import ThresholdOptimizer


def apply_pp_technique(technique, model_instance, best_class_thresh, dataset, dataset_pred, unprivileged_groups, privileged_groups, sensible_attribute, target_variable):
    technique_name = technique["name"].lower()
    technique_params = technique["params"]
    already_fitted = False
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
        technique_instance = RejectOptionClassification(
            unprivileged_groups=unprivileged_groups,
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
        technique_instance = CalibratedEqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            cost_constraint=cost_constraint,
            seed=randseed)
    elif "equalized odds" in technique_name:
        randseed = technique_params["randseed"]
        technique_instance = EqOddsPostprocessing(
            privileged_groups = privileged_groups,
            unprivileged_groups = unprivileged_groups,
            seed = randseed)
    elif "reranking" in technique_name:
        '''
        The reranking process enforces constraints as minimum
        and/or maximum representation of each group at different positions
        in the ranked list.
        '''
        technique_instance  = DeterministicReranking(
            privileged_groups = privileged_groups,
            unprivileged_groups = unprivileged_groups)
        technique_instance = technique_instance.fit(dataset)
        reranking_type = technique_params["type"]
        sensible_attribute = list(privileged_groups[0].keys())[0]
        privileged_index = list(privileged_groups[0].values())[0]
        unprivileged_index = list(unprivileged_groups[0].values())[0]
        df_pred = dataset_pred.convert_to_dataframe()[0]
        len_pred = df_pred.shape[0]
        group_sizes = {
        "privileged": df_pred.groupby(sensible_attribute).size()[privileged_index],
        "unprivileged": df_pred.groupby(sensible_attribute).size()[unprivileged_index],
        }
        total_size = sum(group_sizes.values())
        target_prop = [
            group_sizes["unprivileged"] / total_size,
            group_sizes["privileged"] / total_size
        ]

        already_fitted = True

    elif "threshold" in technique_name:
        constraints = technique_params["constraints"]
        objective = technique_params["objective"]
        technique_instance = ThresholdOptimizer(estimator=model_instance,
                   constraints=constraints,
                   objective=objective,
                   prefit=True,
                   predict_method='predict_proba')
        df_pred = dataset_pred.convert_to_dataframe()[0]
        X = df_pred.drop([target_variable], axis=1)
        y = df_pred[target_variable]
        y.astype(int)
        sensible_array = df_pred[sensible_attribute]
        technique_instance = technique_instance.fit(X, y, sensitive_features=sensible_array)
        already_fitted = True
    else:
        raise ValueError(f"Technique {technique_name} not supported.")
    
    if not already_fitted:
        technique_instance = technique_instance.fit(dataset, dataset_pred)

    fav_inds = dataset_pred.scores > best_class_thresh
    dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
    dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label

    if not already_fitted:
        dataset_transf = technique_instance.predict(dataset_pred)
    elif "reranking" in technique_name:
        dataset_transf = technique_instance.predict(
            dataset=dataset_pred,
            rec_size=len_pred,
            target_prop=target_prop,
            rerank_type=reranking_type,
            renormalize_scores = False)
        
        # target_prop = [0.5,0.5]
        # dataset_transf = technique_instance.fit_predict(
        #     dataset=dataset_pred,
        #     rec_size=len_pred,
        #     target_prop=target_prop,
        #     rerank_type=reranking_type,
        #     renormalize_scores = True)
    elif "threshold" in technique_name:
        TO_transf_test_pred = technique_instance.predict(X, sensitive_features=sensible_array)
        dataset_transf = dataset.copy(deepcopy=True)
        dataset_transf.labels = TO_transf_test_pred.reshape(-1, 1)

    return dataset_transf