import json
import logging
import os
import pandas as pd

from models.models_utils import calculate_best_thr, calculate_model_performance, get_scores, instanciate_and_fit_model, predict_model
from post_processing_techniques.pptech_utils import apply_pp_techinque
from utils.association_rules_utils import clean_association_rules, compute_association_rules, compute_diff_association_rules, export_association_rules
from utils.data_utils import load_data, save_data, generate_bld, X_y_train
from data.adult_utils import preprocess_adult, prev_unprev
from utils.fairness_utils import compute_fairness_metrics
from utils.quality_utils import compute_accuracy, compute_consistency

logging.basicConfig(level=logging.INFO)

performance_list = []
accuracy_list = []
fairness_list = []

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
    
def main(config_path):
    """
    Main pipeline to handle multiple datasets, models, and techniques.
    
    Args:
        config_path (str): Path to the JSON configuration file.
    """
    # Load configuration
    config = load_config(config_path)
    tables_dir = config["tables_dir"]
    plots_dir = config["plots_dir"]
    num_thresh = config["num_thresh_test"]
    user_min_support = config["min_support"]
    user_min_confidence = config["min_confidence"]

    # os.makedirs(tables_dir, exist_ok=True)

    technique_counter = 0
    model_counter = 0

    for dataset in config["datasets"]:
        dataset_path = dataset["path"]
        dataset_name = dataset["name"]

        target_variable = dataset["parameters"]["target_variable"]
        sensible_attribute = dataset["parameters"]["sensible_attribute"]

        print(f"Processing dataset: {dataset_name}")
        data = load_data(dataset_path)
        if "adult" in dataset_name:
            data = preprocess_adult(data)
            #plot dataset's distribution
            
        # understand which sensitive attribute is privileged and unprivileged
        privileged_groups, unprivileged_groups = prev_unprev(data, sensible_attribute, target_variable)
        #generate binary dataset (AIF360)
        binary_label_dataset = generate_bld(data, target_variable, sensible_attribute)
        dataset_orig_train, dataset_orig_test = binary_label_dataset.split([0.7], shuffle=True)
        scale_orig, X_train, y_train = X_y_train(dataset_orig_train)
        #plot data distribution train/test?
        for technique in config["techniques"]:
            technique_counter += 1
            technique_name = technique["name"]
            print(f"Applying technique: {technique_name}")
            association_rules_technique = []
            for model in config["models"]:
                model_counter += 1
                model_name = model["name"]
                print(f"Training model: {model_name}")
                model_instance = instanciate_and_fit_model(model, X_train, y_train)
                pos_ind, dataset_orig_train_pred = predict_model(model_instance, dataset_orig_train, X_train)
                dataset_orig_test_pred = get_scores(pos_ind, model_instance, dataset_orig_test, scale_orig)
                best_class_thresh = calculate_best_thr(num_thresh, dataset_orig_test_pred, dataset_orig_test, unprivileged_groups, privileged_groups)
                #plot model's performance
                #potrei ritornarlo già df?
                dataset_transf_test_pred = apply_pp_techinque(technique=technique, best_class_thresh=best_class_thresh,
                                                            dataset=dataset_orig_test, dataset_pred=dataset_orig_test_pred,
                                                            unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                #transform dataset in dataframe
                df_orig_test = dataset_orig_test.convert_to_dataframe()[0]
                df_orig_test_pred = dataset_orig_test_pred.convert_to_dataframe()[0]
                df_transf_test_pred = dataset_transf_test_pred.convert_to_dataframe()[0]

                model_accuracy, model_recall, model_precision, model_F1 = calculate_model_performance(dataset_orig_test, dataset_transf_test_pred)
                performance_list.append([dataset_name,
                                        technique_name,
                                        model_name,
                                        model_accuracy,
                                        model_recall,
                                        model_precision,
                                        model_F1])
    
                #compute conf_matrix, accuracy and fairness metrics
                GroupFairness, PredictiveParity, EqualOpportunity = compute_fairness_metrics(df_orig_test, df_orig_test_pred, df_transf_test_pred, unprivileged_groups, privileged_groups, target_variable, sensible_attribute)
                fairness_list.append([dataset_name,
                                        technique_name,
                                        model_name,
                                        GroupFairness,
                                        PredictiveParity,
                                        EqualOpportunity])

                priv_accuracy, unpriv_accuracy, total_accuracy = compute_accuracy(df_orig_test, df_transf_test_pred, unprivileged_groups, privileged_groups, target_variable, sensible_attribute)
                accuracy_list.append([dataset_name,
                                      technique_name,
                                      model_name,
                                      priv_accuracy,
                                      unpriv_accuracy,
                                      total_accuracy])
                ## Association Rules
                #compute association rules only for the first model in the list
                if technique_counter == 1 and model_counter == 1:
                    orig_asso_rules_target = compute_association_rules(dataset = dataset_orig_test, dataset_name = dataset_name, target_variable = df_orig_test[target_variable].unique, support = user_min_support, confidence = user_min_confidence)
                    print(orig_asso_rules_target)
                    export_association_rules(orig_asso_rules_target, dataset_name,tables_dir+"/"+dataset_name+"_orig_asso_rules_target.csv")
                    orig_asso_rules_target = orig_asso_rules_target.rename(columns={'support': 'orig_support', 'confidence': 'orig_confidence'})
                    association_rules_technique = orig_asso_rules_target.copy()
                transf_asso_rules_target = compute_association_rules(dataset = dataset_transf_test_pred, dataset_name = dataset_name, target_variable = target_variable, support = user_min_support, confidence = user_min_confidence)
                compute_diff_association_rules(association_rules_technique, transf_asso_rules_target,model_name, tables_dir+"/"+technique_name+"_complete_asso_rules.csv")
                # export_association_rules(diff_asso_rules, tables_dir+"/diff_asso_rules.csv")
                compute_consistency(dataset_orig_test, dataset_transf_test_pred, orig_asso_rules_target, dataset_name)
        #save association_rules_technique


    performance_df = pd.DataFrame(performance_list, columns=[
        "dataset_name", "technique_name", "model_name",
        "accuracy", "recall", "precision", "F1"
    ])

    performance_df.to_csv(f"{tables_dir}/performance_results.csv", index=False)

    accuracy_df = pd.DataFrame(accuracy_list, columns=[
        "dataset_name", "technique_name", "model_name",
        "priv_accuracy", "unpriv_accuracy", "overall_accuracy"
    ])
    accuracy_df.to_csv(f"{tables_dir}/accuracy_results.csv", index=False)

    fairness_df = pd.DataFrame(fairness_list, columns=[
        "dataset_name", "technique_name", "model_name",
        "GroupFairness", "PredictiveParity", "EqualOpportunity"
    ])
    fairness_df.to_csv(f"{tables_dir}/fairness_results.csv", index=False)



if __name__ == "__main__":
    main("config/config.json")