import json
import logging
import os
import pandas as pd

from data.compas_utils import preprocess_compas
from models.models_utils import calculate_best_thr, compute_model_performance, get_scores, instantiate_and_fit_model, predict_model
from post_processing_techniques.pptech_utils import apply_pp_technique
from utils.association_rules_utils import clean_association_rules, compute_association_rules, compute_diff_association_rules, export_association_rules
from utils.data_utils import load_data, save_data, prev_unprev, generate_bld, X_y_train, train_test_distribution_plot, stages_distribution_plot
from data.adult_utils import preprocess_adult
from utils.fairness_utils import compute_fairness_metrics
from utils.gen_utils import sns_line_plotting
from utils.quality_utils import compute_accuracy, compute_consistency, plot_accuracy_list, plot_consistency, plot_consistency_list

logging.basicConfig(level=logging.INFO)

performance_list = []
accuracy_list = []
fairness_list = []
consistency_list = []


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
            target_variable_values = ["<50K", ">50K"]
            data = preprocess_adult(data)
        if "compas" in dataset_name:
            target_variable_values = ["risk_low", "risk_high"]
            data = preprocess_compas(data)
        
        # understand which sensitive attribute is privileged and unprivileged
        privileged_groups, unprivileged_groups = prev_unprev(data, sensible_attribute, target_variable)
        #generate binary dataset (AIF360)
        binary_label_dataset = generate_bld(data, target_variable, sensible_attribute)
        dataset_orig_train, dataset_orig_test = binary_label_dataset.split([0.7], shuffle=True)
        scale_orig, X_train, y_train = X_y_train(dataset_orig_train)
        train_test_distribution_plot(binary_label_dataset, dataset_orig_train,
                                     dataset_orig_test, sensible_attribute, target_variable,
                                     plots_dir, dataset_name)
        technique_counter = 0
        for technique in config["techniques"]:
            technique_counter += 1
            technique_name = technique["name"]
            print(f"Applying technique: {technique_name}")
            association_rules_technique = []
            model_counter = 0
            for model in config["models"]:
                model_counter += 1
                model_name = model["name"]
                print(f"Training model: {model_name}")
                model_instance = instantiate_and_fit_model(model, X_train, y_train)
                pos_ind, dataset_orig_train_pred = predict_model(model_instance, dataset_orig_train, X_train)
                dataset_orig_test_pred = get_scores(
                    pos_ind,
                    model_instance,
                    dataset_orig_test,
                    scale_orig)
                best_class_thresh = calculate_best_thr(
                    num_thresh,
                    dataset_orig_test_pred,
                    dataset_orig_test,
                    unprivileged_groups,
                    privileged_groups)
                dataset_transf_test_pred = apply_pp_technique(
                    technique=technique,
                    model_instance=model_instance,
                    best_class_thresh=best_class_thresh,
                    dataset=dataset_orig_test,
                    dataset_pred=dataset_orig_test_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                    sensible_attribute=sensible_attribute)
                #transform dataset in dataframe
                df_orig_test = dataset_orig_test.convert_to_dataframe()[0]
                df_orig_test_pred = dataset_orig_test_pred.convert_to_dataframe()[0]
                df_transf_test_pred = dataset_transf_test_pred.convert_to_dataframe()[0]
                os.makedirs(plots_dir+"/"+dataset_name, exist_ok=True)
                filepath = plots_dir+"/"+dataset_name
                stages_distribution_plot(
                    df_orig_test,
                    df_orig_test_pred,
                    df_transf_test_pred,
                    sensible_attribute,
                    target_variable,
                    plots_dir,
                    dataset_name,
                    technique_name, model_name)
                
                model_accuracy, model_recall, model_precision, model_F1 = compute_model_performance(
                    dataset_orig_test, dataset_orig_test_pred,
                    dataset_transf_test_pred, filepath,
                    technique_name, model_name)                                                                              
                performance_list.append([dataset_name,
                                        technique_name,
                                        model_name,
                                        model_accuracy,
                                        model_recall,
                                        model_precision,
                                        model_F1])
    
                GroupFairness, PredictiveParity, EqualOpportunity = compute_fairness_metrics(
                    df_orig_test, df_orig_test_pred, df_transf_test_pred,
                    unprivileged_groups, privileged_groups, target_variable,
                    sensible_attribute, filepath,
                    technique_name, model_name)
                fairness_list.append([dataset_name,
                                        technique_name,
                                        model_name,
                                        GroupFairness,
                                        PredictiveParity,
                                        EqualOpportunity])

                priv_accuracy, unpriv_accuracy, total_accuracy = compute_accuracy(
                    df_orig_test, df_orig_test_pred, df_transf_test_pred,
                    target_variable,sensible_attribute, filepath,
                    technique_name, model_name)
                accuracy_list.append([dataset_name,
                                      technique_name,
                                      model_name,
                                      priv_accuracy,
                                      unpriv_accuracy,
                                      total_accuracy])
                
                ## Association Rules
                #compute association rules only for the first model in the list
                if model_counter == 1:
                    orig_asso_rules_target = compute_association_rules(
                        dataset = dataset_orig_test, dataset_name = dataset_name,
                        target_variable = target_variable_values,
                        support = user_min_support, confidence = user_min_confidence)
                    filepath = tables_dir+"/"+dataset_name
                    os.makedirs(filepath, exist_ok=True)
                    export_association_rules(orig_asso_rules_target, dataset_name,filepath+"/orig_association_rules_target.csv")
                    orig_asso_rules_target = orig_asso_rules_target.rename(columns={'support': 'orig_support', 'confidence': 'orig_confidence'})
                    association_rules_technique = orig_asso_rules_target.copy()
                transf_asso_rules_target = compute_association_rules(
                    dataset = dataset_transf_test_pred, dataset_name = dataset_name,
                    target_variable = target_variable_values, support = user_min_support,
                    confidence = user_min_confidence)
                print(f"Association rules for {technique_name} technique {model_name} model")
                print(transf_asso_rules_target)
                #transf_asso_rules_target.to_csv(f"{tables_dir}/{dataset_name}/{technique_name}{model_name}_asso_rules.csv", index=False)
                diff_asso_rules = compute_diff_association_rules(
                    association_rules_technique, 
                    transf_asso_rules_target, 
                    model_name )
                consistency = compute_consistency(
                    dataset_orig_test, dataset_transf_test_pred,
                    orig_asso_rules_target, dataset_name)
                diff_asso_rules = compute_diff_association_rules(
                    association_rules_technique, 
                    transf_asso_rules_target, 
                    model_name )
                consistency = compute_consistency(
                    dataset_orig_test, dataset_transf_test_pred,
                    orig_asso_rules_target, dataset_name)
                consistency_list.append([
                    dataset_name, 
                    technique_name,
                    model_name, 
                    consistency
                ])
            if diff_asso_rules is not None:
                diff_asso_rules.to_csv(f"{tables_dir}/{dataset_name}/{technique_name}_diff_asso_rules.csv", index=False)
            plot_consistency_list(consistency_list, plots_dir, dataset_name, technique_name)
            # plot_accuracy_list(accuracy_list, plots_dir, dataset_name, technique_name)

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
    

    consistency_df = pd.DataFrame(consistency_list, columns=[
        "dataset_name", "technique_name", "model_name", "consistency"
    ])
    consistency_df.to_csv(f"{tables_dir}/consistency_results.csv", index=False)
    filepath = plots_dir+"/"+dataset_name+"_consistency.png"
    plot_consistency(consistency_df, dataset_name, filepath)

if __name__ == "__main__":
    main("config/config.json")