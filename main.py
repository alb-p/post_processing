import json

from models.models_utils import calculate_best_thr, get_scores, instanciate_and_fit_model, predict_model
from post_processing_techniques.pptech_utils import apply_pp_techinque
from utils.data_utils import load_data, save_data, generate_bld, X_y_train
from data.adult_utils import preprocess_adult, prev_unprev


def load_config(config_path):
    """
    Loads configuration from a JSON file.
    
    Args:
        config_path (str): Path to the JSON configuration file.
    
    Returns:
        dict: Loaded configuration.
    """
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
    num_thresh = config["num_thresh_test"]
    user_min_support = config["min_support"]
    user_min_confidence = config["min_confidence"]

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
        print(f"Privileged groups: {privileged_groups}")
        print(f"Unprivileged groups: {unprivileged_groups}")
        #generate binary dataset (AIF360)
        binary_label_dataset = generate_bld(data, target_variable, sensible_attribute)
        dataset_orig_train, dataset_orig_test = binary_label_dataset.split([0.7], shuffle=True)
        scale_orig, X_train, y_train = X_y_train(dataset_orig_train)
        #plot data distribution train/test?
        for technique in config["techniques"]:
            technique_name = technique["name"]
            print(f"Applying technique: {technique_name}")
            for model in config["models"]:
                model_name = model["name"]
                print(f"Training model: {model_name}")
                model_instance = instanciate_and_fit_model(model, X_train, y_train)
                pos_ind, dataset_orig_train_pred = predict_model(model_instance, dataset_orig_train, X_train)
                dataset_orig_test_pred = get_scores(pos_ind, model_instance, dataset_orig_test, scale_orig)
                best_class_thresh = calculate_best_thr(num_thresh, dataset_orig_test_pred, dataset_orig_test, unprivileged_groups, privileged_groups)
                #plot model's performance
                dataset_transf_test_pred = apply_pp_techinque(technique, best_class_thresh, dataset_orig_test, dataset_orig_test_pred, privileged_groups, unprivileged_groups)
                #compute conf_matrix, accuracy and fairness metrics






if __name__ == "__main__":
    main("config/config.json")