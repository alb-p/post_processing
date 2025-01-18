import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from utils.gen_utils import sns_line_plotting

def instantiate_and_fit_model(model, X_train, y_train):
    model_name = model["name"].lower()
    if "decision tree" in model_name:
        model_instance = DecisionTreeClassifier()
    elif "logistic regression" in model_name:
        model_instance = LogisticRegression()
    elif "random forest" in model_name:
        model_instance = RandomForestClassifier()
    elif "knn" in model_name:
        model_instance = KNeighborsClassifier()
    else:
        raise ValueError(f"Model {model} not supported.")
    model_instance.fit(X_train, y_train)
    return model_instance

def predict_model(model_instance, dataset_train, X_train):
    y_train_pred = model_instance.predict(X_train)
    pos_ind = np.where(model_instance.classes_ == dataset_train.favorable_label)[0][0]
    dataset_orig_train_pred = dataset_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred
    return pos_ind, dataset_orig_train_pred

def get_scores(pos_ind, model_instance, dataset_orig_test, scale_orig):
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    dataset_orig_test_pred.scores = model_instance.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    return dataset_orig_test_pred

def calculate_best_thr(num_thresh, dataset_orig_test_pred, dataset_orig_test, unprivileged_groups, privileged_groups):
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        fav_inds = dataset_orig_test_pred.scores > class_thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        # TODO: serve?
        # classified_metric_orig_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        ba_arr[idx] = balanced_accuracy_score(dataset_orig_test.labels, dataset_orig_test_pred.labels)

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]
    return best_class_thresh

def compute_model_performance(dataset_orig_test, dataset_orig_test_pred,dataset_transf_test_pred, filepath, technique_name, model_name):
    accuracy_before = round(accuracy_score(dataset_orig_test.labels, dataset_orig_test_pred.labels), 3)
    recall_before = round(recall_score(dataset_orig_test.labels, dataset_orig_test_pred.labels), 3)
    precision_before = round(precision_score(dataset_orig_test.labels, dataset_orig_test_pred.labels), 3)
    F1_before = round(f1_score(dataset_orig_test.labels, dataset_orig_test_pred.labels), 3)

    accuracy_after = round(accuracy_score(dataset_orig_test.labels, dataset_transf_test_pred.labels), 3)
    recall_after = round(recall_score(dataset_orig_test.labels, dataset_transf_test_pred.labels), 3)
    precision_after = round(precision_score(dataset_orig_test.labels, dataset_transf_test_pred.labels), 3)
    F1_after = round(f1_score(dataset_orig_test.labels, dataset_transf_test_pred.labels), 3)
    return round(accuracy_after - accuracy_before, 3), round(recall_after - recall_before, 3), round(precision_after - precision_before, 3), round(F1_after - F1_before, 3)