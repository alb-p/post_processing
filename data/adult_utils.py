
def preprocess_adult(dataset):
    to_drop = ["fnlwgt", "education", "relationship", "workclass", "occupation"]
    dataset_preproc = dataset.dropna()
    dataset_preproc.drop(to_drop, axis=1, inplace=True)

    dataset_preproc.loc[:, 'income'] = dataset_preproc['income'].map({'<=50K': 0, '>50K': 1})
    dataset_preproc['income'] = dataset_preproc['income'].astype(int)
    dataset_preproc.loc[:, 'sex'] = dataset_preproc['sex'].map({'Female': 0, 'Male': 1})
    dataset_preproc['sex'] = dataset_preproc['sex'].astype(int)
    dataset_preproc['marital.status'] = dataset_preproc['marital.status'].map({
        'Never-married': 0,
        'Divorced': 0,
        'Separated': 0,
        'Widowed': 0,
        'Married-civ-spouse': 1,
        'Married-spouse-absent': 1,
        'Married-AF-spouse': 1
    })
    dataset_preproc['marital.status'] = dataset_preproc['marital.status'].astype(int)
    dataset_preproc['race'] = dataset_preproc['race'].apply(lambda x: 1 if x == 'White' else 0)
    dataset_preproc['race'] = dataset_preproc['race'].astype(int)
    dataset_preproc['native.country'] = dataset_preproc['native.country'].apply(lambda x: 1 if x == 'United-States' else 0)
    dataset_preproc['native.country'] = dataset_preproc['native.country'].astype(int)

    return dataset_preproc


def prev_unprev(dataset, sensible_attribute, target_variable):
    income_gender_counts = dataset.groupby([sensible_attribute, target_variable]).size().unstack()
    if (income_gender_counts.loc[0][1] < income_gender_counts.loc[1][1]):
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    else:
        privileged_groups = [{'sex': 0}]
        unprivileged_groups = [{'sex': 1}]
    return privileged_groups, unprivileged_groups