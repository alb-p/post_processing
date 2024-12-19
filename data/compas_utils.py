import pandas as pd

# sex,age,race,juv-fel-count,juv-misd-count,juv-other-count,c-charge-degree,priors-count,recidive,risk
# Sex,Race,ScaleSet_ID,ScaleSet,AssessmentReason,Language,LegalStatus,CustodyStatus,MaritalStatus,Scale_ID,Age,Risk
# sex,age,race,juv-fel-count,juv-misd-count,juv-other-count,priors-count,risk,c-charge-degree,recidive

def preprocess_compas(dataset):
    """The custom pre-processing function is adapted from
        https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
    """
    # Select relevant columns
    dataset_preproc = dataset[['sex', 'race', 'age', 'c_charge_degree', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'is_recid', 'decile_score']].copy()

    def group_race(x):
        if x == "Caucasian":
            return 1.0
        elif x == "African-American":
            return 0.0
        else:
            return -1.0

    # Map 'race' values
    dataset_preproc['race'] = dataset_preproc['race'].apply(lambda x: group_race(x))

    # Filter rows where 'race' is valid
    dataset_preproc = dataset_preproc[dataset_preproc['race'] != -1.0]

    # Map other columns
    dataset_preproc['risk'] = dataset_preproc['decile_score'].apply(lambda x: 1.0 if x < 6 else 0.0)
    dataset_preproc.drop(columns=['decile_score'], inplace=True)
    dataset_preproc['c_charge_degree'] = dataset_preproc['c_charge_degree'].map({'F': 1.0, 'M': 0.0})
    dataset_preproc['sex'] = dataset_preproc['sex'].map({'Female': 0.0, 'Male': 1.0})

    return dataset_preproc

def prepare_compas_asso_rules(dataset_to_prepare): 
    print("Preparing COMPAS dataset for association rules...")
    # Convert the dataset to a DataFrame
    if not isinstance(dataset_to_prepare, pd.DataFrame):
        df_prepared = dataset_to_prepare.convert_to_dataframe()[0]
    else:
        df_prepared = dataset_to_prepare

    def map_priors_count(x):
        if x == 0:
            return 'No'
        elif 1 <= x <= 5:
            return 'Few'
        else:
            return 'Many'
    
    df_prepared['male'] = df_prepared['sex'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['female'] = df_prepared['sex'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='sex', inplace=True) 

    # Process 'race' column
    df_prepared['white'] = df_prepared['race'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['n_white'] = df_prepared['race'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='race', inplace=True)

    # Group age by decade (i.e. decade_20 represents ages between 20 and 29)
    df_prepared['Age (decade)'] = df_prepared['age'].apply(lambda x: x//10*10)
    df_prepared = df_prepared.drop(columns='age')

    # Process 'c_charge_degree' column
    df_prepared['c_felony'] = df_prepared['c_charge_degree'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['c_misdemeanor'] = df_prepared['c_charge_degree'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='c_charge_degree', inplace=True)

    df_prepared['juv_fel>0'] = df_prepared['juv_fel_count'].apply(lambda x: 1 if x > 0 else 0)
    df_prepared['juv_fel=0'] = df_prepared['juv_fel_count'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='juv_fel_count', inplace=True)

    df_prepared['juv_misd>0'] = df_prepared['juv_misd_count'].apply(lambda x: 1 if x > 0 else 0)
    df_prepared['juv_misd=0'] = df_prepared['juv_misd_count'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='juv_misd_count', inplace=True)

    df_prepared['priors'] = df_prepared['priors_count'].apply(map_priors_count)
    df_prepared.drop(columns='priors_count', inplace=True)

    df_prepared['recidive'] = df_prepared['is_recid'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['n_recidive'] = df_prepared['is_recid'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='is_recid', inplace=True)

    # Process 'risk' column
    df_prepared['risk_low'] = df_prepared['risk'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['risk_high'] = df_prepared['risk'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='risk', inplace=True)

    df_prepared = pd.get_dummies(df_prepared, columns=['priors','Age (decade)'],drop_first=False) 
    return df_prepared

def group_edu(x):
    if x <= 5:
        return '<6'
    elif x >= 13:
        return '>12'
    else:
        return x
    