import pandas as pd
from utils.data_utils import preprocess_dataset

def preprocess_adult(dataset):
    to_drop = ["fnlwgt", "education", "relationship", "workclass", "occupation"]
    dataset_preproc = dataset.preprocess_dataset()
    dataset_preproc.drop(to_drop, axis=1, inplace=True)

    dataset_preproc.loc[:, 'income'] = dataset_preproc['income'].map({'<=50K': 0, '>50K': 1})
    dataset_preproc['income'] = dataset_preproc['income'].astype(int)
    dataset_preproc.loc[:, 'sex'] = dataset_preproc['sex'].map({'Female': 0, 'Male': 1})
    dataset_preproc['sex'] = dataset_preproc['sex'].astype(int)
    dataset_preproc['marital.status'] = dataset_preproc['marital.status'].map({
        'Never-married': 0,
        'Divorced': 0,
        'Separated': 0,
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

def prepare_adult_asso_rules(dataset_to_prepare): 
    print("Preparing Adult dataset for association rules...")
    # Convert the dataset to a DataFrame
    if not isinstance(dataset_to_prepare, pd.DataFrame):
        df_prepared = dataset_to_prepare.convert_to_dataframe()[0]
    else:
        df_prepared = dataset_to_prepare

    # Education number from 0 to 5 are put in '<6' and those above 13 into '>12'
    df_prepared['education.num'] = df_prepared['education.num'].apply(lambda x: group_edu(x))
    df_prepared['education.num'] = df_prepared['education.num'].astype(str)
    # Group age by decade (i.e. decade_20 represents ages between 20 and 29)
    df_prepared['Age (decade)'] = df_prepared['age'].apply(lambda x: x//10*10)
    df_prepared = df_prepared.drop(columns='age')
    # Group hours per week
    df_prepared['hours.per.week'] = df_prepared['hours.per.week'].apply(lambda x: '>40' if x > 40 else '<=40')

    # Capital gain is splitted into two columns capital.gain>0 and capital.gain=0
    df_prepared['capital.gain>0'] = df_prepared['capital.gain'].apply(lambda x: 1 if x > 0 else 0)
    df_prepared['capital.gain=0'] = df_prepared['capital.gain'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='capital.gain', inplace=True)

    # Capital loss is splitted into two columns capital.loss>0 and capital.loss=0
    df_prepared['capital.loss>0'] = df_prepared['capital.loss'].apply(lambda x: 1 if x > 0 else 0)
    df_prepared['capital.loss=0'] = df_prepared['capital.loss'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='capital.loss', inplace=True)

    df_prepared['male'] = df_prepared['sex'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['female'] = df_prepared['sex'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='sex', inplace=True) 

    # Process 'race' column
    df_prepared['white'] = df_prepared['race'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['n_white'] = df_prepared['race'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='race', inplace=True)

    # Process 'income' column
    df_prepared['>50K'] = df_prepared['income'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['<50K'] = df_prepared['income'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='income', inplace=True)

    # 1 if married, 0 otherwise
    df_prepared['married'] = df_prepared['marital.status'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['n_married'] = df_prepared['marital.status'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='marital.status', inplace=True)

    # United-States is 1 all the other countries are 0
    df_prepared['UnitedStates'] = df_prepared['native.country'].apply(lambda x: 1 if x == 1 else 0)
    df_prepared['n_UnitedStates'] = df_prepared['native.country'].apply(lambda x: 1 if x == 0 else 0)
    df_prepared.drop(columns='native.country', inplace=True)
    
    df_prepared.drop(columns=['capital.loss>0', 'capital.loss=0', 'capital.gain>0', 'capital.gain=0'], inplace=True)
    df_prepared.drop(columns=['hours.per.week'], inplace=True)

    df_prepared = pd.get_dummies(df_prepared, columns=['education.num','Age (decade)'],drop_first=False) 
    return df_prepared

def group_edu(x):
    if x <= 5:
        return '<6'
    elif x >= 13:
        return '>12'
    else:
        return x
    