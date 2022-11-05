import pandas as pd
import joblib
from sklearn import metrics


def predict_ML(df_common):

    X_common = df_common.drop(['guide_seq'], axis=1)

    # load model
    model = 'XGBoost_model2.sav'
    loaded_model = joblib.load(model)

    df_common['label'] = loaded_model.predict(X_common)

    return df_common




