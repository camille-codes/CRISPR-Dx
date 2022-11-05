import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import math

from sklearn.tree import DecisionTreeClassifier

import equalSplitting
from sklearn.metrics import classification_report, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
import math
import equalSplitting
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import time


def hyperparameter_tuning_DTC_merged(X_train, X_test, y_train, y_test, dataname):

    model_dtc = tree.DecisionTreeClassifier()

    param_grid = {
        'max_features': range(1, 20),
        'max_depth': range(5, 40, 5),
        'min_samples_split': range(10, 500, 10),
        'criterion': ['gini'],
    }

    CV_dtc = GridSearchCV(estimator=model_dtc, param_grid=param_grid, cv=10, n_jobs=-1, verbose=3, scoring=['recall','precision','f1','roc_auc','accuracy'], refit='recall')
    CV_dtc.fit(X_train, y_train)
    print(CV_dtc.best_params_)

    df_hyperparameters = pd.DataFrame(data={"Parameters": CV_dtc.cv_results_['params'], "mean_test_recall": CV_dtc.cv_results_['mean_test_recall'], \
                                            "mean_test_precision": CV_dtc.cv_results_['mean_test_precision'], "mean_test_f1": CV_dtc.cv_results_['mean_test_f1'],\
                                            "mean_test_roc_auc": CV_dtc.cv_results_['mean_test_roc_auc'], "mean_test_accuracy": CV_dtc.cv_results_['mean_test_accuracy']})
    df_hyperparameters.to_csv(dataname+" Decision Tree Classification hyperparameters with scores.csv", sep=',',index=False)


def auROC_plot(y_test, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'DTR AUC = %0.2f' % roc_auc) # blue=cpf1, orange=adapt, green=dc
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def Decision_Tree_Model_merged(model, filename,X_train, X_test, y_train, y_test):

    start_time = time.time()
    predictions_cv = model.predict(X_test)
    print("--- %s seconds ---" % (time.time() - start_time))

    dict_unscaled_dtc_cv = {'Predicted Activity': predictions_cv, 'True Activity': y_test}
    df_predicted_rf_cv = pd.DataFrame(dict_unscaled_dtc_cv)

    # get aurROC plot
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # AUROC SCORE
    y_pred_proba = model.predict_proba(X_test)[::, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

    list_fpr = list(fpr)
    list_tpr = list(tpr)
    df_scores_auc = pd.DataFrame({'FPR': list_fpr, 'TPR': list_tpr})

    df_scores_auc.to_csv(filename+"_fpr_tpr.csv", sep=',')

    auROC_plot(y_test, y_pred_proba)

    # PRECISION-RECALL SCORE
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    aupr = auc(recall, precision)
    list_pr = list(precision)
    list_rc = list(recall)
    df_scores_rc = pd.DataFrame({'Precision': list_pr, 'Recall': list_rc})
    df_scores_rc.to_csv(filename+"_pr_rc.csv", sep=',')

    # create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    # add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()

    print(roc_auc)
    auROC_plot(y_test, y_pred_proba)

    # report metrics
    report = classification_report(y_test, predictions_cv, output_dict=True)
    m_corr = matthews_corrcoef(y_test, predictions_cv, sample_weight=None)
    df_report = pd.DataFrame(report).transpose()

    # save confusion matrix
    df_confusion_matrix = pd.DataFrame(confusion_matrix(predictions_cv, y_test))
    df_confusion_matrix.to_csv(filename + "_confusion matrix.csv", sep=',')

    print(df_report)
    print("MCorr: ", m_corr)
    print("aupr: ", aupr)
    print("auroc: ", roc_auc)

    #output report
    df_report.to_csv(filename+"_report.csv", sep=',')

    # save the model to disk via joblib
    filename_jl = 'DT.sav'
    joblib.dump(model, filename_jl)


# import data
df_train = pd.read_csv('train_and_test_data/training_data_w_cas.csv')
df_holdout = pd.read_csv('train_and_test_data/hold_out_test_w_cas.csv')

X_train = df_train.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_train = df_train['label']

X_test = df_holdout.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_test = df_holdout['label']


#hyperparameter_tuning_DTC_merged(X_train, X_test, y_train, y_test, "Merged_w_pos")
# {'criterion': 'gini', 'max_depth': 25, 'max_features': 19, 'min_samples_split': 80}
#
# model_file = 'DT.sav'
# model = joblib.load(model_file)
# # Decision_Tree_Model_merged(model,"DT", X_train, X_test, y_train, y_test)
#
# # RFECV
# from sklearn.feature_selection import RFECV
#
# selector = RFECV(model, step=1, cv=5, n_jobs=-1, verbose=3, min_features_to_select=19)
# selector = selector.fit(X_train, y_train)
#
# features = list(X_train.columns)
#
# print(features)
#
# features_rank = pd.DataFrame({'Features': features, 'Rank': selector.ranking_})
#features_rank.to_csv("DT_rfe.csv", sep=',')
features_rank = pd.read_csv('DT_rfe.csv')


# extract top 5 features
top_ft = features_rank[features_rank['Rank'] <= 5]['Features'].to_list()
# features_rank.to_csv("DT_rfe.csv", sep=',')
# features_rank = pd.read_csv('DT_rfe.csv')

X_train2 = X_train[top_ft]
X_test2 = X_test[top_ft]
#
# # Refit on features with rank 1-5
# # {'criterion': 'gini', 'max_depth': 25, 'max_features': 19, 'min_samples_split': 80}
# model2 = DecisionTreeClassifier(criterion='gini', max_depth=25, max_features=19, min_samples_split=80, random_state=0)
# DT_rfe = model2.fit(X_train2, y_train)
#
# filename_jl = 'DT_rfe.sav'
# joblib.dump(DT_rfe, filename_jl)


model_file = 'DT_rfe.sav'
model = joblib.load(model_file)

Decision_Tree_Model_merged(model,"DT_rfe", X_train2, X_test2, y_train, y_test)
