import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_recall_curve, auc, classification_report, matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
import math
from sklearn.utils import shuffle
import joblib

# import data
df_train = pd.read_csv('train_and_test_data/training_data_w_cas.csv')
df_holdout = pd.read_csv('train_and_test_data/hold_out_test_w_cas.csv')

X_train = df_train.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_train = df_train['label']
X_test = df_holdout.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_test = df_holdout['label']

df_shuffled_train = shuffle(df_train, random_state=42)
df_train_60 = df_shuffled_train.head(math.ceil(len(df_shuffled_train)*0.60))
df_val_40 = df_shuffled_train.tail(math.ceil(len(df_shuffled_train)*0.40)-1)

# separate X from y
X_train_60 = df_train_60.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_train_60 = df_train_60['label']
X_val_40 = df_val_40.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_val_40 = df_val_40['label']

evaluation = [( X_train_60, y_train_60), ( X_val_40, y_val_40)]

# model_xgb = XGBClassifier(objective='binary:logistic', learning_rate = 0.03,
#               max_depth = 40, n_estimators = 350, gamma=0, subsample=0.6, colsample_bytree=0.6)
#
# model_xgb.fit(X_train_60, y_train_60, eval_set=evaluation, eval_metric="auc")

def auROC_plot(y_test, predictions_cv):
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions_cv)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'SVM AUC = %0.2f' % roc_auc) # blue=cpf1, orange=adapt, green=dc
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def XGBoost_merged(model, filename, X_train, X_test, y_train, y_test):
    start_time = time.time()
    predictions_cv = model.predict(X_test)
    print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # predictions_cv = model.predict(X_test)
    # print("--- %s seconds ---" % (time.time() - start_time))

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


# model_file = 'XGBoost_model2.sav'
# model = joblib.load(model_file)

#XGBoost_merged(model,"XGB", X_train, X_test, y_train, y_test)

# RFECV
# from sklearn.feature_selection import RFE
#
# selector = RFE(model, step=3, verbose=3)
# selector = selector.fit(X_train, y_train)
#
# features = list(X_train.columns)
#
# print(features)

# features_rank = pd.DataFrame({'Features': features, 'Rank': selector.ranking_})
# features_rank.to_csv("XGB_rfe.csv", sep=',')
# features_rank = pd.read_csv('XGB_rfe.csv')
#
#
# # # extract top 5 features
# top_ft = features_rank[features_rank['Rank'] <= 5]['Features'].to_list()
#
# X_train2 = X_train[top_ft]
# X_test2 = X_test[top_ft]

# Refit on features with rank 1-5
# model2 = XGBClassifier(objective='binary:logistic', learning_rate = 0.03,
#               max_depth = 40, n_estimators = 350, gamma=0, subsample=0.6, colsample_bytree=0.6)
# model2 = model2.fit(X_train2, y_train)
#
# model_file = 'XGB_rfe.sav'
# joblib.dump(model2, model_file)

from xgboost import plot_tree

model_file = 'XGBoost_model2.sav'

model = joblib.load(model_file)


# model.get_booster().get_dump()
# model.get_booster().dump_model("out.txt")

plot_tree(model, num_trees=349)
plt.show()

# scores = model.get_booster().get_score(importance_type="gain")
#
# keys = list(scores.keys())
# values = list(scores.values())
#
# data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score")
# data.nlargest(50, columns="score").plot(kind='barh')
# plt.xlabel('Gain score')
# plt.ylabel('Features')
# plt.title('Feature Importance from XGBoost model')
# plt.grid(alpha=0.5)
# plt.show()
