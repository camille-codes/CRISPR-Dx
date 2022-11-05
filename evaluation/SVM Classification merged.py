import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import math

# # import datasets with precomputed on-target features
# df_cpf1 = pd.read_csv('cpf1_max_normalized.csv', usecols=['guide_seq', 'Activity', 'GC_content', 'tandem_repeats','entropy', 'melting_temperature', 'contiguous_repetitiveness'])
# df_adapt = pd.read_csv('adapt_max_normalized.csv', usecols=['guide_seq', 'Activity', 'GC_content', 'tandem_repeats','entropy', 'melting_temperature', 'contiguous_repetitiveness']) # zero hamming distance
# df_dc = pd.read_csv('dc_max_normalized.csv', usecols=['guide_seq', 'Activity', 'GC_content', 'tandem_repeats','entropy', 'melting_temperature', 'contiguous_repetitiveness'])
#
# data1 = df_cpf1
# data2 = df_adapt
# data3 = df_dc
#
# df_train = (equalSplitting.equalSplit(data1, data2, data3)[1])
# df_holdout = (equalSplitting.equalSplit(data1, data2, data3)[0])


def hyperparameter_tuning_SVM_merged(X_train, X_test, y_train, y_test, dataname):

    model_SVM = SVC()

    param_grid = {'C': [0.1, 1, 10, 100, 1000], # low C overfits, high C underfits. C adds penalty for miscassified datapoint
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], # for rbf - how much curvature we want
                  #'kernel': ["sigmoid"],
                  #'kernel': ["poly"],
                  'kernel': ["rbf"],
                  #'kernel': ["linear","poly", "rbf", "sigmoid", "precomputed"],
                  #'degree': range(1,5)
                  }

    CV_svm = GridSearchCV(estimator=model_SVM, param_grid=param_grid, cv= 5, n_jobs=-1, verbose=3, scoring=['recall','precision','f1','roc_auc','accuracy'], refit='recall')
    CV_svm.fit(X_train, y_train)
    print(CV_svm.best_params_)

    # outputs the best parameters
    #df_hyperparameters = pd.DataFrame(data={"Parameters": CV_svm.cv_results_['params'], "Mean_test_score_CV": CV_svm.cv_results_['mean_test_score'], "Rank": CV_svm.cv_results_['rank_test_score']})

    df_hyperparameters = pd.DataFrame(
        data={"Parameters": CV_svm.cv_results_['params'], "mean_test_recall": CV_svm.cv_results_['mean_test_recall'], \
              "mean_test_precision": CV_svm.cv_results_['mean_test_precision'],
              "mean_test_f1": CV_svm.cv_results_['mean_test_f1'], \
              "mean_test_roc_auc": CV_svm.cv_results_['mean_test_roc_auc'],
              "mean_test_accuracy": CV_svm.cv_results_['mean_test_accuracy']})

    df_hyperparameters.to_csv(dataname+" SVM Classification hyperparameters with scores.csv", sep=',',index=False)



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

def SVM_Model_merged(model, filename, X_train, X_test, y_train, y_test):
    # {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
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


# import data
df_train = pd.read_csv('train_and_test_data/training_data_w_cas.csv')
df_holdout = pd.read_csv('train_and_test_data/hold_out_test_w_cas.csv')

X_train = df_train.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_train = df_train['label']

X_test = df_holdout.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_test = df_holdout['label']

model_file = 'SVM.sav'
model = joblib.load(model_file)

# RFECV
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_train, y_train)

feature_names = list(X_train.columns)
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()

print(sorted_idx)

plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()


# selector = RFE(model, step=3, verbose=3)
# selector = selector.fit(X_train, y_train)
#
# features = list(X_train.columns)
#
# print(features)
#
# features_rank = pd.DataFrame({'Features': features, 'Rank': selector.ranking_})
# features_rank.to_csv("SVM_rfe.csv", sep=',')
# features_rank = pd.read_csv('SVM_rfe.csv')
#
#
# # # extract top 5 features
# top_ft = features_rank[features_rank['Rank'] <= 5]['Features'].to_list()
# #
# X_train2 = X_train[top_ft]
# X_test2 = X_test[top_ft]
# #
# # Refit on features with rank 1-5
# #{'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
# model2 = SVC(C=1000, gamma=0.001, kernel='rbf')
# model2 = model2.fit(X_train2, y_train)
#
# model_file = 'SVM_rfe.sav'
# joblib.dump(model2, model_file)


#hyperparameter_tuning_SVM_merged(X_train, X_test, y_train, y_test, "Merged rbf w cas")

# SVM_Model_merged(df_train, df_holdout, gamma=1, c=1000, kernel='rbf', dataname="Merged")
#SVM_Model_merged(df_train, df_holdout, gamma=1, c=1000, kernel='poly', degree=3, dataname="Merged")
# SVM_Model_merged(df_train, df_holdout, gamma=0.1, c=10, kernel='sigmoid', dataname="Merged")
# SVM_Model_merged(df_train, df_holdout, gamma='auto', kernel='linear', dataname="Merged")
