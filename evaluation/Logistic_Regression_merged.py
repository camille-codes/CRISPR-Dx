import time

from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report, matthews_corrcoef, confusion_matrix, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import joblib


def hyperparameter_tuning_LR_merged(X_train, X_test, y_train, y_test, dataname):
    model_LR = LogisticRegression()

    # define grid search
    param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                  'penalty': ['l2', 'l1'],
                  'C': [100, 10, 1.0, 0.1, 0.01],
                  }

    CV_LR = GridSearchCV(estimator=model_LR, param_grid=param_grid, cv= 10, n_jobs=-1, verbose=3, scoring=['recall','precision','f1','roc_auc','accuracy'], refit='recall')
    CV_LR.fit(X_train, y_train)
    print(CV_LR.best_params_)

    df_hyperparameters = pd.DataFrame(
        data={"Parameters": CV_LR.cv_results_['params'], "mean_test_recall": CV_LR.cv_results_['mean_test_recall'], \
              "mean_test_precision": CV_LR.cv_results_['mean_test_precision'],
              "mean_test_f1": CV_LR.cv_results_['mean_test_f1'], \
              "mean_test_roc_auc": CV_LR.cv_results_['mean_test_roc_auc'],
              "mean_test_accuracy": CV_LR.cv_results_['mean_test_accuracy']})

    df_hyperparameters.to_csv(dataname+" Logistic Regression Classification hyperparameters with scores.csv", sep=',',index=False)

def auROC_plot(y_test, y_pred_proba):
    # fpr, tpr, threshold = metrics.roc_curve(y_test, predictions_cv)
    # roc_auc = metrics.auc(fpr, tpr)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'Logistic Regression AUC = %0.2f' % roc_auc) # blue=cpf1, orange=adapt, green=dc
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def Logistic_Regression_merged(model, filename, X_train, X_test, y_train, y_test):
    #{'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}

    # model_LR = LogisticRegression(C=100, penalty='l2', solver='lbfgs', random_state=0, n_jobs=-1)
    # model_LR.fit(X_train, y_train)

    start_time = time.time()
    predictions_cv = model.predict(X_test)
    print("--- %s seconds ---" % (time.time() - start_time))

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

    # classification report
    # report metrics
    report = classification_report(y_test, predictions_cv, output_dict=True)
    m_corr = matthews_corrcoef(y_test, predictions_cv, sample_weight=None)
    df_report = pd.DataFrame(report).transpose()

    # save confusion matrix
    df_confusion_matrix = pd.DataFrame(confusion_matrix(predictions_cv, y_test))
    df_confusion_matrix.to_csv(filename+"_confusion matrix.csv", sep=',')

    print(df_report)
    print("MCorr: ", m_corr)
    print("aupr: ", aupr)
    print("auroc: ", roc_auc)

    #output report
    df_report.to_csv(filename+"_report.csv", sep=',')

    # save the model to disk via joblib
    # filename_jl = 'LR.sav'
    # joblib.dump(model_LR, filename_jl)


# import data
df_train = pd.read_csv('train_and_test_data/training_data_w_cas.csv')
df_holdout = pd.read_csv('train_and_test_data/hold_out_test_w_cas.csv')

X_train = df_train.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_train = df_train['label']

X_test = df_holdout.drop(["Activity", 'guide_seq', 'label'], axis=1)
y_test = df_holdout['label']

#hyperparameter_tuning_LR_merged(X_train, X_test, y_train, y_test, "Merged w cas")


model_file = 'LR.sav'
loaded_model = joblib.load(model_file)

features_rank = pd.read_csv('LR_rfe.csv')

# extract top 5 features
top_ft = features_rank[features_rank['Rank'] <= 5]['Features'].to_list()

X_train2 = X_train[top_ft]
X_test2 = X_test[top_ft]

Logistic_Regression_merged(loaded_model, "LR", X_train, X_test, y_train, y_test)

# RFECV
# from sklearn.feature_selection import RFECV
#
# model = 'saved_models/LR.sav'
# loaded_model = joblib.load(model)


# selector = RFECV(loaded_model, step=1, cv=5, n_jobs=-1, verbose=3, )
# selector = selector.fit(X_train, y_train)
#
# features = list(X_train.columns)
#
# features_rank = pd.DataFrame({'Features': features, 'Rank': selector.ranking_})
# features_rank.to_csv("LR_rfe.csv", sep=',')
features_rank = pd.read_csv('LR_rfe.csv')

# extract top 5 features
top_ft = features_rank[features_rank['Rank'] <= 5]['Features'].to_list()

X_train2 = X_train[top_ft]
X_test2 = X_test[top_ft]

# Refit on features with rank 1-5
# model_LR2 = LogisticRegression(C=100, penalty='l2', solver='lbfgs', random_state=0, n_jobs=-1)
# LR_rfe = model_LR2.fit(X_train2, y_train)
# filename_jl = 'LR_rfe.sav'
# joblib.dump(LR_rfe, filename_jl)

#
# model = 'LR_rfe.sav'
# loaded_model = joblib.load(model)
#
# import time
#
#
# rfe_pred = loaded_model.predict(X_test2)