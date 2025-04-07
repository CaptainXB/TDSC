import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import math
import warnings
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from savedata import savedata
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=245)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)


# 核函数选择


def fit_predict(clf, Xs, Ys, Xt, Yt):
    '''
    Transform Xs and Xt, then make predictions on target using 1NN
    :param Xs: ns * n_feature, source feature
    :param Ys: ns * 1, source label
    :param Xt: nt * n_feature, target feature
    :param Yt: nt * 1, target label
    :return: Accuracy and predicted_labels on the target domain
    '''


    clf.fit(Xs, Ys.ravel())
    y_pred = clf.predict(Xt)

    # evaluation metrics
    TN = 0
    FN = 0
    TP = 0
    FP = 0

    for i in range(np.shape(Xt)[0]):
        if Yt[i] == 0:
            if y_pred[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if y_pred[i] == 0:
                FN = FN + 1
            else:
                TP = TP + 1

    PD = TP / (TP + FN)
    PF = FP / (FP + TN)
    Bal = 1 - math.sqrt(((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    AUC = metrics.roc_auc_score(Yt, y_pred)
    return PD, PF, Bal, AUC


if __name__ == '__main__':
    source = ['linux.csv', 'mysql.csv', 'netbsd.csv']
    target = ['linux.csv', 'mysql.csv', 'netbsd.csv']
    down = 0
    for source_name in source:
        left = 0
        for target_name in target:
            if source_name == target_name:
                pass
            else:
                source_data = np.loadtxt('F:\\dataset\\data\\' + source_name, delimiter=',', skiprows=1)
                target_data = np.loadtxt('F:\\dataset\\data\\' + target_name, delimiter=',', skiprows=1)

                source_x = np.delete(source_data, -1, axis=1)
                source_y = source_data[:, 82]
                target_x = np.delete(target_data, -1, axis=1)
                target_y = target_data[:, 82]

                # standard = StandardScaler()
                #
                # source_x = standard.fit_transform(source_x)
                # target_x = standard.fit_transform(target_x)

                train_model = [
                    KNeighborsClassifier(n_neighbors=25),
                    GaussianNB(),
                    LogisticRegression(C=1),
                    SVC(gamma='auto', kernel='linear'),
                    DecisionTreeClassifier(criterion='entropy', max_depth=1, min_samples_split=3, random_state=1),
                    RandomForestClassifier(n_estimators=5, max_depth=1, random_state=1)
                ]
                index = 0
                for model in train_model:
                    PD1 = []
                    PF1 = []
                    Bal1 = []
                    AUC1 = []
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        ros = RandomOverSampler(random_state=1)
                        Xs_new, Ys = ros.fit_resample(Xs, Ys)
                        PD, PF, Bal, AUC = fit_predict(model, Xs_new, Ys, Xt, Yt)

                        PD1.append(PD)
                        PF1.append(PF)
                        Bal1.append(Bal)
                        AUC1.append(AUC)

                    print(np.mean(PD1))
                    print(np.mean(PF1))
                    print(np.mean(Bal1))
                    print(np.mean(AUC1))

                    performance = [np.mean(PD1), np.mean(PF1), np.mean(AUC1), np.mean(Bal1)]
                    if left == 0:
                        col = 1
                    else:
                        col = 12
                    savedata('F:\\paper result\\dimension and transfer method compare.xls', 3 + down * 9, index + col,
                             performance)
                    index += 1
                left += 1
        down += 1
