import math

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from savedata import savedata
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

                train_model = [
                    sklearn.neighbors.KNeighborsClassifier(n_neighbors=20),
                    GaussianNB(),
                    LogisticRegression(C=1),
                    SVC(probability=True, gamma='auto'),
                    sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=3,
                                                        random_state=2),
                    RandomForestClassifier(n_estimators=10, max_depth=1, random_state=1)
                ]
                index = 0
                for model in train_model:
                    PD1 = []
                    PF1 = []
                    Bal1 = []
                    AUC1 = []
                    mcc1 = []
                    f11 = []
                    gm1 = []
                    bal1 = []
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        smo = SMOTE(random_state=1)
                        Xs_balance, Ys_balance = smo.fit_resample(Xs, Ys)
                        model.fit(Xs_balance, Ys_balance)
                        Y_tar_pseudo = model.predict(Xt, Yt)

                        AUC = roc_auc_score(Yt, Y_tar_pseudo)

                        # print(mcc)
                        TP = 0
                        FN = 0
                        FP = 0
                        TN = 0
                        y_pred = Y_tar_pseudo
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
                    savedata('F:\\comparative experiment\\result.xls', 3 + down * 9, index + col, performance)
                    index += 1
                left += 1
        down += 1