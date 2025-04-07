# encoding=utf-8
"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import xlrd
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm
from imblearn.over_sampling import SMOTE
import math
import sklearn.metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from savedata import savedata
# 数据预处理
from xlutils.copy import copy



def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return roc_auc_score(label, y_pred)


def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    # print(prec)
    # print(reca)
    f1s = 2 * (prec * reca) / (prec + reca)
    # print(f1s)
    return max(f1s)


def bal_optim(label, y_pred):
    # print(label)
    # print(y_pred)
    bals = []
    y_pred_b = y_pred.copy()
    # y_pred_b[y_pred_b < 0.6] = 0
    # y_pred_b[y_pred_b >= 0.6] = 1
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0 + t * 0.01] = 0
        y_pred_b[y_pred_b >= 0 + t * 0.01] = 1
        for i in range(len(label)):
            if label[i] == 0:
                if y_pred_b[i] == 0:
                    TN = TN + 1
                else:
                    FP = FP + 1
            else:
                if y_pred_b[i] == 0:
                    FN = FN + 1
                else:
                    TP = TP + 1

        PD = TP / (TP + FN)
        PF = FP / (FP + TN)
        Bal = 1 - math.sqrt(((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
        bals.append(Bal)
    return max(bals)


def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec * reca), 0.5)
    return max(gms)


def mcc_optim(label, y_pred):
    '''Compute optimal MCC score.'''
    mccs = []
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0 + t * 0.01] = 0
        y_pred_b[y_pred_b >= 0 + t * 0.01] = 1
        mcc = matthews_corrcoef(label, y_pred_b)
        mccs.append(mcc)
    return max(mccs)


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10, model=LogisticRegression()):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T
        self.model = model

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(2):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            model = RandomOverSampler(random_state=1)
            # model = RandomOverSampler(random_state=1)
            x_resampled, y_resampled = model.fit_resample(Xs_new, Ys)

            clf = self.model
            clf.fit(x_resampled, y_resampled)
            Y_tar_pseudo = clf.predict(Xt_new)
            # acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            AUC = auc_prc(Yt, Y_tar_pseudo)
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
            # list_acc.append(acc)
            # print('BDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            # print('BDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            # print('BDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            # print('BDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))

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

                stand = StandardScaler()
                source_x = stand.fit_transform(source_x)
                target_x = stand.fit_transform(target_x)



                train_model = [
                    KNeighborsClassifier(n_neighbors=25),
                    GaussianNB(),
                    LogisticRegression(C=1),
                    SVC(probability=True, gamma='auto', kernel='linear'),
                    DecisionTreeClassifier(criterion='entropy', max_depth=1, min_samples_split=3, random_state=1),
                    RandomForestClassifier(n_estimators=5, max_depth=1, random_state=1)
                ]
                index = 0
                for model in train_model:
                    PD1 = []
                    PF1 = []
                    Bal1 = []
                    AUC1 = []
                    bal1 = []
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        jda = JDA(kernel_type='primal', dim=5, lamb=1, gamma=1, model=model)
                        PD, PF, Bal, AUC = jda.fit_predict(Xs, Ys, Xt, Yt)

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
                    savedata('F:\\paper result\\transfer learning compare_new.xls', 5 + down * 9, index + col, performance)
                    index += 1
                left += 1
        down += 1