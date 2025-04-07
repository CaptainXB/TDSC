"""
In this python script we provided an example of how to use our
implementation of ensemble methods to perform classification.

Usage:
```
python run_example.py --method=SPEnsemble --n_estimators=10 --runs=10
```
or with shortopts:
```
python run_example.py -m SPEnsemble -n 10 -r 10
```

run arguments:
    -m / --methods: string
    |   Specify which method were used to build the ensemble classifier.
    |   support: 'SPEnsemble', 'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'Cascade'
    -n / --n_estimators: integer
    |   Specify how much base estimators were used in the ensemble.
    -r / --runs: integer
    |   Specify the number of independent runs (to obtain mean and std)

"""
import scipy.io
import scipy.linalg
import sklearn.metrics
from numpy import linalg
import math
import numpy as np
import scipy.io
import time
import scipy.linalg
import sklearn.metrics
from imblearn.under_sampling import RandomUnderSampler
import math
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm
import warnings

from xlutils.copy import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
import os
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=245)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)
from numpy import linalg
from scipy import linalg
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
import scipy.io
import scipy.linalg
import sklearn.metrics
import scipy.io
import scipy.linalg
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from self_paced_ensemble_revise import SelfPacedEnsemble
from utils import *
from savedata import savedata


def SPEtest(Xs, Ys, Xt, Yt, model):
    y_test = Yt
    spe = SelfPacedEnsemble(base_estimator=model, random_state=42)
    spe.fit(Xs, Ys)
    y_pred = spe.predict_proba(Xt)[:, 1]

    pdx, pf, bal = bal_optim(y_test, y_pred)
    AUC = auc_prc(y_test, y_pred)
    y_pred_b = y_pred.copy()
    y_pred_b[y_pred_b < 0.5] = 0
    y_pred_b[y_pred_b >= 0.5] = 1
    return y_pred_b, AUC, pdx, pf, bal


class IDSA:
    def __init__(self, model=LogisticRegression()):
        self.model = model

    # 方差计算（协方差）
    def _cov(self, X):
        """Estimate covariance matrix.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        s : array, shape (n_features, n_features)
            Estimated covariance matrix.
        """
        s = np.cov(X, rowvar=0, bias=1)
        return s

    def isda(self, Xs, Ys, Xt, Yt):
        '''ISDA'''
        H1 = 1

        TN = 0
        FN = 0
        TP = 0
        FP = 0

        X1_train = []
        X2_train = []
        Ys = Ys.reshape((Ys.shape[0], 1))
        mm = np.hstack((Xs, Ys))

        # 将train_x按照0，1分类为X1（defective），X2（free）
        for i in range(np.shape(mm)[0]):
            if mm[i][-1] == 1:
                X1_train.append(mm[i, :])
            else:
                X2_train.append(mm[i, :])

        # 删除最后一列，得到完整的X1，X2
        XX = np.vstack((X1_train, X2_train))
        XX_y = XX[:, -1]
        X1 = np.delete(X1_train, -1, axis=1)
        X2 = np.delete(X2_train, -1, axis=1)
        X = np.vstack((X1, X2))

        '''不加ISDA'''
        # H2 = np.round(np.shape(X2)[0] / (np.shape(X1)[0] / H1)).astype(int)
        #
        # # 利用Kmeans将X1，X2分为H1，H2个子类
        # estimator_x1 = KMeans(n_clusters=H1)
        # estimator_x2 = KMeans(n_clusters=H2)
        # estimator_x1.fit(X1)
        # estimator_x2.fit(X2)
        # label_pred_x1 = estimator_x1.labels_
        # label_pred_x2 = estimator_x2.labels_
        # x1 = []
        # x2 = []
        # for i in range(H1):
        #     x1.append(X1[label_pred_x1 == i])
        # for i in range(H2):
        #     x2.append(X2[label_pred_x2 == i])
        #
        # # 求解类间离散矩阵，数据的协方差，及最终求解投影向量
        #
        # B = np.zeros((np.shape(X1)[1], np.shape(X1)[1]))
        # n = np.shape(X)[0]
        #
        # # 类间散度
        # for j in range(1, H1 + 1):
        #     for l in range(1, H2 + 1):
        #         pij = np.shape(x1[j - 1])[0] / n
        #         pkl = np.shape(x2[l - 1])[0] / n
        #         uij = x1[j - 1].mean(0)
        #         ukl = x2[l - 1].mean(0)
        #         a = uij - ukl
        #         b = a.reshape((np.shape(X1)[1], 1))
        #         B = B + pij * pkl * (a * b)
        #
        # # 协方差矩阵
        # XM = self._cov(X)
        # XM = np.asarray(XM)
        #
        # # 计算特征向量
        # evals, evecs = linalg.eigh(B / XM)
        # evecs = evecs[:, np.argsort(evals)[::-1]]
        #
        # Xs_new = np.dot(X, evecs)
        # Yt_new = np.dot(Xt, evecs)

        # print(Xs_new.shape)
        # print(Yt_new.shape)
        # print(Xs.shape)
        # print(Xt.shape)

        # '''不加SPE的LR预测，x_train:Xs_new, y_train:XX_y, x_test:Yt_new, y_test:Yt'''
        # # 使用逻辑回归进行分类预测
        # lr_model = LogisticRegression()
        # lr_model.fit(Xs_new, XX_y.ravel())  # 使用源域数据和标签进行训练
        # y_pred = lr_model.predict(Yt_new)  # 对目标域数据进行预测
        #
        # pdx, pf, bal = bal_optim(Yt, y_pred)
        # AUC = auc_prc(Yt, y_pred)
        # y_pred_b = y_pred.copy()
        # y_pred_b[y_pred_b < 0.5] = 0
        # y_pred_b[y_pred_b >= 0.5] = 1
        #
        # if TP + FN == 0:
        #     PD = 0
        # else:
        #     PD = TP / (TP + FN)
        #
        # if (FP + TN) == 0:
        #     PF = 0
        # else:
        #     PF = FP / (FP + TN)
        # Bal = 1 - math.sqrt(((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)

       # '''SPE'''
       #  y_pred, AUC, pdx, pf, bal = SPEtest(Xs_new, XX_y, Yt_new, Yt, self.model)
        '''不加ISDA，BDA+SPE， x_train:Xs, y_train:Ys, x_test:Xt, y_test:Yt'''
        y_pred, AUC, pdx, pf, bal = SPEtest(X, XX_y, Xt, Yt, self.model)
        '''计算最后结果'''
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
        return PD, PF, Bal, AUC, pdx, pf, bal, y_pred


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                         np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(0, C):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu


class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0., gamma=1, T=10, mode='BDA', estimate_mu=False,
                 model=LogisticRegression()):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu
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
        '''BDA的处理方法'''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt and len(
                    Y_tar_pseudo[np.where(Y_tar_pseudo == 1)]) != 0 and len(
                Y_tar_pseudo[np.where(Y_tar_pseudo == 0)]) != 0:
                for c in range(2):
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1

                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot(
                [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            '''ISDA'''
            dsa = IDSA(model=self.model)
            PD, PF, Bal, AUC, pdx, pf, bal, y_pred = dsa.isda(Xs_new, Ys, Xt_new, Yt)

            auc = metrics.roc_auc_score(Yt, y_pred)
            Y_tar_pseudo = y_pred

            print('BDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            print('BDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            print('BDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            print('BDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
            print('BDA iteration [{}/{}]: Pd: {:.4f}'.format(t + 1, self.T, pdx))
            print('BDA iteration [{}/{}]: pf: {:.4f}'.format(t + 1, self.T, pf))
            print('BDA iteration [{}/{}]: bal_gailv: {:.4f}'.format(t + 1, self.T, bal))
        return PD, PF, Bal, AUC, pdx, pf, bal, auc


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, model=LogisticRegression()):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.model = model

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)

        dsa = IDSA(model=self.model)
        PD, PF, Bal, AUC, pdx, pf, bal, y_pred = dsa.isda(Xs_new, Ys, Xt_new, Yt)

        Y_tar_pseudo = y_pred

        auc = metrics.roc_auc_score(Yt, y_pred)

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

        return PD, PF, Bal, AUC, pdx, pf, bal, auc

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
                source_data = np.loadtxt('/Users/orange/Documents/code/徐斌TDSC/data/' + source_name, delimiter=',', skiprows=1)
                target_data = np.loadtxt('/Users/orange/Documents/code/徐斌TDSC/data/' + target_name, delimiter=',', skiprows=1)

                source_x = np.delete(source_data, -1, axis=1)
                source_y = source_data[:, 82]
                target_x = np.delete(target_data, -1, axis=1)
                target_y = target_data[:, 82]

                stand = StandardScaler()
                source_x = stand.fit_transform(source_x)
                target_x = stand.fit_transform(target_x)

                train_model = [
                    KNeighborsClassifier(n_neighbors=20),
                    GaussianNB(),
                    LogisticRegression(C=1),
                    SVC(probability=True, gamma='auto'),
                    DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=3, random_state=1),
                    RandomForestClassifier(n_estimators=10, max_depth=1, random_state=1)
                ]
                index = 0
                for model in train_model:
                    PD1 = []
                    PF1 = []
                    Bal1 = []
                    AUC1 = []
                    pd1 = []
                    pf1 = []
                    bal1 = []
                    auc1 = []
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        # BDA+ISDA+SPE
                        # bda = BDA(kernel_type='primal', dim=5, lamb=1, mu=0.1, mode='BDA', gamma=1, estimate_mu=False,
                        #           model=model)
                        # PD, PF, Bal, AUC, pdx, pf, bal, auc = bda.fit_predict(Xs, Ys, Xt, Yt)
                        '''TCA+ISDA+SPE'''
                        tca = TCA(kernel_type='primal', dim=5, lamb=1, gamma=1, model=model)
                        PD, PF, Bal, AUC, pdx, pf, bal, auc = tca.fit_predict(Xs, Ys, Xt, Yt)

                        PD1.append(PD)
                        PF1.append(PF)
                        Bal1.append(Bal)
                        AUC1.append(AUC)
                        bal1.append(bal)
                        pd1.append(pdx)
                        pf1.append(pf)
                        auc1.append(auc)  #小的

                    print('PD', np.mean(PD1))
                    print('PF', np.mean(PF1))
                    print('bal', np.mean(Bal1))
                    print('auc', np.mean(AUC1))
                    print('pdx', np.mean(pd1))
                    print('pf', np.mean(pf1))
                    print('bal1', np.mean(bal1))
                    print('auc', np.mean(auc1))

                    performance = [np.mean(pd1), np.mean(pf1), np.mean(auc1), np.mean(Bal1)]
                    if left == 0:
                        col = 1
                    else:
                        col = 12
                    '''BDA+ISDA =7, BDA+SPE=8, IPSE=6'''
                    savedata('/Users/orange/Documents/code/徐斌TDSC/result/model result finally.xls', 8 + down * 9, index + col, performance)
                    index += 1
                left += 1
        down += 1
