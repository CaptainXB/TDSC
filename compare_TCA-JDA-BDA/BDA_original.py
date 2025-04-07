# encoding=utf-8
"""
    Created on 9:52 2018/11/14
    @author: Jindong Wang
"""
import xlrd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm
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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from savedata import savedata
from sklearn.decomposition import KernelPCA


def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return roc_auc_score(label, y_pred)


def bal_optim(label, y_pred):
    pds = []
    pfs = []
    bals = []
    y_pred_b = y_pred.copy()

    for t in range(100):
        TN = 0
        FN = 0
        TP = 0
        FP = 0
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
        pds.append(PD)
        pfs.append(PF)
        bals.append(Bal)
    for i in range(len(bals)):
        if bals[i] == max(bals):
            m = i
    return pds[m], pfs[m], max(bals)


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
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu=False,
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
        global pdx
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))  # (Xs: Xt)
        X /= np.linalg.norm(X, axis=0)  # axis=0按列 求向量的范式
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))  # 数组去重
        H = np.eye(n) - 1 / n * np.ones((n, n))  # np.eye生成对角阵
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
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
            # print(mu)
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot(
                [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            # print(w.shape)
            A = V[:, ind[:self.dim]]
            # print(A.shape)
            # print(K.shape)
            Z = np.dot(A.T, K)

            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            smo = RandomOverSampler(random_state=1)
            x_resampled, y_resampled = smo.fit_resample(Xs_new, Ys)
            clf = self.model
            clf.fit(x_resampled, y_resampled)
            Y_tar_pseudo = clf.predict(Xt_new)

            AUC = roc_auc_score(Yt, Y_tar_pseudo)

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
        return PD, PF, Bal, AUC


if __name__ == '__main__':
    # 'linux.csv', 'mysql.csv', 'netbsd.csv'
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
                        bda = BDA(kernel_type='primal', dim=5, lamb=1, mu=0.1, mode='BDA', gamma=1, estimate_mu=True,
                                  model=model)
                        PD, PF, Bal, AUC = bda.fit_predict(Xs, Ys, Xt, Yt)

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
                    savedata('F:\\paper result\\transfer learning compare_new.xls', 6 + down * 9,
                             index + col, performance)
                    index += 1
                left += 1
        down += 1
