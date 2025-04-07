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
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 数据预处理
from xlutils.copy import copy
from sklearn.decomposition import PCA
linux = np.loadtxt('E:\\class-overlap-master\\data\\aging data\\linux.csv', delimiter=',', skiprows=1)
linux_x = np.delete(linux, -1, axis=1)
linux_y = linux[:, 82]
# print(linux_y)

netbsd = np.loadtxt('E:\\class-overlap-master\\data\\aging data\\netbsd.csv', delimiter=',', skiprows=1)
netbsd_x = np.delete(netbsd, -1, axis=1)
netbsd_y = netbsd[:, 82]

i = 0
j = 0
linux_mean = np.mean(linux_x, axis=0)
linux_std = np.std(linux_x, axis=0)
netbsd_mean = np.mean(netbsd_x, axis=0)
netbsd_std = np.std(netbsd_x, axis=0)
linux_pre1 = linux_x
netbsd_pre1 = netbsd_x
pca = PCA(n_components=5)
linux_pre = pca.fit_transform(linux_pre1)
netbsd_pre = pca.fit_transform(netbsd_pre1)

for i in range(np.shape(linux_pre)[0]):
    for j in range(np.shape(linux_pre)[1]):
        if linux_std[j] != 0:
           linux_pre[i][j] = (linux_x[i][j] - linux_mean[j]) / linux_std[j]
        else:
            linux_pre[i][j] = linux_x[i][j] - linux_mean[j]
for i in range(np.shape(netbsd_pre)[0]):
    for j in range(np.shape(netbsd_pre)[1]):
        if netbsd_std[j] != 0:
           netbsd_pre[i][j] = (netbsd_x[i][j] - netbsd_mean[j]) / netbsd_std[j]
        else:
           netbsd_pre[i][j] = netbsd_x[i][j] - netbsd_mean[j]
def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return roc_auc_score(label, y_pred)

def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    #print(prec)
    #print(reca)
    f1s = 2 * (prec * reca) / (prec + reca)
    #print(f1s)
    return max(f1s)

def bal_optim(label, y_pred):
    #print(label)
    #print(y_pred)
    pds = []
    pfs = []
    bals = []
    y_pred_b = y_pred.copy()
    #y_pred_b[y_pred_b < 0.6] = 0
    #y_pred_b[y_pred_b >= 0.6] = 1

    for t in range(100):
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0+t*0.01] = 0
        y_pred_b[y_pred_b >= 0+t*0.01] = 1
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
    print(m)
    return pds[m],pfs[m], max(bals)

def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec*reca), 0.5)
    return max(gms)

def mcc_optim(label, y_pred):
    '''Compute optimal MCC score.'''
    mccs = []
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0+t*0.01] = 0
        y_pred_b[y_pred_b >= 0+t*0.01] = 1
        mcc = matthews_corrcoef(label, y_pred_b)
        mccs.append(mcc)
    return max(mccs)






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
    for i in range(1, C + 1):
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
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=1, mode='BDA', estimate_mu=False):
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

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''

        for t in range(self.T):
            model = SMOTE()
            x_resampled, y_resampled = model.fit_resample(Xs, Ys)

            # clf=sklearn.tree.DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=3, random_state=2)
            # clf=LogisticRegression(C=1)
            # clf=RandomForestClassifier(n_estimators=10,max_depth=1,random_state=1)
            # clf=GaussianNB()
            clf = SVC(probability=True, gamma='auto')
            # clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
            clf.fit(x_resampled, y_resampled)
            Y_tar_pseudo = clf.predict(Xt)
            #acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)

            AUC = auc_prc(Yt, Y_tar_pseudo)
            mcc = mcc_optim(Yt, Y_tar_pseudo)
            f1 = f1_optim(Yt, Y_tar_pseudo)
            gm = gm_optim(Yt, Y_tar_pseudo)
            pdx, pf, bal = bal_optim(Yt, Y_tar_pseudo)
            #print(mcc)
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
            #list_acc.append(acc)
            print('BDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            print('BDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            print('BDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            print('BDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
            print('BDA iteration [{}/{}]: mcc: {:.4f}'.format(t + 1, self.T, mcc))
            print('BDA iteration [{}/{}]: f1: {:.4f}'.format(t + 1, self.T, f1))
            print('BDA iteration [{}/{}]: gm: {:.4f}'.format(t + 1, self.T, gm))
            print('BDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, pdx))
            print('BDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, pf))
            print('BDA iteration [{}/{}]: bal_gailv: {:.4f}'.format(t + 1, self.T, bal))
        return PD, PF, Bal, AUC, mcc, f1, gm, pdx, pf, bal


if __name__ == '__main__':
    PD1 = []
    PF1 = []
    Bal1 = []
    AUC1 = []
    mcc1 = []
    f11 = []
    gm1 = []
    pd1 = []
    pf1 = []
    bal1 = []

    for i in range(10):
        Xt, Yt, Xs, Ys = linux_pre, linux_y, netbsd_pre, netbsd_y

        bda = BDA(kernel_type='primal', dim=5, lamb=0.9, mu=0.1, mode='BDA', gamma=1, estimate_mu=False)
        PD, PF, Bal, AUC, mcc, f1, gm, pdx, pf,  bal = bda.fit_predict(Xs, Ys, Xt, Yt)

        PD1.append(PD)
        PF1.append(PF)
        Bal1.append(Bal)
        AUC1.append(AUC)
        mcc1.append(mcc1)
        f11.append(f1)
        gm1.append(gm)
        pd1.append(pdx)
        pf1.append(pf)

        bal1.append(bal)

    print('pd', np.mean(PD1))
    print('pf', np.mean(PF1))
    print('bal', np.mean(Bal1))
    print('auc', np.mean(AUC1))
    # print(np.mean(mcc1))
    # print(np.mean(f11))
    # print(np.mean(gm1))
    # print(np.mean(bal1))
    # print(np.mean(Bal1))
    oldWb = xlrd.open_workbook('D:\\实验数据\\ROS RUS SMOTE.xls')
    newWb = copy(oldWb)
    newWs = newWb.get_sheet('Sheet1')
    newWs.write(5, 17, '{:.4f}'.format(np.mean(PD1)))
    newWs.write(36, 17, '{:.4f}'.format(np.mean(PF1)))
    newWs.write(67, 17, '{:.4f}'.format(np.mean(Bal1)))
    newWs.write(98, 17, '{:.4f}'.format(np.mean(AUC1)))
    newWb.save('D:\\实验数据\\ROS RUS SMOTE.xls')