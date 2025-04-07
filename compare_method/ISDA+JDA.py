import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
import math
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import xlrd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import  SMOTE
from imblearn.under_sampling import  RandomUnderSampler
#from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import RUSBoostClassifier
from numpy.linalg import eig
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from numpy.linalg import eig
from numpy import linalg
from scipy import linalg
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

#数据预处理
from xlutils.copy import copy

netbsd = np.loadtxt('E:\\class-overlap-master\\data\\aging data\\netbsd.csv', delimiter=',', skiprows=1)
netbsd_x = np.delete(netbsd, -1, axis=1)
netbsd_y = netbsd[:, -1]
# print(netbsd_y)

mysql = np.loadtxt('E:\\class-overlap-master\\data\\aging data\\mysql.csv', delimiter=',', skiprows=1)
mysql_x = np.delete(mysql, -1, axis=1)
mysql_y = mysql[:, -1]


#标准化数据
netbsd_mean = np.mean(netbsd_x,axis=0)
netbsd_std = np.std(netbsd_x,axis=0)
mysql_mean = np.mean(mysql_x,axis=0)
mysql_std = np.std(mysql_x,axis=0)
netbsd_pre = netbsd_x
mysql_pre = mysql_x

#N3
# for i in range(np.shape(netbsd_pre)[0]):
#     for j in range(np.shape(netbsd_pre)[1]):
#         if netbsd_std[j] != 0:
#            netbsd_pre[i][j] = (netbsd_x[i][j] - netbsd_mean[j]) / netbsd_std[j]
#         else:
#             netbsd_pre[i][j] = netbsd_x[i][j] - netbsd_mean[j]
# for i in range(np.shape(mysql_pre)[0]):
#     for j in range(np.shape(mysql_pre)[1]):
#         if netbsd_std[j] != 0:
#            mysql_pre[i][j] = (mysql_x[i][j] - netbsd_mean[j]) / netbsd_std[j]
#         else:
#            mysql_pre[i][j] = mysql_x[i][j] - netbsd_mean[j]

# N2
for i in range(np.shape(netbsd_pre)[0]):
    for j in range(np.shape(netbsd_pre)[1]):
        if netbsd_std[j] != 0:
           netbsd_pre[i][j] = (netbsd_x[i][j] - netbsd_mean[j]) / netbsd_std[j]
        else:
            netbsd_pre[i][j] = netbsd_x[i][j] - netbsd_mean[j]
for i in range(np.shape(mysql_pre)[0]):
    for j in range(np.shape(mysql_pre)[1]):
        if mysql_std[j] != 0:
           mysql_pre[i][j] = (mysql_x[i][j] - mysql_mean[j]) / mysql_std[j]
        else:
           mysql_pre[i][j] = mysql_x[i][j] - mysql_mean[j]


netbsd_y = netbsd_y.reshape(1734,1)
mysql_y = mysql_y.reshape(470,1)




class IDSA:
    def __init__(self, type = 'nb'):
        self.type = type


    #方差计算（协方差）
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
        s = np.cov(X, rowvar=0, bias = 1)
        return s

    def machine(self, Xs, Ys, Xt):

        y_pred = []
        if self.type == 'nb':
            model = GaussianNB()
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)

        elif self.type == 'lr':
            model = LogisticRegression(C=1)
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)
        elif self.type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)
        elif self.type == 'dt':
            model = DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=3, random_state=2)
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)
        elif self.type == 'rf':
            model = RandomForestClassifier(n_estimators=10,max_depth=1,random_state=1)
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)
        elif self.type == 'svm':
            model = SVC(probability=True, gamma='auto')
            model.fit(Xs, Ys)
            y_pred = model.predict(Xt)

        return y_pred


    #留一运算
    def loop(self, Xs, Ys):
        Train = np.hstack((Xs, Ys))
        X1_1 = []
        X2_1 = []
        for i in range(np.shape(Xs)[0]):
            if Train[i][-1] == 1:
                X1_1.append(Train[i, :])
            else:
                X2_1.append(Train[i, :])

        #重新组合后的training set X_train
        #X_train = np.vstack((X1_1, X2_1))
        n1 = np.shape(X1_1)[0]
        n2 = np.shape(X2_1)[0]
        #nw = n1 + n2
        bal = []

        for H1 in range(1, np.shape(X1_1)[0]-1):
            #evaluation metrics
            TN = 0
            FN = 0
            TP = 0
            FP = 0
            H2 = np.round(n2/(n1/H1)).astype(int)

            # LOOP留一运算（依次取出训练集中的每一个样本）
            for j in range(np.shape(Train)[0]):

                X1_train = []
                X2_train = []

                test_y_feature = Xs[j]
                test_y = Ys[j]

                training = np.delete(Train, j, axis=0)
                #将training按照0，1分类为X1（defective），X2（free）
                for i in range(np.shape(training)[0]):
                    if training[i][-1] == 1:
                        X1_train.append(training[i, :])
                    else:
                        X2_train.append(training[i, :])

                X222 = np.vstack((X1_train, X2_train))
                #除一之后的训练集label
                X_y = X222[:, -1]


                #删除最后一列，得到完整的X1，X2
                X1 = np.delete(X1_train, -1, axis=1)
                X2 = np.delete(X2_train, -1, axis=1)
                X = np.vstack((X1, X2))


                #利用Kmeans将X1，X2分为H1，H2个子类
                estimator_x1 = KMeans(n_clusters=H1)
                estimator_x2 = KMeans(n_clusters=H2)
                estimator_x1.fit(X1)
                estimator_x2.fit(X2)
                label_pred_x1 = estimator_x1.labels_
                label_pred_x2 = estimator_x2.labels_
                x1 = []
                x2 = []
                for i in range(H1):
                    x1.append(X1[label_pred_x1 == i])
                for i in range(H2):
                    x2.append(X2[label_pred_x2 == i])

                #求解类间离散矩阵，数据的协方差，及最终求解投影向量

                B = np.zeros((np.shape(X1)[1], np.shape(X1)[1]))
                n = np.shape(X)[0]

                for j in range(1, H1 + 1):
                    for l in range(1, H2 + 1):
                        pij = np.shape(x1[j - 1])[0] / n
                        pkl = np.shape(x2[l - 1])[0] / n
                        uij = x1[j - 1].mean(0)
                        ukl = x2[l - 1].mean(0)
                        a = uij -ukl
                        b = a.reshape((np.shape(X1)[1],1))
                        B = B + pij * pkl * (a * b)


                XC = self._cov(X)
                XC = np.asarray(XC)


                evals, evecs = linalg.eigh(B/XC)
                evecs = evecs[:, np.argsort(evals)[::-1]]
                Xs_new = np.dot(X, evecs)
                Y_new = np.dot([test_y_feature], evecs)



                #预测
                y_pred = self.machine(Xs_new, X_y, Y_new)
                #print(np.shape(y_pred))
                #print(y_pred)
                if test_y == 0:
                    if y_pred[0] == 0:
                        TN = TN + 1
                    else:
                        FP = FP + 1
                else:
                    if y_pred[0] == 0:
                        FN = FN + 1
                    else:
                        TP = TP + 1

            PD = TP / (TP + FN)
            PF = FP / (FP + TN)
            Bal = 1 - math.sqrt(((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
            bal.append(Bal)
        return bal, n1, n2


    def isda(self, Xs, Ys, Xt, Yt):
        '''
        bal, n1, n2 = self.loop(Xs, Ys)
        print(bal)
        #在训练集上进行预测
        m = 0
        for i in range(len(bal)):
           if bal[i] > bal[m]:
               m = i
        '''
        #print(m)
        #z = m + 1
        #H1 = z
        H1 = 1
        # print(H1)
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        H2 = np.round(689 / (41 / H1)).astype(int)
        X1_train = []
        X2_train = []
        #X1 = []
        #X2 = []
        mm = np.hstack((Xs, Ys))

        # 将train_x按照0，1分类为X1（defective），X2（free）
        for i in range(np.shape(mm)[0]):
            if mm[i][-1] == 1:
                X1_train.append(mm[i, :])
            else:
                X2_train.append(mm[i, :])
        print(X1_train)
        # 删除最后一列，得到完整的X1，X2
        XX = np.vstack((X1_train, X2_train))
        XX_y = XX[:, -1]
        X1 = np.delete(X1_train, -1, axis=1)
        X2 = np.delete(X2_train, -1, axis=1)
        X = np.vstack((X1, X2))


        # 利用Kmeans将X1，X2分为H1，H2个子类
        estimator_x1 = KMeans(n_clusters=H1)
        estimator_x2 = KMeans(n_clusters=H2)
        estimator_x1.fit(X1)
        estimator_x2.fit(X2)
        label_pred_x1 = estimator_x1.labels_
        label_pred_x2 = estimator_x2.labels_
        x1 = []
        x2 = []
        for i in range(H1):
            x1.append(X1[label_pred_x1 == i])
        for i in range(H2):
            x2.append(X2[label_pred_x2 == i])

        # 求解类间离散矩阵，数据的协方差，及最终求解投影向量

        B = np.zeros((np.shape(X1)[1], np.shape(X1)[1]))
        n = np.shape(X)[0]


        #类间散度
        for j in range(1, H1 + 1):
            for l in range(1, H2 + 1):
                pij = np.shape(x1[j - 1])[0] / n
                pkl = np.shape(x2[l - 1])[0] / n
                uij = x1[j - 1].mean(0)
                ukl = x2[l - 1].mean(0)
                a = uij - ukl
                b = a.reshape((np.shape(X1)[1], 1))
                B = B + pij * pkl * (a * b)


        #协方差矩阵
        XM = self._cov(X)
        XM = np.asarray(XM)


        #计算特征向量
        evals, evecs = linalg.eigh(B/XM)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        #print(evecs)
        #print(np.shape(evecs))
        Xs_new = np.dot(X, evecs)
        Yt_new = np.dot(Xt, evecs)

        model = RandomUnderSampler()
        x_resampled, y_resampled = model.fit_resample(Xs_new, XX_y)

        y_pred = self.machine(x_resampled, y_resampled, Yt_new)

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

        return PD, PF, Bal, AUC, y_pred

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
            #k = sklearn.metrics.pairwise.
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    elif ker == 'Laplacian':
        if X2:
            #实验拉普拉斯核函数
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
            #K = sklearn.metrics.pairwise.polynomial_kernel()

        else:
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, None, gamma)
    return K



class JDA:
    def __init__(self, kernel_type='primal', dim=54, lamb=1, gamma=1, T=10):
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


    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        #list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        #M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            #M = e * e.T

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


            dsa = IDSA(type='knn')
            PD, PF, Bal, AUC, y_pred = dsa.isda(Xs_new, Ys, Xt_new, Yt)

            print('JDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1,  self.T, PD))
            print('JDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1,  self.T, PF))
            print('JDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            print('JDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
        return PD, PF, Bal, AUC, y_pred


if __name__ == '__main__':
    #domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    PD1 = []
    PF1 = []
    Bal1 = []
    AUC1 =[]
    for i in range(10):
        Xt, Yt, Xs, Ys = netbsd_pre, netbsd_y, mysql_pre, mysql_y
        jda = JDA(kernel_type='primal', dim=5, lamb=1, gamma=1)
        PD, PF, Bal, AUC, y_pred = jda.fit_predict(Xs, Ys, Xt, Yt)
        #PD, Bal = jda.fit_predict_imbalance(Xs_new, Ys, Xt_new,Yt)


        PD1.append(PD)
        PF1.append(PF)
        Bal1.append(Bal)
        AUC1.append(AUC)

    print(np.mean(PD1))
    print(np.mean(PF1))
    print(np.mean(Bal1))
    print(np.mean(AUC1))

    # oldWb = xlrd.open_workbook('D:\\实验数据\\Trad TLAP JDA BISP.xls')
    # newWb = copy(oldWb)
    # newWs = newWb.get_sheet('Sheet1')
    # i = 12
    # newWs.write(25, i, '{:.4f}'.format(np.mean(PD1)))
    # newWs.write(55, i, '{:.4f}'.format(np.mean(PF1)))
    # newWs.write(85, i, '{:.4f}'.format(np.mean(Bal1)))
    # newWs.write(115, i, '{:.4f}'.format(np.mean(AUC1)))
    # newWb.save('D:\\实验数据\\Trad TLAP JDA BISP.xls')