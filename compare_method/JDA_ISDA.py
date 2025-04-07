import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
import math
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from numpy.linalg import eig
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from numpy import linalg
from scipy import linalg
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from savedata import savedata, savedata_order
import warnings
warnings.filterwarnings("ignore")

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

    def machine(self, Xs, Ys, Xt):

        y_pred = []
        cls = self.model
        cls.fit(Xs, Ys)
        y_pred = cls.predict(Xt)
        return y_pred

    # 留一运算
    def loop(self, Xs, Ys):
        Train = np.hstack((Xs, Ys))
        X1_1 = []
        X2_1 = []
        for i in range(np.shape(Xs)[0]):
            if Train[i][-1] == 1:
                X1_1.append(Train[i, :])
            else:
                X2_1.append(Train[i, :])

        # 重新组合后的training set X_train
        # X_train = np.vstack((X1_1, X2_1))
        n1 = np.shape(X1_1)[0]
        n2 = np.shape(X2_1)[0]
        # nw = n1 + n2
        bal = []

        for H1 in range(1, np.shape(X1_1)[0] - 1):
            # evaluation metrics
            TN = 0
            FN = 0
            TP = 0
            FP = 0
            H2 = np.round(n2 / (n1 / H1)).astype(int)

            # LOOP留一运算（依次取出训练集中的每一个样本）
            for j in range(np.shape(Train)[0]):

                X1_train = []
                X2_train = []

                test_y_feature = Xs[j]
                test_y = Ys[j]

                training = np.delete(Train, j, axis=0)
                # 将training按照0，1分类为X1（defective），X2（free）
                for i in range(np.shape(training)[0]):
                    if training[i][-1] == 1:
                        X1_train.append(training[i, :])
                    else:
                        X2_train.append(training[i, :])

                X222 = np.vstack((X1_train, X2_train))
                # 除一之后的训练集label
                X_y = X222[:, -1]

                # 删除最后一列，得到完整的X1，X2
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

                for j in range(1, H1 + 1):
                    for l in range(1, H2 + 1):
                        pij = np.shape(x1[j - 1])[0] / n
                        pkl = np.shape(x2[l - 1])[0] / n
                        uij = x1[j - 1].mean(0)
                        ukl = x2[l - 1].mean(0)
                        a = uij - ukl
                        b = a.reshape((np.shape(X1)[1], 1))
                        B = B + pij * pkl * (a * b)

                XC = self._cov(X)
                XC = np.asarray(XC)

                evals, evecs = linalg.eigh(B / XC)
                evecs = evecs[:, np.argsort(evals)[::-1]]
                Xs_new = np.dot(X, evecs)
                Y_new = np.dot([test_y_feature], evecs)

                # 预测
                y_pred = self.machine(Xs_new, X_y, Y_new)
                # print(np.shape(y_pred))
                # print(y_pred)
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
        # print(m)
        # z = m + 1
        # H1 = z
        H1 = 1
        # print(H1)
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        H2 = np.round(689 / (41 / H1)).astype(int)
        X1_train = []
        X2_train = []
        # X1 = []
        # X2 = []

        Ys = Ys.reshape((Ys.shape[0], 1))
        mm = np.hstack((Xs, Ys))

        # 将train_x按照0，1分类为X1（defective），X2（free）
        for i in range(np.shape(mm)[0]):
            if mm[i][-1] == 1:
                X1_train.append(mm[i, :])
            else:
                X2_train.append(mm[i, :])
        # print(X1_train)
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

        # 类间散度
        for j in range(1, H1 + 1):
            for l in range(1, H2 + 1):
                pij = np.shape(x1[j - 1])[0] / n
                pkl = np.shape(x2[l - 1])[0] / n
                uij = x1[j - 1].mean(0)
                ukl = x2[l - 1].mean(0)
                a = uij - ukl
                b = a.reshape((np.shape(X1)[1], 1))
                B = B + pij * pkl * (a * b)

        # 协方差矩阵
        XM = self._cov(X)
        XM = np.asarray(XM)

        # 计算特征向量
        evals, evecs = linalg.eigh(B / XM)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        # print(evecs)
        # print(np.shape(evecs))
        Xs_new = np.dot(X, evecs)
        Yt_new = np.dot(Xt, evecs)

        model = RandomUnderSampler(random_state=1)
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
            # k = sklearn.metrics.pairwise.
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    elif ker == 'Laplacian':
        if X2:
            # 实验拉普拉斯核函数
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
            # K = sklearn.metrics.pairwise.polynomial_kernel()

        else:
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=54, lamb=1, gamma=1, T=10, model=LogisticRegression()):
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
        # list_acc = []
        global PD
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            # M = e * e.T

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

            dsa = IDSA(model=self.model)
            PD, PF, Bal, AUC, y_pred = dsa.isda(Xs_new, Ys, Xt_new, Yt)

            print('JDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            print('JDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            print('JDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            print('JDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
        return PD, PF, Bal, AUC, y_pred


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
                    KNeighborsClassifier(n_neighbors=20),
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
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        jda = JDA(kernel_type='primal', dim=5, lamb=1, gamma=1, model=model)
                        PD, PF, Bal, AUC, y_pred = jda.fit_predict(Xs, Ys, Xt, Yt)

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
                    # savedata('F:\\paper result\\model result finally.xls', 5 + down * 43, index + col, performance)
                    savedata_order('F:\\paper result\\order.xls', 6 + down * 14, index + col, performance)
                    index += 1
                left += 1
        down += 1
