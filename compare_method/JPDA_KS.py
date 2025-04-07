import numpy as np
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kmeans_smote import KMeansSMOTE
import warnings

warnings.filterwarnings("ignore")
import scipy.io
import scipy.linalg
import sklearn.metrics

import math
import warnings

import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from savedata import savedata, savedata_order


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=245)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)



# 核函数选择:实验中不用核函数，即选用primal
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    elif ker == 'Laplacian':
        if X2:
            # 拉普拉斯核函数
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, None, gamma)
    # K = sklearn.metrics.pairwise.polynomial_kernel()
    return K


# 两个项目间的联合概率分布差异 M ； 本文采用JPDA，选用djp-mmd
def get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, mu, type='djp-mmd'):
    M = 0
    if type == 'jmmd':
        N = 0
        n = ns + nt
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(1, C + 1):
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

    if type == 'jp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        M = Rmin / np.linalg.norm(Rmin, 'fro')

    if type == 'djp-mmd':
        ohe = OneHotEncoder()  # 独热编码
        ohe.fit(np.unique(Ys).reshape(-1, 1))  # 对源项目标签矩阵编码
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability 可迁移性（参考原文公式）
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # For discriminability 可辨别性（参考原文公式）
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax  # 项目间的联合概率分布差异表达式，要求M最小

    return M


# 独热编码
def onehot(y, c):
    nc = len(c)
    ny = len(y)
    y_onehot = np.zeros((len(y), nc))
    for i in range(ny):
        for j in range(len(c)):
            if y[i] == c[j]:
                y_onehot[i, j] = 1
    return y_onehot


class JPDA_KS:
    def __init__(self, kernel_type='primal', mmd_type='djp-mmd', dim=82, lamb=1, gamma=1, mu=0.1, T=10,
                 model=LogisticRegression()):
        '''
		Init func
		:param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
		:param dim: dimension after transfer
		:param lamb: lambda value in equation
		:param gamma: kernel bandwidth for rbf kernel
		:param T: iteration number
		'''
        self.kernel_type = kernel_type
        self.mmd_type = mmd_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.mu = mu
        self.T = T
        self.model = model

    def predict(self, Xs, Ys, Xt, Yt):
        # 预测的目标项目标签
        y_pred = []
        clf = self.model
        clf.fit(Xs, Ys.ravel())
        y_pred = clf.predict(Xt)
        score = clf.predict_proba(Xt)[:, 1]

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
        AUC = roc_auc_score(Yt, y_pred)
        # pd, pf, Bal = utils.bal_optim(Yt, score)
        return AUC, PD, PF, Bal, y_pred

    def fit_predict(self, Xs, Ys, Xt, Yt):
        # 求转换矩阵Z
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0)))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)

        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            M = get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, self.mu, type=self.mmd_type)

            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            # model_pca = PCA(n_components=1)
            # Xs_new = model_pca.fit_transform(Xs_new)
            # Xt_new = model_pca.fit_transform(Xt_new)

            model = KMeansSMOTE(kmeans_args={'n_clusters': 5},
                                smote_args={'k_neighbors': 4}, random_state=1)

            x_resampled, y_resampled = model.fit_resample(Xs_new, Ys)

            AUC, PD, PF, Bal, y_pred = self.predict(x_resampled, y_resampled, Xt_new, Yt)
            Y_tar_pseudo = y_pred

            print('JPDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
            print('JPDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            print('JPDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            print('JPDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))

        return AUC, PD, PF, Bal


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
                        jpda = JPDA_KS(kernel_type='primal', mmd_type='djp-mmd', dim=2, lamb=1, gamma=1, mu=0.4, T=10,
                                       model=model)
                        AUC, PD, PF, Bal = jpda.fit_predict(Xs, Ys, Xt, Yt)

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
                    # savedata('F:\\paper result\\model result finally.xls', 4 + down * 9, index + col, performance)
                    savedata_order('F:\\paper result\\order.xls', 8 + down * 14, index + col, performance)
                    index += 1
                left += 1
        down += 1
