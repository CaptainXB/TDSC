import scipy.io
import scipy.linalg
import sklearn.metrics
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
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
from savedata import savedata, savedata_order
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=245)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)


# 核函数选择
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
            # 实验拉普拉斯核函数
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.laplacian_kernel(np.asarray(X1).T, None, gamma)
    # K = sklearn.metrics.pairwise.polynomial_kernel()
    return K


class TCA:
    def __init__(self, kernel_type='rbf', dim=30, lamb=1, gamma=1, model=LogisticRegression()):
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

    def fit(self, Xs, Xt, Ys):
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

        # 类不平衡处理
        ros = RandomOverSampler(random_state=1)
        Xs_new, Ys = ros.fit_resample(Xs_new, Ys)

        return Xs_new, Xt_new, Ys

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''

        Xs_new, Xt_new, Ys = self.fit(Xs, Xt, Ys)
        clf = self.model
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)

        # evaluation metrics
        TN = 0
        FN = 0
        TP = 0
        FP = 0

        for i in range(np.shape(Xt_new)[0]):
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

                standard = StandardScaler()

                source_x = standard.fit_transform(source_x)
                target_x = standard.fit_transform(target_x)

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
                    for i in range(10):
                        Xs, Ys, Xt, Yt = source_x, source_y, target_x, target_y
                        tca = TCA(kernel_type='primal', dim=5, lamb=1, gamma=1, model=model)
                        PD, PF, Bal, AUC = tca.fit_predict(Xs, Ys, Xt, Yt)

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
                    # savedata('F:\\paper result\\order.xls', 4 + down * 14, index + col, performance)
                    savedata_order('F:\\paper result\\order.xls', 4 + down * 14, index + col, performance)
                    index += 1
                left += 1
        down += 1
