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
import xlrd
from sklearn.cluster import KMeans
import scipy.io
import scipy.linalg
import sklearn.metrics
from numpy import linalg
from scipy import linalg
from sklearn.model_selection import  cross_val_score
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

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=245)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)
from numpy import linalg
from scipy import linalg
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
import scipy.io
import scipy.linalg
import sklearn.metrics

import scipy.io
import scipy.linalg
import sklearn.metrics

from sklearn.neighbors import KNeighborsClassifier

import scipy.io
import scipy.linalg

import sklearn
import warnings

warnings.filterwarnings("ignore")

from self_paced_ensemble1 import SelfPacedEnsemble
from canonical_ensemble import *
from utils import *
import argparse
from tqdm import trange
from sklearn.decomposition import PCA

# 数据预处理
linux = np.loadtxt('/Users/orange/Downloads/linux.csv', delimiter=',', skiprows=1)
linux_x = np.delete(linux, -1, axis=1)
linux_y = linux[:, 82]
# print(linux_y)

netbsd = np.loadtxt('/Users/orange/Downloads/netbsd.csv', delimiter=',', skiprows=1)
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

# print(linux_mean)
# print(netbsd_mean)
# print(netbsd_pre)
pca = PCA(n_components=5)
linux_pre = pca.fit_transform(linux_pre1)
netbsd_pre = pca.fit_transform(netbsd_pre1)
netbsd_max = np.max(netbsd_x, axis=0)
netbsd_min = np.min(netbsd_x, axis=0)
linux_max = np.max(linux_x, axis=0)
linux_min = np.min(linux_x, axis=0)

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


linux_y = linux_y.reshape(3400,1)
netbsd_y = netbsd_y.reshape(1734,1)
METHODS = ['SPEnsemble', 'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'Cascade']
RANDOM_STATE = 42

def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Self-paced Ensemble',
        usage='run_example.py --method <method> --n_estimators <integer> --runs <integer>'
    )
    parser.add_argument('--method', type=str, default='SPEnsemble',
                        choices=METHODS + ['all'], help='Name of ensmeble method')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of base estimators')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs')
    return parser.parse_args()


def init_model(method, base_estimator, n_estimators):
    '''return a model specified by "method".'''
    if method == 'SPEnsemble':
        model = SelfPacedEnsemble(base_estimator=base_estimator, n_estimators=n_estimators)
    elif method == 'SMOTEBoost':
        model = SMOTEBoost(base_estimator=base_estimator, n_estimators=n_estimators)
    elif method == 'SMOTEBagging':
        model = SMOTEBagging(base_estimator=base_estimator, n_estimators=n_estimators)
    elif method == 'RUSBoost':
        model = RUSBoost(base_estimator=base_estimator, n_estimators=n_estimators)
    elif method == 'UnderBagging':
        model = UnderBagging(base_estimator=base_estimator, n_estimators=n_estimators)
    elif method == 'Cascade':
        model = BalanceCascade(base_estimator=base_estimator, n_estimators=n_estimators)
    else:
        raise Error('No such method support: {}'.format(method))
    return model


def SPEtest(Xs, Ys, Xt, Yt):

    # Parse arguments
    args = parse()
    method_used = args.method
    n_estimators = args.n_estimators
    runs = args.runs
    #tca = TCA(kernel_type='primal', dim=2, lamb=1, gamma=1)
    #Xs_new, Xt_new = tca.fit(linux_pre, netbsd_pre)
    runs = 1
    X_train, X_test, y_train, y_test =  Xs, Xt, Ys, Yt
    #print(X_train)
    #print(X_test)
    #print(y_train)
    #print(y_test)
    # Train & Record
    method_list = METHODS if method_used == 'all' else [method_used]
    for method in method_list:
        print('\nRunning method:\t\t{} - {} estimators in {} independent run(s) ...'.format(
            method, n_estimators, runs))
        # print('Running ...')
        scores = [];
        times = []
        try:
            with trange(runs) as t:
                for _ in t:
                    model = init_model(
                        method=method,
                        n_estimators=n_estimators,
                        # base_estimator=sklearn.tree.DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=3, random_state=2),
                        # base_estimator=KNeighborsClassifier(n_neighbors=5),
                        # base_estimator=LogisticRegression(C=1),
                        # base_estimator=RandomForestClassifier(n_estimators=10,max_depth=1, random_state=1),
                        # base_estimator=GaussianNB(),
                        # base_estimator=SVC(probability=True,gamma='auto'),
                    )
                    start_time = time.process_time()
                    model.fit(X_train, y_train)
                    times.append(time.process_time() - start_time)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    scores.append([
                        auc_prc(y_test, y_pred),
                        f1_optim(y_test, y_pred),
                        gm_optim(y_test, y_pred),
                        mcc_optim(y_test, y_pred),
                        bal_optim(y_test, y_pred)
                    ])
        except KeyboardInterrupt:
            t.close()
            raise
        t.close

        # Print results to console
        print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
        print('------------------------------')
        print('Metrics:')
        df_scores = pd.DataFrame(scores, columns=['AUC', 'F1', 'G-mean', 'MCC', 'Bal'])
        #for metric in df_scores.columns.tolist():
         #   print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    #print(y_pred)
    AUC = auc_prc(y_test, y_pred)
    F1 = f1_optim(y_test, y_pred)
    gm = gm_optim(y_test, y_pred)
    mcc = mcc_optim(y_test, y_pred)
    bal = bal_optim(y_test, y_pred)


    y_pred_b = y_pred.copy()

    y_pred_b[y_pred_b < 0.6] = 0
    y_pred_b[y_pred_b >= 0.6] = 1
    return y_pred_b, AUC, F1, gm, mcc, bal


class IDSA:
    def __init__(self, type='nb'):
        self.type = type

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
        H1 = 1
        #print(H1)
        #print(z)
        TN = 0
        FN = 0
        TP = 0
        FP = 0

        #H2 = np.round(np.shape(X2)[0] / (np.shape(X1)[0] / H1)).astype(int)
        X1_train = []
        X2_train = []
        #X1 = []
        #X2 = []
        mm = np.hstack((Xs, Ys))
        #print(np.shape(mm)[0])

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
        H2 = np.round(np.shape(X2)[0] / (np.shape(X1)[0] / H1)).astype(int)
        #H2 = np.round(689 / (41 / H1)).astype(int)


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

        #model = RandomUnderSampler()
        #x_resampled, y_resampled = model.fit_resample(Xs_new, XX_y)

        #y_pred = self.machine(x_resampled, y_resampled, Yt_new)
        #y_pred = self.machine(Xs_new, XX_y, Yt_new)
        #print(XX_y)
        #print(Yt)
        #print(np.shape(Ys))
        #Ys1 = Ys.reshape(-1)
        #print(np.shape(Ys1))
        #Yt1 = Yt.reshape(470, 1)
        y_pred, AUC, F1, gm, mcc, bal= SPEtest(Xs_new, XX_y, Yt_new, Yt)
        #y_pred, AUC, F1, gm, mcc, bal = SPEtest(Xs, Ys, Xt, Yt)
        #print(auc)
        #print(f1)
        #y_pred = SPEtest(Xs, Ys, Xt, Yt)
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



        return PD, PF, Bal, AUC, F1, gm, mcc, bal, y_pred


       # return PD, PF, Bal, y_pred
class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0, gamma=1, T=1, mode='BDA', estimate_mu=False):
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
        list_acc = []

        for t in range(self.T):
            dsa = IDSA(type='lr')
            PD, PF, Bal, AUC, F1, gm, mcc, bal, y_pred = dsa.isda(Xs, Ys, Xt, Yt)
            Y_tar_pseudo = y_pred

            print('BDA iteration [{}/{}]: PD: {:.4f}'.format(t + 1, self.T, PD))
            print('BDA iteration [{}/{}]: PF: {:.4f}'.format(t + 1, self.T, PF))
            print('BDA iteration [{}/{}]: Bal: {:.4f}'.format(t + 1, self.T, Bal))
            print('BDA iteration [{}/{}]: AUC: {:.4f}'.format(t + 1, self.T, AUC))
            print('BDA iteration [{}/{}]: mcc: {:.4f}'.format(t + 1, self.T, mcc))
            print('BDA iteration [{}/{}]: f1: {:.4f}'.format(t + 1, self.T, F1))
            print('BDA iteration [{}/{}]: gm: {:.4f}'.format(t + 1, self.T, gm))
            print('BDA iteration [{}/{}]: bal_gailv: {:.4f}'.format(t + 1, self.T, bal))
        return PD, PF, Bal, AUC, F1, gm, mcc, bal


if __name__ == '__main__':
    PD1 = []
    PF1 = []
    Bal1 = []
    AUC1 = []
    mcc1 = []
    f11 = []
    gm1 = []
    bal1 = []

    for i in range(10):
        Xt, Yt, Xs, Ys = linux_pre, linux_y, netbsd_pre, netbsd_y

        bda = BDA(kernel_type='primal', dim=5, lamb=0.9, mu=0.1, mode='BDA', gamma=1, estimate_mu=False)
        PD, PF, Bal, AUC, F1, gm, mcc, bal = bda.fit_predict(Xs, Ys, Xt, Yt)

        PD1.append(PD)
        PF1.append(PF)
        Bal1.append(Bal)
        AUC1.append(AUC)
        mcc1.append(mcc1)
        f11.append(F1)
        gm1.append(gm)
        bal1.append(bal)

    print(np.mean(PD1))
    print(np.mean(PF1))
    print(np.mean(Bal1))
    print(np.mean(AUC1))

    print(np.mean(f11))
    print(np.mean(gm1))
    print(np.mean(bal1))
