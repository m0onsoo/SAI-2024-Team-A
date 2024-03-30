!pip install hyperopt

import pandas as pd
import sklearn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

PATH = "~~~~"

data = pd.read_csv(PATH + 'RF_hyperparameter_tuning_data/train.csv')
sample_submission = pd.read_csv(PATH + 'RF_hyperparameter_tuning_data/sample_submission.csv')
results = pd.read_csv(PATH + 'results.csv')


np.random.seed(42)
train, valid = train_test_split(data, test_size = 0.2, random_state = 0, stratify = data['login'])
train_X = train[train.columns[1:-1]]
train_Y = train['login']
valid_X = valid[valid.columns[1:-1]]
valid_Y = valid['login']
X = data[data.columns[1:-1]]
Y = data['login']



RF = RandomForestClassifier()

n = 7
stf = StratifiedKFold(n_splits = n, random_state = 42, shuffle = True)

criterion = ['gini', 'entropy']
max_depth = [4, 5, 6, 7]
max_features = ['auto', 'sqrt', 'log2', None]

# space: 알고리즘이 탐색할 범위를 정의한다.
# hp.choice: 리스트 내의 값을 무작위로 추출
# hp.uniform: 정의된 범위 내에서 임의의 숫자를 무작위 추출
# hp.quniform: 정의된 범위 내에서 마지막 숫자만큼의 간격을 두어 숫자를 추출
space = {'n_estimators' : hp.quniform('n_estimators', 100, 200, 10),
         'criterion': hp.choice('criterion', ['gini']),
         'max_depth': hp.choice('max_depth', max_depth),
         'min_samples_split' : hp.uniform ('min_samples_split', 80, 100),
         'min_samples_leaf': hp.uniform ('min_samples_leaf', 20, 30),
         'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
         'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.1),
         'max_features': hp.choice('max_features', max_features),
         'max_leaf_nodes': hp.uniform('max_leaf_nodes', 20, 80),
         'bootstrap': hp.choice('bootstrap', [True])
    }
# objective함수는 최소화(리턴하기 전에 음의 부호 취하기)
def RF_hyperparameter_tuning(space):
    hopt = RandomForestClassifier(n_estimators = int(space['n_estimators']),
                                  criterion = space['criterion'],
                                  max_depth = space['max_depth'],
                                  min_samples_split = int(space['min_samples_split']),
                                  min_samples_leaf = int(space['min_samples_leaf']),
                                  min_weight_fraction_leaf= space['min_weight_fraction_leaf'],
                                  min_impurity_decrease = space['min_impurity_decrease'],
                                  max_leaf_nodes = int(space['max_leaf_nodes']),
                                  max_features = space['max_features'],
                                  random_state = 42,
                                  verbose = 0,
                                 )

    auc_score = cross_val_score(hopt, X, Y, scoring = 'roc_auc', cv = stf)

    return {
        'loss':  (-1) * np.mean(auc_score),
        'status': STATUS_OK
    }

trials = Trials()
best = fmin(fn= RF_hyperparameter_tuning,
            space= space,
            algo= tpe.suggest,
            max_evals = 200,
            trials= trials)

best

best['n_estimators'] = int(best['n_estimators'])
best['criterion'] = 'gini'
best['max_depth'] = max_depth[best['max_depth']]
best['max_features'] = max_features[best['max_features']]
best['max_leaf_nodes'] = int(best['max_leaf_nodes'])
if best['min_samples_split'] >= 2:
  best['min_samples_split'] = int(best['min_samples_split'])
if best['min_samples_leaf'] >=1:
  best['min_samples_leaf'] = int(best['min_samples_leaf'])
best['bootstrap'] = True

print(best)

