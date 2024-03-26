#!/usr/bin/env python
# coding: utf-8

# In[263]:


# !pip install optuna


# 평가 서버 RF 모델 학습 및 추론 환경
# 
# OS : Ubuntu 18.04.3 LTS  
# Python : 3.6.9  
# Sklearn : 0.21.3  
# Random Seed : 42
# 
# - https://sosoeasy.tistory.com/597

# In[18]:


import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import optuna
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# In[33]:


sklearn.__version__


# In[29]:


get_ipython().system('pip freeze > requirements.txt')


# ## Data load

# In[20]:


df = pd.read_csv('data/train.csv')
df.head()


# In[21]:


feature_names = df.columns.to_list()
feature_names.remove('login')
feature_names.remove('person_id')

label_name = 'login'

X = df[feature_names]
y = df[label_name]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# ## 학습 준비
# 
# ```
# RF 모델을 학습시킬 모델 하이퍼파라미터 목록
# 
# n_estimators: 
#     기본값: 10
#     범위: 10 ~ 1000 사이의 양의 정수. 일반적으로 값이 클수록 모델 성능이 좋아지지만, 계산 비용과 시간도 증가합니다.
# criterion:
#     기본값: 'gini'
#     옵션: 'gini', 'entropy'. 'gini'는 진니 불순도를, 'entropy'는 정보 이득을 기준으로 합니다.
# max_depth:
#     기본값: None
#     범위: None 또는 양의 정수. None으로 설정하면 노드가 모든 리프가 순수해질 때까지 확장됩니다. 양의 정수를 설정하면 트리의 최대 깊이를 제한합니다.
# min_samples_split:
#     기본값: 2
#     범위: 2 이상의 정수 또는 0과 1 사이의 실수 (비율을 나타냄, (0, 1] ). 내부 노드를 분할하기 위해 필요한 최소 샘플 수를 지정합니다.
# min_samples_leaf:
#     기본값: 1
#     범위: 1 이상의 정수 또는 0과 0.5 사이의 실수 (비율을 나타냄, (0, 0.5] ). 리프 노드가 가져야 하는 최소 샘플 수를 지정합니다.
# min_weight_fraction_leaf:
#     기본값: 0.0
#     범위: 0.0에서 0.5 사이의 실수. 리프 노드에 있어야 하는 샘플의 최소 가중치 비율을 지정합니다.
# max_features:
#     기본값: 'auto'
#     옵션: 'auto', 'sqrt', 'log2', None 또는 양의 정수/실수. 최적의 분할을 찾기 위해 고려할 특성의 수 또는 비율을 지정합니다. 
#           'auto'는 모든 특성을 사용함을 의미하며, 'sqrt'와 'log2'는 각각 특성의 제곱근과 로그2를 사용합니다. 
#           None은 'auto'와 동일하게 모든 특성을 의미합니다.
# max_leaf_nodes:
#     기본값: None
#     범위: None 또는 양의 정수. 리프 노드의 최대 수를 제한합니다. None은 무제한을 의미합니다.
# min_impurity_decrease:
#     기본값: 0.0
#     범위: 0.0 이상의 실수. 노드를 분할할 때 감소해야 하는 불순도의 최소량을 지정합니다.
# bootstrap:
#     기본값: True
#     옵션: True, False. True는 부트스트랩 샘플을 사용하여 개별 트리를 학습시킵니다. False는 전체 데이터셋을 사용하여 각 트리를 학습시킵니다.
# ```

# ### 1. Fold 없이
# - https://velog.io/@halinee/Optuna%EB%A1%9C-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D%ED%95%98%EA%B8%B0

# In[31]:


# 목적 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    # min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5, step=0.1)
    # max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    # max_features = trial.suggest_float('max_features', 0, 0.5, step=0.1)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 20)
    # min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0, 0.5)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    # Random Forest 모델 생성
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        # min_weight_fraction_leaf=min_weight_fraction_leaf,
        # max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        # min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        random_state=42
    )
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 검증 데이터에 대한 예측
    y_valid_pred = model.predict(X_valid)
    
    # 검증 데이터에 대한 성능 계산
    roc = roc_auc_score(y_valid, y_valid_pred)
    # acc =  accuracy_score(y_valid, y_valid_pred)
    # fscore = f1_score(y_valid, y_valid_pred)
    
    return roc


# In[32]:


# 학습 객체 생성
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

best_params = study.best_params
best_score = study.best_value
print('Best Parameters:', best_params)
print('Best score:', best_score)


# In[39]:


best_params


# In[40]:


# best 파라미터로 모델 학습
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    # min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    # min_impurity_decrease=best_params['min_impurity_decrease'],
    # max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)
best_model.fit(X_train, y_train)


# In[41]:


y_valid_pred = best_model.predict(X)

print('roc:', roc_auc_score(y, y_valid_pred))
print('accuracy_score:', accuracy_score(y, y_valid_pred))
print('f1_score:', f1_score(y, y_valid_pred))


# ### 2. Fold 사용
# 
# - https://teddylee777.github.io/data-science/optuna/
# 
# - 수정중...

# In[6]:


df = pd.read_csv('data/train.csv')

feature_names = df.columns.to_list()
feature_names.remove('login')
feature_names.remove('person_id')

label_name = 'login'

X = df[feature_names]
y = df[[label_name]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[8]:


def objective(trial, X, y):
    # 하이퍼파라미터 탐색 공간 정의
    param = {
             "n_estimators": trial.suggest_int('n_estimators', 100, 1000, step=100),
             "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy']),
             "max_depth": trial.suggest_int('max_depth', 10, 50),
             "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
             "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 10),
             'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5, step=0.1),
             'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 20),
             'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.5),
             'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
             # 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
             # 'max_features': trial.suggest_float('max_features', 0, 0.5, step=0.1),
            }

    cv = KFold(n_splits=5, shuffle=True)
    cv_scores = []

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_t, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_val = y[train_idx], y[val_idx]

        model = RandomForestClassifier(random_state=42, **param)
        model.fit(X_t, y_t)
        y_pred = model.predict(X_val)
        cv_scores.append(roc_auc_score(y_val, y_pred))

    roc_score = np.mean(cv_scores)
    return roc_score


# In[9]:


study = optuna.create_study(
                            study_name='RandomForestClassifier', direction='maximize', 
                            # sampler=TPESampler(seed=21)
                           )
study.optimize(lambda trial: objective(trial, X_train, y_train), 
               n_trials=30)

print()
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)


# ## Save

# In[42]:


submit = pd.read_csv('data/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('submit_file.csv', index=False)


# ```
# SUMIT 01 >
# RandomForestClassifier(bootstrap=False, max_depth=41, max_leaf_nodes=20,
#                        min_samples_leaf=4, min_samples_split=9,
#                        n_estimators=200, random_state=42)
# -> 제출점수 0.78085
# ```

# In[28]:


# .py파일로 변환
get_ipython().system('jupyter nbconvert --to script RF_tuning_optuna.ipynb')

