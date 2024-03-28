# RandomSearch를 이용한 모델 튜닝

import copy
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




data = pd.read_csv('./train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']




param_search_space = {
    'n_estimators': [10, 40, 100],
    'min_impurity_decrease': uniform(0.0001, 0.01),
    'max_depth': randint(10, 40),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV 객체 생성
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_search_space, n_iter=250, cv=2, n_jobs=-1, verbose=2, scoring='roc_auc', random_state=42)

# RandomizedSearchCV를 사용한 학습
random_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = random_search.best_params_
best_score = random_search.best_score_

print(best_params, best_score)




submit = pd.read_csv('./sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('./baseline_submit.csv', index=False)

