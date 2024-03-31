# 유전 알고리즘을 이용한 모델 튜닝

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score





data = pd.read_csv('./train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
Y_train = data['login']
X_data = X_train
Y_data = Y_train






# 예제 데이터 로드 및 분할
# X, y = load_your_data()
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# 초기 하이퍼파라미터 설정
population_size = 10
n_generations = 200
mutation_rate = 0.1

# 초기 모델 인구 생성
population = [{
    'n_estimators': np.random.choice([10, 40, 100]),
    'min_impurity_decrease': np.random.uniform(0.0001, 0.01),
    'max_depth': np.random.randint(10, 40),
    'min_samples_split': np.random.randint(2, 20),
    'min_samples_leaf': np.random.randint(1, 20)
} for _ in range(population_size)]

def mutate(param):
    # 간단한 변이 로직 구현
    if np.random.rand() < mutation_rate:
        return np.random.randint(1, 20)
    return param

for generation in range(n_generations):
    print(f"Generation {generation + 1}/{n_generations}")
    
    scores = []
    for params in population:
        # 모델 학습 및 평가
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, Y_train)
        predictions = rf.predict_proba(X_test)[:, 1]
        score = roc_auc_score(Y_test, predictions)
        scores.append(score)
        
    # 가장 성능이 좋은 모델의 파라미터 선택
    best_index = np.argmax(scores)
    best_params = population[best_index]
    
    # 새로운 세대를 생성 (가장 좋은 모델의 파라미터를 기반으로)
    new_population = []
    for _ in range(population_size):
        new_params = copy.deepcopy(best_params)
        new_params['min_samples_leaf'] = mutate(new_params['min_samples_leaf'])
        new_population.append(new_params)
    
    population = new_population
    
    print(f"Best Score in Generation {generation + 1}: {scores[best_index]}")
    print(f"Best Params: {best_params}")

# 최종 결과 출력
print("Final Best Params:", best_params)






submit = pd.read_csv('./sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('./baseline_submit.csv', index=False)
