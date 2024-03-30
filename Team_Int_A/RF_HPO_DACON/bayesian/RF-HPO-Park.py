import sys
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization

# 훈련 데이터와 레이블 
data = pd.read_csv("../data/train.csv")
X, y = data.iloc[:, 1:-1], data['login']

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical 변수 지정
# ex. criterion, bootstrap
# 실행 시
# python RF-HPO-Park.py gini(entropy, log_loss) 1(0) sqrt(log2, 입력x -> None)
criterion = sys.argv[1] if len(sys.argv) > 1 else 'gini'
bootstrap = bool(sys.argv[2]) if len(sys.argv) > 2 else bool(1)
max_features = sys.argv[3] if len(sys.argv) > 3 else None

# RandomFroestClassifier AUC로 평가 함수
def rf_params(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, bootstrap=bootstrap, criterion=criterion, max_features=max_features):
    model = RandomForestClassifier( n_estimators = int(n_estimators),
                                    criterion = criterion,
                                    max_depth = int(max_depth),
                                    min_samples_split = int(min_samples_split),
                                    min_samples_leaf = min_samples_leaf,
                                    min_weight_fraction_leaf = min_weight_fraction_leaf,
                                    max_features = max_features,
                                    max_leaf_nodes = int(max_leaf_nodes),
                                    min_impurity_decrease = min_impurity_decrease,
                                    bootstrap = bootstrap,
                                    random_state = 42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, y_pred_proba)


# Bounded region of parameter space
pbounds = {
            'n_estimators' : (10, 1000),
            # 'criterion' : ('gini', 'entropy', 'log_loss'),
            'max_depth' : (5, 20),
            'min_samples_split' : (2, 5),
            'min_samples_leaf' : (0.0, 0.5),
            'min_weight_fraction_leaf' : (0.0, 0.5),
            # 'max_features' : ('sqrt', 'log2', 'auto'),
            'max_leaf_nodes' : (32, 2048),
            'min_impurity_decrease' : (0, 0.25)
            # 'bootstrap' : (True, False)
           }

optimizer = BayesianOptimization(
    f=rf_params,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=40,
    n_iter=120,
)

best_params = optimizer.max['params']

# 소수형 int로 변환
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['max_leaf_nodes'] = int(best_params['max_leaf_nodes'])

# 카테고리 인자 딕셔너리에 넣기
best_params['criterion'] = criterion
if max_features:
  best_params['max_features'] = max_features
else:
  best_params['max_features'] = None
best_params['bootstrap'] = bootstrap

cols = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap']
submission = pd.DataFrame(data = [best_params], columns = cols)

path = "../data/"
submission.to_csv(path + f"submission_{criterion}_{bootstrap}_{max_features}.csv")

print()
print('==========================================================================================')
print("Best Parameters")
print()
print(f"'criterion' = {criterion}, 'bootstrap' = {bootstrap}, 'max_features' = {max_features}")
print(best_params)
print('==========================================================================================')
