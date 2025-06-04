import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB # Байес
from sklearn.tree import DecisionTreeClassifier, plot_tree # Дерево решений
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier # Лог. регрессия
from sklearn.neighbors import KNeighborsClassifier # К-ближайших соседей
from sklearn.svm import SVC # Опорные вектора

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from warnings import filterwarnings
filterwarnings('ignore')

sns.set_style('darkgrid')

df = pd.read_csv('heart.csv')
df.drop(index = 163, inplace=True)

plt.figure(figsize=(12,8),dpi=200)
sns.boxplot(data=df.drop(['trestbps', 'chol', 'thalach', 'age', 'target', 'oldpeak', 'ca'], axis=1), fill=False, color='orange', linewidth=1.75,
           flierprops={"marker": "x"})
plt.title('Распределения категориальных признаков', fontsize=20);
plt.show()

plt.figure(figsize=(12,8),dpi=200)
sns.boxplot(data=df.drop(['sex', 'cp', 'fbs', 'restecg', 'target', 'exang', 'slope', 'thal'], axis=1), fill=False, color='orange', linewidth=1.75,
           flierprops={"marker": "x"})
plt.title('Распределения численных признаков', fontsize=20);
plt.show()

plt.figure(figsize=(12,8),dpi=200)
sns.boxplot(data=df.drop(['sex', 'cp', 'fbs', 'restecg', 'target', 'exang', 'slope', 'thal', 'age', 'trestbps', 'chol', 'thalach'], axis=1), fill=False, color='orange', linewidth=1.75,
           flierprops={"marker": "x"})
plt.title('Распределения Oldpeak и Ca', fontsize=20);
plt.show()

for col in ['trestbps', 'chol', 'thalach', 'age', 'oldpeak', 'ca']:
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)][col].mean()
    df[col] = df[col].mask(df[col] > upper_bound, mean)
    df[col] = df[col].mask(df[col] < lower_bound, mean)

plt.figure(figsize=(12,8),dpi=200)
sns.boxplot(data=df.drop(['sex', 'cp', 'fbs', 'restecg', 'target', 'exang', 'slope', 'thal'], axis=1), fill=False, color='orange', linewidth=1.75,
           flierprops={"marker": "x"})
plt.title('Распределения численных признаков после обработки выбросов', fontsize=20);
plt.show()

plt.figure(figsize=(12,6), dpi=200)
sns.countplot(data=df, x='target', palette='Paired');
plt.ylabel('Количество');
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'size':10});
plt.show()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

bayes = GaussianNB()
bayes.fit(X_train_scale, y_train)

print("Априорные вероятности классов:", *bayes.class_prior_)

max_depth = np.linspace(1, 10, 10).astype('int')
min_samples_split = np.linspace(1, 10, 10).astype('int')
ccp_alpha = np.linspace(0, 1, 10)
params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'ccp_alpha': ccp_alpha}

tree = DecisionTreeClassifier(criterion='gini')
grid_tree_model = GridSearchCV(tree, param_grid=params)
grid_tree_model.fit(X_train_scale, y_train)

plt.figure(figsize=(12,6), dpi=200)
plot_tree(grid_tree_model.best_estimator_, feature_names=X_train.columns, filled=True);
plt.title('Визуализация дерева');
plt.show()

l1_ratio = np.linspace(0, 1, 30)
C = np.logspace(-4, 4, num=10)
log_cv_model = LogisticRegressionCV(cv=5, Cs=C,
                                    max_iter=10000, solver='saga', multi_class='ovr', n_jobs=-1)

log_cv_model.fit(X_train_scale, y_train)

print(f'Коэффициенты лог. регрессии: {log_cv_model.coef_}')
print()
print(f'Свободный коэффициент лог. регрессии: {log_cv_model.intercept_}')

coefs = log_cv_model.coef_[0]
columns = X_train.columns
feature_df = pd.DataFrame({'columns': columns, 'coefs': coefs})
feature_df = feature_df.sort_values(by='coefs')

plt.figure(figsize=(12,8), dpi=200)
sns.barplot(x=feature_df['columns'], y=feature_df['coefs'], palette='Paired');
plt.title('Коэффициенты уравнения Логистической регрессии');
plt.show()

p = [1, 2]
metrics = ['minkowsky', 'cosine']
n_neighbors = [3, 4, 5, 6, 7]
params = {'n_neighbors': n_neighbors, 'p': p, 'metric': metrics}

knn_model = KNeighborsClassifier(n_jobs=-1)
grid_knn_model = GridSearchCV(knn_model, param_grid=params)
grid_knn_model.fit(X_train_scale, y_train)

print(f'Количество соседей, которые участвуют в классификации одного образца: {grid_knn_model.estimator.n_neighbors}')
print(f'Лучшие гиперпараметры: {grid_knn_model.best_params_}')

kernel = ['linear', 'rbf']
C = np.linspace(1, 10, 20)
gamma = np.linspace(0.1, 100, 30)
params = {'kernel': kernel, 'C': C, 'gamma': gamma}

svc = SVC()
grid_svc = GridSearchCV(svc, param_grid=params)
grid_svc.fit(X_train_scale, y_train)

print(f'Лучшие гиперпараметры: {grid_svc.best_params_}')

coefs = grid_svc.best_estimator_.coef_[0]
columns = X_train.columns
feature_df = pd.DataFrame({'columns': columns, 'coefs': coefs})
feature_df = feature_df.sort_values(by='coefs')
print(f"Свободный член: {grid_svc.best_estimator_.intercept_[0]:.4f}")

plt.figure(figsize=(12,8), dpi=200)
sns.barplot(x=feature_df['columns'], y=feature_df['coefs'], palette='rocket');
plt.title('Коэффициенты уравнения SVM');
plt.show()


def print_all_metrics(model, X_test, y_test, name):
    preds = model.predict(X_test)

    # Conf matrix
    print('Матрица ошибок:')
    print(confusion_matrix(y_test, preds))
    print()
    print()

    # Display conf matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='rocket', ax=ax)
    plt.grid(False)
    plt.title(f'Матрица ошибок для {name}')
    plt.show()

    # Classification report
    print('Отчёт по классификации:')
    print(classification_report(y_test, preds))
    print()

    # PR-кривая
    PrecisionRecallDisplay.from_estimator(model, X_test_scale, y_test);
    plt.title(f'PR-кривая для {name}')
    plt.show()

    # ROC-кривая
    RocCurveDisplay.from_estimator(model, X_test_scale, y_test);
    plt.title(f'ROC-кривая для {name}')
    plt.show()

print_all_metrics(bayes, X_test_scale, y_test, 'Байеса')
print_all_metrics(grid_tree_model, X_test_scale, y_test, 'дерева решений')
print_all_metrics(log_cv_model, X_test_scale, y_test, 'лог. регрессии')
print_all_metrics(grid_knn_model, X_test_scale, y_test, 'К-ближайших соседей')
print_all_metrics(grid_svc, X_test_scale, y_test, 'SVM')

patient = [[54., 1., 0., 122., 286., 0., 0., 116., 1.,
            3.2, 1., 2., 2.]]


bayes = [0.97, 0.97, 0.97, 0.97, 0]
decision_tree = [0.74, 0.74, 0.74, 0.74, 0]
log_regr = [0.93, 0.93, 0.93, 0.94, 0]
knn = [0.77, 0.78, 0.77, 0.77, 1]
svm = [0.87, 0.87, 0.87, 0.87, 0]
pd.DataFrame(data=[bayes, decision_tree, log_regr, knn, svm],
             columns=['Precision', 'Recall', 'F1-score', 'Accuracy', 'Итоговое предсказание'],
             index=['Байес', 'Дерево решений', 'Логистическая регрессия', 'К-ближайших соседей', 'Метод опорных векторов']
            )
