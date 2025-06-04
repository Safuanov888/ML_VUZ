import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.DataFrame({'X': [2, 3, 4], 'Y': [3, 5, 10]})

print(df_wine.head())

plt.scatter(df_wine['X'], df_wine['Y'])
plt.title('Исходные данные')
plt.grid(True)
plt.show()

df_wine['X'] = df_wine['X'] - df_wine['X'].mean()
df_wine['Y'] = df_wine['Y'] - df_wine['Y'].mean()

print('Нормализованные данные')
print(df_wine.head())

cov_mat = np.cov(df_wine.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('Ковариационная матрица:')
print(pd.DataFrame(cov_mat))

print('Собственные значения \n', np.round(eigen_vals, 2))

print(f'Собственные вектора \n {np.round(eigen_vecs, 2)}')

# Создаём списки (собственное значение, собственный вектор)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Сортируем список по убыванию собственного значения
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

print(f'Отсортированные пары собственных значений и собственных векторов:\n {eigen_pairs, 2}')

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Матрица W:\n', np.round(w, 2))

X_train_pca = df_wine.dot(w)
print('Матрица с компонентами')
print(np.round(X_train_pca, 2))

plt.scatter(X_train_pca[0], X_train_pca[1])

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.grid(True)
# plt.savefig('figures/05_03.png', dpi=300)
plt.title('Преобразованные данные')
plt.show()
