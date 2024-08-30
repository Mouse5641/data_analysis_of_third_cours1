import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# def pca_analysis(data):
#     X = np.array([item[1] for item in data]).T
#
#     pca = PCA(n_components=X.shape[1])
#     X_pca = pca.fit_transform(X)
#
#     eigen_vectors = pca.components_.T
#     eigen_values = pca.explained_variance_
#     explained_variance_ratio = pca.explained_variance_ratio_ * 100
#     cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
#
#     # Створення таблиці
#     data_matrix = np.vstack([eigen_vectors.T, eigen_values, explained_variance_ratio, cumulative_variance_ratio])
#     columns = [f'x`{i + 1}' for i in range(eigen_vectors.shape[1])]
#     index = [item[0] for item in data] + ['Власні числа', '% на напрям', 'Накопичений %']
#
#     df = pd.DataFrame(data_matrix, columns=columns, index=index)
#
#     # Виведення таблиці з форматуванням чисел
#     return df.to_string(float_format="{:.2f}".format)


def pca_analysis(X):
    result = []

    data = np.array([item[1] for item in X])
    data = data.T

    data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    corr_matrix = np.corrcoef(data_std, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    data_pca = data_std.dot(eigenvectors)

    corr_matrix_pca = np.corrcoef(data_pca, rowvar=False)

    df_corr_pca = pd.DataFrame(corr_matrix_pca, index=[f'x`{i + 1}' for i in range(data_pca.shape[1])],
                               columns=[f'x`{i + 1}' for i in range(data_pca.shape[1])])

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues) * 100
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    columns = [f'Ознака x{i + 1}' for i in range(len(eigenvalues))]
    index = [f'x`{i + 1}' for i in range(data.shape[1])]
    index.extend(['Власні числа', '% на напрям', 'Накопичений %'])

    data_matrix = np.vstack([eigenvectors.T, eigenvalues, explained_variance_ratio, cumulative_variance_ratio])
    df = pd.DataFrame(data_matrix, columns=columns, index=index)

    df_transposed = df.T

    data_reconstructed = data_pca.dot(eigenvectors.T)

    data_reconstructed = data_reconstructed * np.std(data, axis=0) + np.mean(data, axis=0)

    result.append(f"{df_transposed.to_string(float_format='{:.3f}'.format)}\n\n")
    result.append(f"Кореляційна матриця у новій системі координат\n")
    result.append(f"{df_corr_pca.to_string(float_format='{:.3f}'.format)}\n\n")
    result.append(f"Дані після зворотнього перетворення:\n")
    result.append(f"{pd.DataFrame(data_reconstructed, columns=[f'Ознака x{i + 1}' for i in range(data.shape[1])])}\n")

    # Виведення результатів у форматі таблиці
    return "".join(result)