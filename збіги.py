import numpy as np
from scipy.stats import chi2


def check_mean_equality(data1, data2, regularization=1e-6):
    result = []
    data_samples = [data1, data2]
    k = len(data_samples)  # кількість вибірок
    n = len(data_samples[0])  # кількість змінних (вимірів)

    # Кількість спостережень у кожній вибірці
    N_d = np.array([len(sample[1]) for sample in data_samples[0]])

    # Обчислення середніх векторів для кожної вибірки
    mean_vectors = []
    for data in data_samples:
        mean_vectors.append(np.array([np.mean(feature[1]) for feature in data]))

    # Обчислення дисперсійно-коваріаційних матриць для кожної вибірки
    covariance_matrices = []
    for i, data in enumerate(data_samples):
        deviations = np.array([feature[1] for feature in data]).T - mean_vectors[i]
        covariance_matrix = np.dot(deviations.T, deviations) / (N_d[i] - 1)
        covariance_matrix += np.eye(n) * regularization  # Додавання регуляризації
        covariance_matrices.append(covariance_matrix)

    # Обчислення узагальненого вибіркового середнього
    inv_cov_sum = sum(N_d[d] * np.linalg.inv(covariance_matrices[d]) for d in range(k))
    inv_cov_sum_inv = np.linalg.inv(inv_cov_sum)
    overall_mean = np.zeros(n)
    for d in range(k):
        overall_mean += N_d[d] * np.dot(np.linalg.inv(covariance_matrices[d]), mean_vectors[d])
    overall_mean = np.dot(inv_cov_sum_inv, overall_mean)

    # Обчислення статистики V
    V = 0
    for d in range(k):
        diff = mean_vectors[d] - overall_mean
        V += N_d[d] * np.dot(np.dot(diff.T, np.linalg.inv(covariance_matrices[d])), diff)

    # Критичне значення для chi-square розподілу
    degrees_of_freedom = n * (k - 1)
    critical_value = chi2.ppf(0.95, degrees_of_freedom)

    # Виведення результатів
    if V <= critical_value:
        result.append(f"Приймаємо нульову гіпотезу. V = {V:.4f} <= {critical_value:.4f}\n\n")
    else:
        result.append(f"Відхиляємо нульову гіпотезу. V = {V:.4f} > {critical_value:.4f}\n\n")

    return "".join(result)


def compute_covariance_matrices(samples):
    # Розрахунок ковариаційних матриць для кожної вибірки
    covariance_matrices = []
    for sample in samples:
        # Отримання числових даних для кожної ознаки
        data_matrix = np.array([feature_data for _, feature_data in sample])
        mean_vector = np.mean(data_matrix, axis=1)
        deviations = data_matrix - mean_vector[:, np.newaxis]
        covariance_matrix = (deviations @ deviations.T) / (data_matrix.shape[1] - 1)
        covariance_matrices.append(covariance_matrix)
    return covariance_matrices


def generalized_covariance_matrix(cov_matrices, N_d, k):
    # Обчислення узагальненої ДК-матриці S
    N = sum(N_d)
    generalized_S = sum((N_d[d] - 1) * cov_matrices[d] for d in range(k)) / (N - k)
    return generalized_S


def test_covariance_equality(data1, data2, alpha=0.05):
    result = []

    data = [data1, data2]

    k = len(data)  # Кількість вибірок
    n = len(data[0][1])  # Розмірність ознак
    N_d = [len(sample[0][1]) for sample in data]  # Кількість спостережень в кожній вибірці

    # Отримуємо ковариаційні матриці
    covariance_matrices = compute_covariance_matrices(data)

    # Узагальнена ДК-матриця
    generalized_S = generalized_covariance_matrix(covariance_matrices, N_d, k)

    # Обчислення статистики V
    V = 0
    for d in range(k):
        det_S_d = np.linalg.det(covariance_matrices[d])
        det_general_S = np.linalg.det(generalized_S)
        term = (N_d[d] - 1) * np.log(det_general_S / det_S_d)
        V += term / 2

    # Обчислення числа ступенів свободи
    v = n * (n + 1) * (k - 1) / 2

    # Критичне значення для χ²-розподілу
    chi_square_critical = chi2.ppf(1 - alpha, df=int(v))

    # Перевірка гіпотези
    if V <= chi_square_critical:
        result.append(f"Приймаємо нульову гіпотезу H0: V = {V:.4f} <= {chi_square_critical:.4f}.")
    else:
        result.append(f"Відхиляємо нульову гіпотезу H1: V = {V:.4f} > {chi_square_critical:.4f}.")

    return "".join(result)