import pandas as pd
import numpy as np
from pingouin import partial_corr
import math
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import f


def correlation_matrix(selected_samples):
    result = []
    data = np.array([sample[1] for sample in selected_samples])

    mean_values = np.mean(data, axis=1, keepdims=True)
    deviations = data - mean_values

    correlation_matrix_manual = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j:
                correlation_matrix_manual[i, j] = 1.0
            else:
                numerator = np.sum(deviations[i] * deviations[j])
                denominator = np.sqrt(np.sum(deviations[i] ** 2) * np.sum(deviations[j] ** 2))

                correlation_matrix_manual[i, j] = numerator / denominator

    correlation_matrix_df = pd.DataFrame(correlation_matrix_manual,
                                         columns=[f'Вібірка {i + 1}' for i in range(data.shape[0])],
                                         index=[f'Вибірка {i + 1}' for i in range(data.shape[0])])

    result.append("\n\t\t\tКореляційна матриця:")
    result.append(f"{correlation_matrix_df}")

    return "\n".join(result)


def partial(lst1, lst2):
    result = []
    df = pd.DataFrame()

    if len(lst1) == 1 and len(lst2) == 1:
        df['x'] = lst1[0]
        df['z'] = lst2[0]

        lst = ['z']
        partial_corr_result = partial_corr(data=df, x='x', covar=lst)
    elif len(lst1) == 2 and len(lst2) == 2:
        df['x'] = lst1[0]
        df['y'] = lst1[1]

        df['z'] = lst2[0]
        df['o'] = lst2[1]

        lst = ['z', 'o']
        partial_corr_result = partial_corr(data=df, x='x', y='y', covar=lst)
    elif len(lst1) == 1 and len(lst2) == 2:
        df['x'] = lst1[0]

        df['z'] = lst2[0]
        df['o'] = lst2[1]

        lst = ['z', 'o']
        partial_corr_result = partial_corr(data=df, x='x', covar=lst)
    elif len(lst1) == 2 and len(lst2) == 1:
        df['x'] = lst1[0]
        df['y'] = lst1[1]

        df['z'] = lst2[0]

        lst = ['z']
        partial_corr_result = partial_corr(data=df, x='x', y='y', covar=lst)
    else:
        result.append("Не вірна кількість обраних вибірок")
        return result

    r = np.round(partial_corr_result['r']['pearson'], 3)

    result.append(f"Оцінка часткового коефіцієнта кореляції:{r}\n")

    t1 = np.round(((r * np.sqrt(len(lst1[0]) - len(lst) - 1)) / np.sqrt(1 - r ** 2)), 3)
    result.append(f"Статистика для обчислення значущості{t1}\n")

    alpha = 0.05
    df = len(lst1[0]) - 1 - 1
    t_value = np.round(t.ppf(1 - alpha / 2, df), 2)

    result.append(f"Табличне значення: {t_value}\n")

    if t_value > t1:
        result.append("Не є статистично значущим")
    else:
        result.append("Є статистично значущим\n")
        u = norm.ppf(alpha)
        down = np.round((1 / 2 * math.log((1 + r) / (1 - r)) - u / (len(lst1[0]) - len(lst) - 3)), 3)
        up = np.round((1 / 2 * math.log((1 + r) / (1 - r)) + u / (len(lst1[0]) - len(lst) - 3)), 3)
        result.append("Довірчі інтервали:\n")
        result.append(f"{up} < {r} < {down}\n")

    return "".join(result)


def plural(selected_samples):
    result = []

    data_matrix = np.array([feature[1] for feature in selected_samples])

    correlation_matrix = np.corrcoef(data_matrix)

    n = data_matrix.shape[1]  # кількість спостережень
    k = data_matrix.shape[0]  # кількість ознак

    multiple_correlation_coefficients = []

    for i in range(k):
        reduced_matrix = np.delete(np.delete(correlation_matrix, i, axis=0), i, axis=1)

        det_correlation_matrix = np.linalg.det(correlation_matrix)
        det_reduced_matrix = np.linalg.det(reduced_matrix)

        R_k = np.sqrt(1 - (det_correlation_matrix / det_reduced_matrix))
        multiple_correlation_coefficients.append(round(R_k, 4))

    f_statistics = []
    critical_values = []
    for R_k in multiple_correlation_coefficients:
        f_statistic = (n - k - 1) * (R_k ** 2) / (k * (1 - R_k ** 2))
        f_statistics.append(round(f_statistic, 4))

        v1 = k
        v2 = n - k - 1

        critical_value = 1 - f.cdf(f_statistic, v1, v2)
        critical_values.append(round(critical_value, 4))

    for i, R_k in enumerate(multiple_correlation_coefficients):
        result.append(f"\nОзнака {i + 1}:\n")
        result.append(f"Множинний коефіцієнт кореляції: {R_k}\n")
        result.append(f"f-статистика: {f_statistics[i]}\n")
        result.append(f"ймовірність отриманої статистики F: {critical_values[i]}\n")
        result.append("Стохастичний зв'язок наявний" if critical_values[i] < 0.05 else "Стохастичний зв'язок відсутній\n")
        result.append("-------------")

    return "".join(result)