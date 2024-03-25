import pandas as pd
import numpy as np


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

    result.append("\nКореляційна матриця:")
    result.append(f"{correlation_matrix_df}")

    return "\n".join(result)


