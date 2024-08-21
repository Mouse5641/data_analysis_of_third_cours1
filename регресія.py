import numpy as np
from scipy.stats import f
from scipy.stats import t


def check_regression_significance(x, y):
    result = []

    X = np.array([feature[1] for feature in x]).T
    Y = np.array(y[0][1])

    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ Y

    Y_hat = X_with_intercept @ beta_hat

    SS_total = np.sum((Y - np.mean(Y)) ** 2)
    SS_residual = np.sum((Y - Y_hat) ** 2)
    R_squared = 1 - (SS_residual / SS_total)

    result.append(f"Коефіцієнт детермінації R²: {round(R_squared, 4)}\n")

    n = X.shape[1]
    N = X.shape[0]

    v1 = n
    v2 = N - n - 1
    alpha = 0.05

    F_calculated = (R_squared / (1 - R_squared)) * ((N - n - 1) / n)

    F_critical = f.ppf(1 - alpha, v1, v2)

    if F_calculated > F_critical:
        result.append(f"Регресія значуща. {round(F_calculated, 4)} > {round(F_critical, 4)}\n")
    else:
        result.append(f"Регресія не значуща. {round(F_calculated, 4)} < {round(F_critical, 4)}\n")

    return "".join(result)


def calculate_regression_coefficients(x, y):
    X = np.array([feature[1] for feature in x]).T
    Y = np.array(y[0][1])

    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y)

    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    A_hat = np.linalg.inv(X_centered.T @ X_centered) @ X_centered.T @ Y_centered
    a0 = round(Y_mean - np.sum(A_hat * X_mean), 4)

    return a0, np.round(A_hat, 4)


def check_regression_coefficients_significance(x, y):
    result = []

    a0, A_hat = calculate_regression_coefficients(x, y)

    X = np.array([feature[1] for feature in x]).T
    Y = np.array(y[0][1])

    result.append(f"Оцінка a0: {a0}")
    result.append(f"Оцінки коефіцієнтів регресії: {A_hat}\n")

    N = X.shape[0]
    n = X.shape[1]

    Y_hat = a0 + X @ A_hat

    residuals = Y - Y_hat
    s_squared = np.sum(residuals ** 2) / (N - n)

    t_values = A_hat / (np.sqrt(s_squared * np.diag(np.linalg.inv(X.T @ X))))

    v = N - n
    alpha = 0.05

    t_critical = t.ppf(1 - alpha / 2, df=v)

    for i, t_value in enumerate(t_values):
        if abs(t_value) > t_critical:
            result.append(
                f"Коефіцієнт a_{i + 1} значущий: t = {round(t_value, 4)}, t_critical = {round(t_critical, 4)}")
        else:
            result.append(
                f"Коефіцієнт a_{i + 1} не значущий: t = {round(t_value, 4)}, t_critical = {round(t_critical, 4)}")

    return "\n".join(result)


def calculate_confidence_intervals(x, y, alpha=0.05):
    result = []
    a0, A_hat = calculate_regression_coefficients(x, y)

    X = np.array([feature[1] for feature in x]).T
    Y = np.array(y[0][1])

    N, n = X.shape

    Y_hat = a0 + X @ A_hat

    # Обчислення залишків
    residuals = Y - Y_hat
    s_squared = np.sum(residuals ** 2) / (N - n)

    # Ковариаційна матриця для оцінок параметрів
    C = np.linalg.inv(X.T @ X)

    t_critical = t.ppf(1 - alpha / 2, df=N - n)

    intervals = []
    for i in range(n):
        # Дисперсія для кожної оцінки параметра
        variance_a_hat_k = s_squared * C[i, i]

        lower_bound = A_hat[i] - t_critical * np.sqrt(variance_a_hat_k)
        upper_bound = A_hat[i] + t_critical * np.sqrt(variance_a_hat_k)

        intervals.append((lower_bound, upper_bound))

    result.append(f"\n\nІнтервальна оцінка параметрів:")
    for i, interval in enumerate(intervals):
        result.append(f"a_{i + 1}: {round(interval[0], 4)} <= a_{i + 1} <= {round(interval[1], 4)}")

    return "\n".join(result)


def calculate_standardized_coefficients(x, y):
    result = []

    a0, A_hat = calculate_regression_coefficients(x, y)

    # Обчислюємо стандартні відхилення для кожної ознаки та цільової змінної
    X = np.array([feature[1] for feature in x]).T
    Y = np.array(y[0][1])
    X_std = np.std(X, axis=0, ddof=1)
    Y_std = np.std(Y, ddof=1)

    # Обчислюємо стандартизовані коефіцієнти
    A_star = A_hat * (X_std / Y_std)

    result.append(f"\n\nСтандартизовані оцінки параметрів регресії: {np.round(A_star, 4)}")

    return "\n".join(result)

