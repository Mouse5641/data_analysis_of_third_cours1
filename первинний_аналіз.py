import tkinter as tk

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
# from гістограма import to_np_array
import seaborn as sns


# Сортування
def sort_and_remove_duplicates(data):
    sorted_data = sorted(data)

    return sorted_data


def move_every_fourth_element_to_new_line(arr):
    result_str = ""
    for i, num in enumerate(arr, start=1):
        result_str += str(num) + "\t\t\t"
        if i % 3 == 0:
            result_str += '\n'
    return result_str


# Середнє знач
def average(data):
    average_value = 0
    for i in range(len(data)):
        average_value += data[i]
    avr = round((average_value / len(data)), 4)
    return avr


# Cередньоквадратичне
def calcul_mean_square(data, average_value):
    root_mean_sq = 0

    for i in range(len(data)):
        root_mean_sq += (data[i] - average_value) ** 2
    root_mean_sq = round((root_mean_sq / (len(data) - 1)) ** (1 / 2), 4)

    return root_mean_sq


# Коефіцієнт ексцесу
def calcul_kurtosis_coef(data, average_value):
    standard_deviation = 0

    for i in range(len(data)):
        standard_deviation += data[i] ** 2 - average_value ** 2
    standard_deviation = round((standard_deviation / (len(data))) ** (1 / 2), 4)

    shift_kurtosis_coeff = 0
    for i in range(len(data)):
        shift_kurtosis_coeff += (data[i] - average_value) ** 4
    shift_kurtosis_coeff = shift_kurtosis_coeff / (len(data) * (standard_deviation ** 4))

    kurtosis_coef = round(
        (((len(data) ** 2 - 1) / ((len(data) - 2) * (len(data) - 3))) * (
                (shift_kurtosis_coeff - 3) + (6 / (len(data) + 1)))), 3)

    return kurtosis_coef


# Коефіцієнт асиметрії
def calcul_asym_coef(data, average_value):
    standard_deviation = 0

    for i in range(len(data)):
        standard_deviation += data[i] ** 2 - average_value ** 2
    standard_deviation = round((standard_deviation / (len(data))) ** (1 / 2), 4)

    shift_asym_coef = 0

    for i in range(len(data)):
        shift_asym_coef += (data[i] - average_value) ** 3

    sftAssmCf = shift_asym_coef / (len(data) * (standard_deviation ** 3))

    asym_coef = round(((((len(data) * (len(data) - 1)) ** (1 / 2)) * sftAssmCf) / (len(data) - 2)), 4)

    return asym_coef


# Коефіцієнт контрексцесу
def calcul_counterexcess_coef(kurtosis_coef):
    counterexcess_coef = round((1 / ((abs(kurtosis_coef)) ** (1 / 2))), 4)
    return counterexcess_coef


# Коефіцієнт Варіації Пірсона
def calcul_pearson_coef(root_mean_sq, average_value):
    if average_value < 0.0001 and average_value > -0.0001:
        return None
    elif average_value == 0:
        return None

    pearson_coef = round((root_mean_sq / average_value), 4)
    return pearson_coef


# Виведення даних
def derivation_values(data):
    list_value = ""
    n = len(data)

    average_value = average(data)
    list_value = list_value + f"Середнє значення: \t\t\t{average_value} Середньокв. інтерр: {round((average_value / np.sqrt(n)), 4)}"

    root_mean_sq = calcul_mean_square(data, average_value)
    list_value = list_value + f"\nСередньоквадратичне:   \t\t{root_mean_sq} Середньокв. інтерр: {round(average_value * (2 / (n - 1)) ** (1 / 4), 4)}"

    kurtosis_coef = calcul_kurtosis_coef(data, average_value)
    list_value = list_value + f"\nКоефіцієнт ексцесу:     \t\t{kurtosis_coef} Середньокв. інтерр: {round((24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** 0.5, 4)}"

    asym_coef = calcul_asym_coef(data, average_value)
    list_value = list_value + f"\nКоефіцієнт асиметрії: \t\t{asym_coef} Середньокв. інтерр: {round((6 * (n - 2) / ((n + 1) * (n + 3))) ** 0.5, 4)}"

    counterexcess_coef = calcul_counterexcess_coef(kurtosis_coef)
    shftSq = 0

    for i in range(n):
        shftSq += data[i] ** 2 - average_value ** 2
    shftSq = round((shftSq / n) ** 0.5, 4)

    shftExCf = 0
    for i in range(n):
        shftExCf += (data[i] - average_value) ** 4
    shftExCf = shftExCf / (n * (shftSq ** 4))
    list_value = list_value + f"\nКоефіцієнт контрексцесу:\t\t{counterexcess_coef} Середньокв. інтерр: {round(((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)}"

    pearson_coef = calcul_pearson_coef(root_mean_sq, average_value)
    list_value = list_value + f"\nКоефіцієнт Варіації:  \t\t{pearson_coef}"

    return list_value


# Вилучкння аномальних значень
def removeAnomalous(data):
    n = len(data)

    arr = []
    average_value = average(data)
    mean_square = calcul_mean_square(data, average_value)
    kurtosis_coef = calcul_kurtosis_coef(data, average_value)
    counterexcess_coef = calcul_counterexcess_coef(kurtosis_coef)

    t = 1.2 + 3.6 * (1 - counterexcess_coef) * np.log10(n / 10)

    a = average_value - t * mean_square
    b = average_value + t * mean_square

    for i in range(n):
        if data[i] < a or data[i] > b:
            continue
        arr.append(data[i])

    return arr


# Створення гістограми
def create_histogram_for_x(x, classes=None):
    plt.clf()
    if classes:
        b = classes
    else:
        if len(x) < 100:
            b = round((len(x) ** (1 / 2)))
        else:
            b = round((len(x) ** (1 / 3)))

    plt.figure(figsize=(6.46, 6.46))
    plt.hist(x, bins=b, edgecolor='k', weights=np.ones_like(x) / len(x))

    plt.xlabel('Значення')
    plt.ylabel('Частота')
    plt.title('Гістограма для заданих даних')

    plt.savefig("histogram_for_x.png")
    return "histogram_for_x.png"


# Емпірична функція розподілу
def create_distribution_function(data, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.figure(figsize=(6.46, 6.46))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)

    n = len(data)

    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    if classes:
        b = classes
    else:
        if n < 100:
            b = round((n ** (1 / 2)))
        else:
            b = round((n ** (1 / 3)))

    s_y = np.arange(1, n + 1) / n
    ax.scatter(x=data, y=s_y, s=5)
    sns.histplot(data, element="step", fill=False,
                 cumulative=True, stat="density", common_norm=False, bins=b, color='red')

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Функція розподілу')

    plt.savefig("distribution_function.png")
    return "distribution_function.png"


# Логарифмування
def logs(data):
    log_data = []

    if data[0] < 0:
        for i in range(len(data)):
            x = round((np.log10(data[i] + abs(data[0]) + 0.01)), 4)
            log_data.append(x)
    else:
        for i in range(len(data)):
            x = round(np.log10(data[i]), 5)
            log_data.append(x)
    return log_data


# Стандартизація
def standr(data):
    standr_вata = []
    average1 = average(data)
    mean_square = calcul_mean_square(data, average1)

    for i in range(len(data)):
        x = round(((data[i] - average1) / mean_square), 4)
        standr_вata.append(x)

    return standr_вata


def create_new_window_for_x(sample_data):

    for sample_name, sample_info in sample_data.items():
        if sample_info["var"].get() == 1:
            data = sample_info["data"]

    def change_image(data):
        arr = removeAnomalous(data)
        new_image = Image.open(create_histogram_for_x(arr))
        photo = ImageTk.PhotoImage(new_image)

        # Оновлення зображення в Label
        label.config(image=photo)
        label.image = photo

        text_widget1.delete("1.0", tk.END)
        text_widget1.insert(tk.END, derivation_values(arr))

    def classes():
        image = Image.open(create_histogram_for_x(data, int(entry.get())))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=0, y=20)

        image = Image.open(create_distribution_function(data, int(entry.get())))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=660, y=20)

    def logs_data(x):
        data = logs(x)

        image = Image.open(create_histogram_for_x(data))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=0, y=20)

        image = Image.open(create_distribution_function(data))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=660, y=20)

        text_widget.delete('1.0', tk.END)
        text_widget.insert(tk.END, move_every_fourth_element_to_new_line(data))

        text_widget1.delete('1.0', tk.END)
        text_widget1.insert(tk.END, derivation_values(data))

    def standr_data(x):
        data = standr(x)

        image = Image.open(create_histogram_for_x(data))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=0, y=20)

        image = Image.open(create_distribution_function(data))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(new_window, image=photo)
        label.photo = photo
        label.place(x=660, y=20)

        text_widget.delete('1.0', tk.END)
        text_widget.insert(tk.END, move_every_fourth_element_to_new_line(data))

        text_widget1.delete('1.0', tk.END)
        text_widget1.insert(tk.END, derivation_values(data))

    # array_data = to_np_array(data)
    # x = sort_and_remove_duplicates(array_data[:, 0])
    # x = [-8.0, -7.0, 1.0, 4.0, 4.0, 5.0, 9.0]

    print(data)
    new_window = tk.Toplevel()
    new_window.title("Первинний аналіз для х")
    new_window.geometry("1340x1300")

    menubar = tk.Menu(new_window)

    # Створення меню "Файл"
    file_menu1 = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Меню", menu=file_menu1)
    new_window.config(menu=menubar)

    # Створення підменю для "Перетворення"
    transform_menu = tk.Menu(file_menu1)
    file_menu1.add_cascade(label="Перетворення", menu=transform_menu)

    # Додавання варіантів дій до підменю "Перетворення"
    transform_menu.add_command(label="Логарифмування", command=lambda: logs_data(data))
    transform_menu.add_command(label="Стандартизація", command=lambda: standr_data(data))
    transform_menu.add_command(label="Вилучення аномальних значень", command=lambda: change_image(data))

    # Додавання зображення з гістограмою
    image = Image.open(create_histogram_for_x(data))
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(new_window, image=photo)
    label.photo = photo
    label.place(x=0, y=20)

    text_widget = tk.Text(new_window, height=8, width=80)  # Встановлення розмірів
    text_widget.place(x=0, y=690)
    text_widget.insert(tk.END, move_every_fourth_element_to_new_line(data))

    text_widget1 = tk.Text(new_window, height=8, width=80)
    text_widget1.place(x=660, y=690)
    text_widget1.insert(tk.END, derivation_values(data))

    entry = tk.Entry(new_window)
    entry.place(x=0, y=0)

    get_input_button = tk.Button(new_window, text="побудувати", command=classes)
    get_input_button.place(x=130, y=0)

    image = Image.open(create_distribution_function(data))
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(new_window, image=photo)
    label.photo = photo
    label.place(x=660, y=20)
