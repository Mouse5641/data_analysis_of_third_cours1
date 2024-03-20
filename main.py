import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import json
from tkinter import filedialog as fd

sample_data = {}


def to_np_array(data_string):
    if data_string:
        data_array = np.array(json.loads(data_string))
        return data_array
    else:
        return None


# def open_file():
#     global content
#     global dadas
#     file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
#     if file_path:
#         with open(file_path, 'r') as file:
#             content = file.read()
#             print(content)
#             content = to_np_array(content)
#             dadas.extend(content)
#             print(dadas)

def open_file():
    window.filename = fd.askopenfilename(initialdir="/", title="Select file", filetypes=[('All Files', '*.*'),
                                                                                         ('Python Files', '*.py'),
                                                                                         ('Text Document', '*.txt'),
                                                                                         ('CSV files', "*.csv")])
    global array

    if window.filename.split('.')[1] == 'txt':
        array = np.loadtxt(window.filename, delimiter=",", dtype='float')
        array = array.flatten().tolist()
        print(array)

    sample_num = len(sample_data) + 1
    sample_name = f"Вибірка {sample_num}"
    sample_var = tkinter.IntVar()
    sample_data[sample_name] = {"data": array, "var": sample_var}

    sample_menu.add_checkbutton(label=sample_name, variable=sample_var)
    arr = np.sort(sample_data[sample_name]["data"])
    # print(sample_data[sample_name]["data"])
    # print(arr)
    # print(sample_data[sample_name]["data"])


window = Tk()
window.geometry("1480x1300")
window.title("Аналіз даних1")

# Створення стрічки меню
menubar = Menu(window)

# Створення меню "Файл"
file_menu1 = Menu(menubar, tearoff=0)
file_menu1.add_command(label="Обрати файл", command=open_file)
file_menu1.add_separator()
file_menu1.add_command(label="Вийти", command=window.quit)
sample_menu = Menu(menubar, tearoff=0)
file_menu1.add_cascade(label="Вибірки", menu=sample_menu)

menubar.add_cascade(label="Файл", menu=file_menu1)

window.config(menu=menubar)

window.mainloop()
