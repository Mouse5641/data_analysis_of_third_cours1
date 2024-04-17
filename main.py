import tkinter
from tkinter import *
from tkinter import ttk
import numpy as np
from tkinter import filedialog as fd
from PIL import Image, ImageTk

from первинний_аналіз import create_new_window_for_x, calcul_mean_square
from візуалізація import parallel_coordinat, scatter_plots, bubble_chart
from кореляція import correlation_matrix

sample_data = {}
sample_data1 = {}
selected_samples = []


def deselect_previous():
    for sample_name, sample_info in sample_data1.items():
        sample_info["var"].set(0)


def select_sample(sample_name):
    deselect_previous()
    sample_data1[sample_name]["var"].set(1)
    create_new_window_for_x(sample_data1)


def open_file():
    window.filename = fd.askopenfilename(initialdir="/", title="Select file", filetypes=[('All Files', '*.*'),
                                                                                         ('Python Files', '*.py'),
                                                                                         ('Text Document', '*.txt'),
                                                                                         ('CSV files', "*.csv")])
    global array

    if window.filename.split('.')[1] == 'txt':
        array = np.loadtxt(window.filename, delimiter=",", dtype='float')
        array = array.flatten().tolist()

    sample_num = len(sample_data) + 1
    sample_name = f"Вибірка {sample_num}"
    sample_var = tkinter.IntVar()
    sample_var1 = tkinter.IntVar()
    sample_data[sample_name] = {"data": array, "var": sample_var}
    sample_data1[sample_name] = {"data": array, "var": sample_var1}

    sample_menu1.add_checkbutton(label=sample_name, variable=sample_var)
    sample_menu.add_checkbutton(label=sample_name, variable=sample_var1, command=lambda: select_sample(sample_name))


class ImageTab(ttk.Frame):
    def __init__(self, master, image_path, **kw):
        super().__init__(master, **kw)
        self.image_path = image_path
        self.label = Label(self)
        self.label.place(x=0, y=50)
        self.load_image()

    def load_image(self):
        image = Image.open(self.image_path)
        image = ImageTk.PhotoImage(image)
        self.label.configure(image=image)
        self.label.image = image


def output():
    global selected_samples

    selected_samples.clear()

    for sample_name, sample_info in sample_data.items():
        if sample_info["var"].get() == 1:
            selected_samples.append((sample_name, sample_info["data"]))

    print("Виділені вибірки:", selected_samples)

    y1 = -10
    for i in range(len(selected_samples)):
        radiobutton1 = Radiobutton(window, text=f"{selected_samples[i][0]}", variable=var, value=f"Option{i+1}")
        radiobutton1.place(x=660, y=y1 + 30)
        y1 += 30

    mean_value = []
    rms_value = []
    text_widget.delete("1.0", END)
    for name, data in selected_samples:
        data_array = np.array(data)
        mean_value.append(round(np.mean(data_array), 4))

        rms_value.append(round(calcul_mean_square(data, np.mean(data_array)), 4))
    content = f"Середнє значення: {mean_value}\n" \
              f"Середньоквадратичне значення: {rms_value}\n"

    for i in range(len(notebook.tabs())):
        notebook.forget(0)

    tab1 = ImageTab(notebook, parallel_coordinat(selected_samples))
    notebook.add(tab1, text="Паралельні координати")

    tab2 = ImageTab(notebook, scatter_plots(selected_samples))
    notebook.add(tab2, text="Матриця діаграм розкиду")

    tab3 = ImageTab(notebook, bubble_chart(selected_samples))
    notebook.add(tab3, text="Бульбашкова діаграма")

    content += f"{correlation_matrix(selected_samples)}"
    text_widget.insert(END, content)


window = Tk()
window.geometry("1480x1300")
window.title("Аналіз даних1")

style = ttk.Style()
style.theme_use('default')
style.configure("TNotebook", background="#F0F0F0")

notebook = ttk.Notebook(window)
notebook.pack(fill=BOTH, expand=True)

# Створення стрічки меню
menubar = Menu(window)

# Створення меню "Файл"
file_menu1 = Menu(menubar, tearoff=0)
file_menu1.add_command(label="Обрати файл", command=open_file)
file_menu1.add_separator()
file_menu1.add_command(label="Вийти", command=window.quit)
sample_menu1 = Menu(menubar, tearoff=0)
file_menu1.add_cascade(label="Вибірки", menu=sample_menu1)
file_menu1.add_command(label="Відобразити", command=output)

menubar.add_cascade(label="Файл", menu=file_menu1)

file_menu2 = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Первинний аналіз", menu=file_menu2)
sample_menu = Menu(menubar, tearoff=0)
file_menu2.add_cascade(label="Вибірки", menu=sample_menu)

window.config(menu=menubar)

text_widget = Text(window, height=13, width=80)  # Встановлення розмірів
text_widget.place(x=0, y=600)

text_widget1 = Text(window, height=13, width=100)
text_widget1.place(x=670, y=600)

var = StringVar()

window.mainloop()


