import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates
import pandas as pd
import seaborn as sns


def parallel_coordinat(data, show=None):
    plt.clf()

    df1 = pd.DataFrame(columns=[f'Feature{i+1}' for i in range(len(data))] + ['Category'])

    transposed_data = list(zip(*[sample[1] for sample in data]))
    formatted_samples = [(f"вібірка{i+1}", list(transposed_data[i])) for i in range(len(transposed_data))]

    for sample_name, sample_data in formatted_samples:
        temp_df = pd.DataFrame(np.array(sample_data).reshape(1, -1), columns=[f'Feature{i+1}' for i in range(len(sample_data))])
        temp_df['Category'] = sample_name
        df1 = pd.concat([df1, temp_df], ignore_index=True)

    parallel_coordinates(df1, 'Category', colormap='viridis', alpha=0.5)
    plt.legend([])
    plt.savefig("parallel_coordinates.png")

    if show == 1:
        plt.show()
        return

    return "parallel_coordinates.png"


def scatter_plots(data, show=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))

    df2 = pd.DataFrame(columns=[f'Feature{i+1}' for i in range(len(data))] + ['Category'])

    transposed_data = list(zip(*[sample[1] for sample in data]))
    formatted_samples = [(f"вібірка{i+1}", list(transposed_data[i])) for i in range(len(transposed_data))]

    for sample_name, sample_data in formatted_samples:
        temp_df = pd.DataFrame(np.array(sample_data).reshape(1, -1), columns=[f'Feature{i+1}' for i in range(len(sample_data))])
        temp_df['Category'] = sample_name
        df2 = pd.concat([df2, temp_df], ignore_index=True)

    g = sns.pairplot(df2)
    g.fig.set_size_inches(6.4, 4.8)
    plt.legend([])
    plt.savefig("scatter_plots.png")

    if show == 1:
        plt.show()
        return

    return "scatter_plots.png"


def bubble_chart(data, show=None):
    plt.clf()

    labels = []
    for k in range(2, len(data)):
        x = data[k - 2][1]
        y = data[k - 1][1]
        z = data[k][1]

        size = [0 for _ in range(len(x))]
        for i in range(len(z)):
            size[i] = abs(z[i]) * 10

        plt.scatter(x, y, s=size, color=np.random.rand(3, ), label=f'Вибірка {k + 1}')
        labels.append(f'Ознаки: {k - 1}, {k}, {k + 1}')

    plt.xlabel('X-ось')
    plt.ylabel('Y-ось')
    plt.title('Метод "бульташкові"')
    plt.colorbar(label='Розмір бульбашки')
    plt.legend(labels, loc='best')
    plt.savefig("bubble_chart.png")

    if show == 1:
        plt.show()
        return

    return "bubble_chart.png"