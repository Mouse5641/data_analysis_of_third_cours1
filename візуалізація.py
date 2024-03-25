import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates
import pandas as pd
import seaborn as sns


def parallel_coordinat(data):
    plt.clf()

    df = pd.DataFrame(columns=[f'Feature{i+1}' for i in range(len(data))] + ['Category'])

    transposed_data = list(zip(*[sample[1] for sample in data]))
    formatted_samples = [(f"вібірка{i+1}", list(transposed_data[i])) for i in range(len(transposed_data))]

    for sample_name, sample_data in formatted_samples:
        temp_df = pd.DataFrame(np.array(sample_data).reshape(1, -1), columns=[f'Feature{i+1}' for i in range(len(sample_data))])
        temp_df['Category'] = sample_name
        df = pd.concat([df, temp_df], ignore_index=True)

    parallel_coordinates(df, 'Category', colormap='viridis', alpha=0.5)
    plt.legend([])
    plt.savefig("parallel_coordinates.png")

    return "parallel_coordinates.png"


def scatter_plots(data):
    plt.clf()
    df = pd.DataFrame(columns=[f'Feature{i+1}' for i in range(len(data))] + ['Category'])

    transposed_data = list(zip(*[sample[1] for sample in data]))
    formatted_samples = [(f"вібірка{i+1}", list(transposed_data[i])) for i in range(len(transposed_data))]

    for sample_name, sample_data in formatted_samples:
        temp_df = pd.DataFrame(np.array(sample_data).reshape(1, -1), columns=[f'Feature{i+1}' for i in range(len(sample_data))])
        temp_df['Category'] = sample_name
        df = pd.concat([df, temp_df], ignore_index=True)

    # plt.figure(figsize=(6.4, 4.8))
    g = sns.pairplot(df)
    g.fig.set_size_inches(6.4, 4.8)
    plt.legend([])
    plt.savefig("scatter_plots.png")

    return "scatter_plots.png"


def bubble_chart(data):
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
        labels.append(f'Вибірки: {k - 1}, {k}, {k + 1}')

    plt.xlabel('X-ось')
    plt.ylabel('Y-ось')
    plt.title('Метод "бульташкові"')
    plt.colorbar(label='Розмір бульбашки')
    plt.legend(labels, loc='best')
    plt.savefig("bubble_chart.png")

    return "bubble_chart.png"