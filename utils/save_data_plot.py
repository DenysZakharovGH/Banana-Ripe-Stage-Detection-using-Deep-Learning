import numpy as np
from matplotlib import pyplot as plt

from data.data_structure import output_data_structure
from settings import dir_path


def save_data_spread_plot(train_data_labels_np):
    dictionary_labels = {}
    #train_data_labels_np = np.array(train_data_labels)
    unique_rows, counts = np.unique(train_data_labels_np, axis=0, return_counts=True)

    for iter, unique_row in enumerate(unique_rows):
        for key in output_data_structure.keys():
            if list(output_data_structure[key].values()) == unique_row.tolist():
                dictionary_labels[key] = counts[iter]
                break
    fig, ax = plt.subplots(figsize=(20,10))
    labels = dictionary_labels.keys()
    counts = dictionary_labels.values()
    bar_labels = dictionary_labels.keys()

    ax.bar(labels, counts, label=bar_labels )

    ax.set_ylabel('Images amount')
    ax.set_title('Test data spread beetween classes')
    ax.legend(title='Class labels')
    plt.savefig(fr"{dir_path}\docs\data_plot.png", dpi=300, bbox_inches="tight")
#plt.show()