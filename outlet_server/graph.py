import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

plt.rc('font', weight='bold')
BAR_YLABEL = 'Accuracy (%)'
BAR_XLABEL = 'Classifier'
FONT_SIZE = 20

def graph_bar(names, values):
    plt.title('General Classifier accuracy', fontsize=FONT_SIZE)

    plt.xlabel(BAR_XLABEL, size=10)
    plt.ylabel(BAR_YLABEL, size=12.5)

    ind = np.arange(len(names))

    plt.xticks(ind, names)
    plt.yticks(np.arange(0, 100, 10))

    plt.bar(np.arange(len(values)), values)

    plt.savefig("Bar graph")
    show_graph()


# def graph_group_bar(names, first_bar, second_bar):
#     N = len(names)
#
#     first_bar = (20, 35, 30, 35, 27)
#     second_bar = (25, 32, 34, 20, 25)
#
#     ind = np.arange(N)
#     width = 0.35
#
#     plt.bar(ind, first_bar, width, label='DT')
#     plt.bar(ind + width, second_bar, width,
#             label='KNN')
#
#     plt.xlabel(BAR_XLABEL)
#     plt.ylabel(BAR_YLABEL)
#     plt.title('General Classifier accuracy')
#
#     plt.xticks(ind + width / 2, names)
#     plt.yticks(np.arange(0, 1.1, 0.1))
#
#     plt.legend(loc='best')
#     plt.savefig("Bar graph")
#
#     show_graph()


def graph_table(table, columns, title):
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])
    table = plt.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')

    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#0D65D8")
        table[(0, i)].get_text().set_color("#ffffff")

    table.auto_set_font_size(False)
    table.set_fontsize(11.5)

    table.scale(1.25, 1.5)
    plt.axis('off')
    plt.title(title, y=.75, fontsize=FONT_SIZE)

    show_graph()


def graph_matrix(cm, title, cmap=None, labels=[], normalize=True, scale=np.arange(1, 12, 1)):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('PuBu')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.xticks(np.arange(0, len(labels), 1), labels)
    plt.yticks(np.arange(0, len(labels), 1), labels)

    plt.title(title)
    plt.colorbar()

    # if scale is not None:
    #     tick_marks = npresentation.arange(len(scale))
    #     plt.xticks(tick_marks, scale, rotation=45)
    #     plt.yticks(tick_marks, scale)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1 if normalize else cm.max() / 0.5

    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.7f}".format(cm[i, j]),
                     horizontalalignment="bottom",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.savefig(title, fontsize=FONT_SIZE)

    show_graph()


def show_graph():
    plt.show()
