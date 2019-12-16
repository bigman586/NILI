import numpy as np

import matplotlib.pyplot as plt

plt.rc('font', weight='bold')


def graphBar(names, values):
    plt.xlabel("Machine Learning Algorithm", size=10)
    plt.ylabel("Accuracy Proportion", size=12.5)

    ind = np.arange(len(names))

    plt.xticks(ind, names)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.bar(np.arange(len(values)), values)

    showGraph()



# def graphMatrix(cm, target_names, title, count, cmap=None, normalize=True):
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy
#
#     if cmap is None:
#         cmap = plt.get_cmap('Blues')
#
#     ax = fig2.add_subplot(3, 3, count)
#     ax.set_aspect('equal')
#     # plt.figure(figsize=(8, 6))
#     ax.imshow(cm, interpolation='nearest', cmap=cmap)
#
#     ax.xaxis.set_ticks(np.arange(0, len(devices), 1))
#     ax.set_xticklabels(devices)
#     ax.yaxis.set_ticks(np.arange(0, len(devices), 1))
#     ax.set_yticklabels(devices)
#
#     ax.set_title(title)
#     #plt.colorbar()
#
#     '''if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)'''
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#
#     ax.set_ylabel('True label')
#     ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

def showGraph():
    plt.show()
