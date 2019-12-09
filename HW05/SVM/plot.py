import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix,log2c,log2g):
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.set_xticklabels([''] + log2g)
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels([''] + log2c)
    for i in range(len(log2c)):
        for j in range(len(log2g)):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('lg(G)')
    ax.set_ylabel('lg(C)')
    plt.show()