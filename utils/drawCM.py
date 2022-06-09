from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def drawCM(pred, true, save_path):
    C = confusion_matrix(true, pred, labels=[str(x) for x in range(23)], normalize="true")
    plt.figure(figsize=(12, 12))
    plt.imshow(C, interpolation='nearest', cmap=plt.cm.viridis)

    for i in range(len(C)):
        for j in range(len(C)):
            if C[j, i] == 0:
                continue
            plt.annotate(C[j, i]*100//1/100, xy=(i, j), horizontalalignment='center', verticalalignment='center', color="black" if C[j, i] > (C.max() / 1.5) else "white")

    plt.tick_params(labelsize=15)

    xlocations = np.array(range(23))
    plt.xticks(xlocations, xlocations)
    plt.yticks(xlocations, xlocations)

    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.tight_layout()
    plt.savefig(save_path)
