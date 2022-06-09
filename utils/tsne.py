from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_with_labels(lowDWeights,labels):
    transferDict = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '15', 8: '16', 9: '17', 10: '18',
                    11: '19', 12: '2', 13: '20', 14: '21', 15: '22', 16: '3', 17: '4', 18: '5', 19: '6', 20: '7',
                    21: '8', 22: '9'}

    plt.figure(figsize=(16, 16))
    X,Y = lowDWeights[:,0],lowDWeights[:,1]
    color = ['black','gray','rosybrown','sienna','peachpuff','gold','red','olive','green','darksalmon','lime','steelblue','dodgerblue','blue','saddlebrown',
             'indigo','royalblue','orange','thistle','hotpink','darkorchid','cyan','lightpink']
    label_names = ["Non-ship", "Air Carrier", "Destroyer", "Landing Craft", "Frigate", "Amphibious Transport Dock", "Cruiser", "Tarawa-class Amphibious Assault Ship",
    "Amphibious Assault Ship", "Command Ship", "Submarine", "Medical Ship", "Combat Boat", "Auxiliary Ship", "Container Ship", "Car Carrier", "Hovercraft", "Bulk Carrier",
"Oil Tanker", "Fishing Boat", "Passenger Ship", "Liquefied Gas Ship", "Barge"]
    cal = [0 for i in range(23)]
    for iter in range(3):
        for x,y,s, in zip(X,Y,labels):
            c = color[int(s)]
            if cal[int(transferDict[int(s)])] == 0:
                if int(transferDict[int(s)]) == 0:
                    plt.scatter(0, 0, c=c, s=25, alpha=0.9, edgecolors='w', linewidths=0.8, label=label_names[int(transferDict[int(s)])])
                    cal[int(transferDict[int(s)])] = 1
                else:
                    tmp = 1
                    for i in range(int(transferDict[int(s)])):
                        if cal[int(i)] == 0:
                            tmp = 0
                    if tmp ==1:
                        plt.scatter(0, 0, c=c, s=25, alpha=0.9, edgecolors='w', linewidths=0.8, label=label_names[int(transferDict[int(s)])])
                        cal[int(transferDict[int(s)])] = 1
                    else:
                        plt.scatter(0, 0, c=c, s=25, alpha=0.9, edgecolors='w', linewidths=0.8)
            else:
                plt.scatter(0, 0,c=c,s=25,alpha=0.9,edgecolors='w',linewidths=0.8)
    plt.xlim(np.min(X)-5,np.max(X)+5)
    plt.ylim(np.min(Y)-5,np.max(Y)+5)
    plt.xticks([])
    plt.yticks([])
    plt.legend(ncol=2)
    plt.style.use('Solarize_Light2')
    figpath = os.path.join("tsne_pca.jpg")
    plt.savefig(figpath,dpi=800,bbox_inches='tight')
    plt.show()

def my_tsne(feature,label_tsne):
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=3000)
    feature = np.array(feature)
    feature=np.reshape(feature,(-1,23))
    low_dim_embs = tsne.fit_transform(feature)
    plot_with_labels(low_dim_embs,label_tsne)