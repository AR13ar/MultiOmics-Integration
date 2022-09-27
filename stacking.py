import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation

from tqdm import tqdm

path = ".../MG_gene_data/"

t_nt = path + "NT3.sav"
t_tp = path + "TP3.sav"

dt_nt = pd.read_spss(t_nt)
dt_tp = pd.read_spss(t_tp)

'''
Stacking
'''

dt_nt_cnp = np.array(dt_nt.iloc[0:18038, 3:].T)
dt_nt_methy = np.array(dt_nt.iloc[36076:, 3:].T)
dt_nt_geneexp = np.array(dt_nt.iloc[18038:36076, 3:].T)
 
dt_tp_cnp = np.array(dt_tp.iloc[:18038, 3:].T)
dt_tp_methy = np.array(dt_tp.iloc[36097:, 3:].T)
dt_tp_geneexp = np.array(dt_tp.iloc[18045:36083, 3:].T)

df1 = np.hstack([dt_nt_cnp, dt_nt_geneexp, dt_nt_methy])
df2 = np.hstack([dt_tp_cnp, dt_tp_geneexp, dt_tp_methy])


pca = PCA(300)

df11 = np.vstack([df1, df2])


df = pca.fit_transform(df11)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

kmeans = KMeans(2)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
y = np.hstack([np.zeros(39), np.ones(364)])
for i in u_labels:
    plt.scatter(df[label == i, 0] , df[label == i, 1] , label = i)
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score

score = silhouette_score(df, label)
print("score: ", score)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y , label)
print('acc: ', acc)