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
Subject relationships
'''
# copy number

dt_nt1 = dt_nt.iloc[:18038,3:]
dt_tp1 = dt_tp.iloc[:18038,3:]

dt_combine = pd.concat([dt_nt1, dt_tp1.reindex(dt_nt1.index)], axis=1)
pca = PCA(2)
df = pca.fit_transform(dt_combine.T)

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

# gene exp

dt_nt1 = dt_nt.iloc[18045:36076,3:]
dt_tp1 = dt_tp.iloc[18045:36076,3:]

dt_combine = pd.concat([dt_nt1, dt_tp1.reindex(dt_nt1.index)], axis=1)
pca = PCA(2)
df = pca.fit_transform(dt_combine.T)

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

# Methylation

dt_nt1 = dt_nt.iloc[36090:54114,3:]
dt_tp1 = dt_tp.iloc[36090:54114,3:]

dt_combine = pd.concat([dt_nt1, dt_tp1.reindex(dt_nt1.index)], axis=1)
pca = PCA(2)
df = pca.fit_transform(dt_combine.T)

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

