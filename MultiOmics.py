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
Combined clustering with formulas
'''

''' Formula 1 : 0.317*cnp + 3.34*methylation for subject level clustering and accuracy 
'''


dt_nt_cnp = dt_nt.iloc[0:18038, 3:]
dt_nt_methy = dt_nt.iloc[36076:, 3:]
dt_nt_geneexp = dt_nt.iloc[18038:36076, 3:]

combined_dt_nt = 0.317*dt_nt_cnp.to_numpy() + 3.34*dt_nt_methy.to_numpy()

dt_tp_cnp = dt_tp.iloc[:18045, 3:]
dt_tp_methy = dt_tp.iloc[36090:, 3:]
dt_tp_geneexp = dt_tp.iloc[18045:36090, 3:]


combined_dt_tp = 0.317*dt_tp_cnp.to_numpy() + 3.34*dt_tp_methy.to_numpy()

dt_combine = np.vstack([combined_dt_nt.T, combined_dt_tp.T[:, :18038] ])

pca = PCA(2)

df =  pca.fit_transform(dt_combine) 

print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

actual_label = np.vstack([np.ones((39,1)), np.zeros((364,1))]).squeeze()

from sklearn.metrics import accuracy_score
acc = accuracy_score(actual_label,label)

from sklearn.metrics import silhouette_score

score = silhouette_score(df, label)
print('score: ', score)
print('acc : ', acc)


''' Formula 2 : 1/ 1 + exp(0.997*cnp + 61.959*methylation) for subject level clustering and accuracy 
'''


import math

dt_nt_cnp = dt_nt.iloc[0:18038, 3:]
dt_nt_methy = dt_nt.iloc[36076:, 3:]
dt_nt_geneexp = dt_nt.iloc[18038:36076, 3:]

combined_dt_nt_cont = 0.997*dt_nt_cnp.to_numpy() + 61.959*dt_nt_methy.to_numpy()

combined_dt_nt = np.zeros((18038, 39))
for i in range(18038):
    for j in range(39):
        combined_dt_nt[i][j] = 1 / (1 + math.exp(-1*combined_dt_nt_cont[i][j])) 
 
dt_tp_cnp = dt_tp.iloc[:18045, 3:]
dt_tp_methy = dt_tp.iloc[36090:, 3:]
dt_tp_geneexp = dt_tp.iloc[18045:36090, 3:]


combined_dt_tp_cont = 0.997*dt_tp_cnp.to_numpy() + 61.959*dt_tp_methy.to_numpy()

combined_dt_tp = np.zeros((18045, 364))
for i in range(18045):
    for j in range(364):
        combined_dt_tp[i][j] = 1 / (1 + math.exp(-1*combined_dt_tp_cont[i][j])) 

dt_combine = np.vstack([combined_dt_nt.T, combined_dt_tp.T[:, :18038] ])
pca = PCA(2)

df = pca.fit_transform(dt_combine) 
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

actual_label = np.vstack([np.zeros((39,1)), np.ones((364,1))]).squeeze()

from sklearn.metrics import accuracy_score
acc = accuracy_score(actual_label,label)

from sklearn.metrics import silhouette_score

score = silhouette_score(df, label)
print('score: ', score)
print('acc : ', acc)


''' Formula 3 : 0.379*geneexp + 3.756*methylation for subject level clustering and accuracy 
'''

dt_nt_cnp = dt_nt.iloc[18038:36076, 3:]
dt_nt_methy = dt_nt.iloc[36076:54114, 3:]
dt_nt_geneexp = dt_nt.iloc[18038:36076, 3:]

combined_dt_nt = 0.379*dt_nt_geneexp.to_numpy() + 3.756*dt_nt_methy.to_numpy()

dt_tp_cnp = dt_tp.iloc[18045:36090, 3:]
dt_tp_methy = dt_tp.iloc[36090:54135, 3:]


combined_dt_tp = 0.379*dt_tp_geneexp.to_numpy() + 3.756*dt_tp_methy.to_numpy()

dt_combine = np.vstack([combined_dt_nt.T, combined_dt_tp.T[:, :18038] ])

pca = PCA(2)

df = pca.fit_transform(dt_combine) 

kmeans = KMeans(2)
kmeans.fit(df)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

actual_label = np.vstack([np.zeros((39,1)), np.ones((364,1))]).squeeze()

from sklearn.metrics import accuracy_score
accuracy_score(actual_label,label)