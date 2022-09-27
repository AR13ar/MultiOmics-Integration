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
import random
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage import util
from torchvision import transforms, models
from torch import optim


from sklearn.model_selection import train_test_split

path = ".../MG_gene_data/"

t_nt = path + "NT3.sav"
t_tp = path + "TP3.sav"

dt_nt = pd.read_spss(t_nt)
dt_tp = pd.read_spss(t_tp)

dt_nt_cnp = np.array(dt_nt.iloc[0:18038, 3:].T)
dt_nt_methy = np.array(dt_nt.iloc[36076:, 3:].T)
dt_nt_geneexp = np.array(dt_nt.iloc[18038:36076, 3:].T)
 
dt_tp_cnp = np.array(dt_tp.iloc[:18038, 3:].T)
dt_tp_methy = np.array(dt_tp.iloc[36097:, 3:].T)
dt_tp_geneexp = np.array(dt_tp.iloc[18045:36083, 3:].T)

def _convt_to_data(dt1, dt2, dt3, indx):
    totl_data = []
    for i in tqdm(range(len(indx))):
        cont_data = []
        for j in range(dt1.shape[1]):
            cont_data.append([ dt1[indx[i]][j], dt2[indx[i]][j], dt3[indx[i]][j] ])    
        totl_data.append([cont_data])
    
    return np.array(totl_data).squeeze()

depth = 403
pca = PCA(depth)


data_cnp = pca.fit_transform(np.vstack([dt_nt_cnp, dt_tp_cnp]))
data_geneexp =  pca.fit_transform(np.vstack([dt_nt_geneexp, dt_tp_geneexp]))
data_methy =  pca.fit_transform(np.vstack([dt_nt_methy, dt_tp_methy]))

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

data_nt_1 = data_cnp[:39]
data_nt_2 = data_geneexp[:39]
data_nt_3 = data_methy[:39]

data_tp_1 = data_cnp[39:]
data_tp_2 = data_geneexp[39:]
data_tp_3 = data_methy[39:]

indx1 = []
indx1 = list(range(39))
for i in range(364 - 39):
    indx1.append(random.randint(0,38))

data_nt = _convt_to_data(data_nt_1, data_nt_2, data_nt_3, indx1)

indx2 = []
indx2 = list(range(364))
data_tp = _convt_to_data(data_tp_1, data_tp_2, data_tp_3, indx2)

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features = 512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features= 400
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features= 400, out_features= 512
        )
        self.decoder_output_layer = nn.Linear(
            in_features= 512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code.view(-1, code.shape[0]*400)

model = AE(input_shape = depth*3)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

X = np.vstack([data_nt, data_tp])
y = np.hstack([np.zeros(39), np.ones(364)])

train_loader = torch.utils.data.DataLoader(X, batch_size= 8, shuffle=True)

test_loader = torch.utils.data.DataLoader(X, batch_size=1, shuffle=False)

epochs = 50
for epoch in tqdm(range(epochs)):
    loss = 0
    for batch_features in train_loader:
        batch_features = batch_features.view(-1, depth*3).float()
        optimizer.zero_grad()

        outputs, latent = model(batch_features)
        
        train_loss = criterion(outputs, batch_features)
        
        train_loss.backward()
        
        optimizer.step()

        loss += train_loss.item()
    
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    #print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

ll = []
for batch_features in test_loader:
    batch_features = batch_features.view(-1, depth*3).float()
    outputs, latent = model(batch_features)
    ll.append(latent.detach().numpy())
ll = np.array(ll).squeeze()
ll_c = np.vstack([ll[:39], ll[364:]])


pca2 = PCA(300)
df = pca2.fit_transform(ll_c)
print(pca2.explained_variance_)
print(pca2.explained_variance_ratio_)
print(pca2.explained_variance_ratio_.cumsum())

kmeans = KMeans(2)

label = kmeans.fit_predict(df)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i,0] , df[label == i,1] , label = i)
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score

score = silhouette_score(df, label)
print("score: ", score)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y , label)
print('acc: ', acc)
