import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
from pylab import rcParams
from sklearn.decomposition import PCA
import os
from matplotlib.image import imread

#Converison of image to array
r_d = os.path.abspath(r'F:\Acads\SEM 9\PCA')
image = imread(os.path.join(r_d,'PCA.png'))
img_arr = np.array(image)
fig_2 = plt.figure(figsize=(7,7))
plt.imshow(image)


#Applying pca
from sklearn.preprocessing import normalize
X_norm = normalize(img_arr)
pca = PCA(n_components=50)
ldd = pca.fit_transform(X_norm)
ldd.shape

#Reconstructing image
app = pca.inverse_transform(ldd)
fig = plt.figure(figsize=(7,7))
plt.imshow(app)


