import cv2
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, seaborn as sns, numpy as np, pandas as pd, random
from pylab import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, show, draw, figure, cm

# --- normal --- #
path = r"/content/image_micro_acc_FFT/normal"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
normal = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          normal[file_count,:,:,:] = im
          file_count += 1


# --- cage fault --- #
path = r"/content/image_micro_acc_FFT/cage_fault"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
cage_fault = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          cage_fault[file_count,:,:,:] = im
          file_count += 1


# --- ball fault --- #
path = r"/content/image_micro_acc_FFT/ball_fault"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
ball_fault = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          ball_fault[file_count,:,:,:] = im
          file_count += 1


# --- horizontal --- #
path = r"/content/image_micro_acc_FFT/horizontal-misalignment"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
horizontal = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          horizontal[file_count,:,:,:] = im
          file_count += 1


# --- imbalance --- #
path = r"/content/image_micro_acc_FFT/imbalance"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
imbalance = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          imbalance[file_count,:,:,:] = im
          file_count += 1


# --- outer race --- #
path = r"/content/image_micro_acc_FFT/outer_race"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
outer_race = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          outer_race[file_count,:,:,:] = im
          file_count += 1


# --- vertical --- #
path = r"/content/image_micro_acc_FFT/vertical-misalignment"
dt = 1/50000
file_count = 0
# count = 0

array_len = len(glob.glob(os.path.join(path, "*.jpg")))
vertical = np.zeros((array_len, 128, 128, 3))
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
          im = cv2.imread(os.path.join(root, file))
          im = cv2.resize(im, (128,128))
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          vertical[file_count,:,:,:] = im
          file_count += 1


# ------------------------------------------- #
#                data & target
# ------------------------------------------- #

normal = normal.reshape(normal.shape[0], -1)
imbalance = imbalance.reshape(imbalance.shape[0], -1)
horizontal = horizontal.reshape(horizontal.shape[0], -1)
vertical = vertical.reshape(vertical.shape[0], -1)
ball_fault = ball_fault.reshape(ball_fault.shape[0], -1)
cage_fault = cage_fault.reshape(cage_fault.shape[0], -1)
outer_race = outer_race.reshape(outer_race.shape[0], -1)

normal = normal/255
imbalance = imbalance/255
horizontal = horizontal/255
vertical = vertical/255
ball_fault = ball_fault/255
cage_fault = cage_fault/255
outer_race = outer_race/255

# --- original feature --- #
data_feature = 0
classes = [normal, imbalance, horizontal, vertical, ball_fault, cage_fault, outer_race]
for class_ in classes:
    if type(data_feature) is not int:
        data_feature = np.vstack((data_feature, class_))
    else:
        data_feature = np.vstack((class_))


# --- target --- #
y_label = []
for i in range(1951):
    if i < 49:
        y_label.append("normal")
    elif 49-1 < i and i <  49+333:
        y_label.append("imbalance")
    elif 49+333-1 < i and i < 49+333+197:
        y_label.append("horizontal misalignment")
    elif 49+333+197-1 < i and i < 49+333+197+301:
        y_label.append("vertical misalignment")
    elif 49+333+197+301-1 < i and i < 49+333+197+301+137+186:
        y_label.append("ball fault")
    elif 49+333+197+301+137+186-1 < i and i < 49+333+197+301+137+186+188+188:
        y_label.append("cage fault")
    elif 49+333+197+301+137+186+188+188-1 < i and i < 49+333+197+301+137+186+188+188+188+184:
        y_label.append("outer race")

y_label = np.array(y_label)


# --- data --- #
df = pd.DataFrame(data_feature)
X = data_feature
y = y_label 
df["target"] = y


# ------------------------------------------- #
#                2-D PCA
# ------------------------------------------- #

# --- define number of Principal Compoment --- #
pc_num = 2
col = []
for i in range(pc_num):
    col.append("PC{}".format(i+1))

pca = PCA(n_components=pc_num)
pca_features = pca.fit_transform(X)
pca_df = pd.DataFrame(data=pca_features, columns=col)

pca.explained_variance_

# --- plot 2-D PCA --- #
sns.set()
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='target', 
    fit_reg=False, 
    legend=True
    )
plt.title('2D PCA Graph')
plt.show()


# ------------------------------------------- #
#                3-D PCA
# ------------------------------------------- #

# --- define number of Principal Compoment --- #
pc_num = 3
col = []
for i in range(pc_num):
    col.append("PC{}".format(i+1))

pca = PCA(n_components=pc_num)
pca_features = pca.fit_transform(X)
pca_df = pd.DataFrame(data=pca_features, columns=col)

pca.explained_variance_


# --- define label by number --- #
y_color = []
for class_ in y:
    if class_ == "normal":
        y_color.append(0)
    elif class_ == "imbalance":
        y_color.append(1)
    elif class_ == "horizontal misalignment":
        y_color.append(2)
    elif class_ == "vertical misalignment":
        y_color.append(3)
    elif class_ == "ball fault":
        y_color.append(4)
    elif class_ == "cage fault":
        y_color.append(5)
    elif class_ == "outer race":
        y_color.append(6)


# --- plot 3-D PCA --- #
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(12,12))

ax = Axes3D(fig) # Method 1

xdata = pca_features[:,0]
ydata = pca_features[:,1]
zdata = pca_features[:,2]

g = ax.scatter(xdata, ydata, zdata, c=y_color, marker='o', cmap = sns.color_palette('hls', n_colors=5 ,as_cmap=True))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

legend = ax.legend(*g.legend_elements(), loc="lower center", title="Classes", borderaxespad=-10, ncol=5)
ax.add_artist(legend)

plt.show()





