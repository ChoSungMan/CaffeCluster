import sys
import os
from shutil import copyfile

from img_to_vec import Img2Vec
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from extract_feature import get_vector

input_path = '/test_images'
files = os.listdir(input_path)

vec_length = 1024

samples = len(files)
k_value = 3

vec_mat = np.zeros((samples, vec_length))
sample_indices = np.random.choice(range(0,len(files)), size=samples, replace=False)

print('Reading Images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    vec = get_vector(os.path.join(input_path, filename))
    vec_mat[index, :] = vec

print('Applying PCA...')
reduced_data = PCA(n_components=2).fit_transform(vec_mat)
kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10)
kmeans.fit(reduced_data)

for i in set(kmeans.labels_):
    try:
        os.mkdir('./'+str(i))
    except FileExistsError:
        continue

print('Predicting...')
preds = kmeans.predict(reduced_data)

print('Copying images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    copyfile(input_path + '/' + filename, './' + str(preds[index]) + '/' + filename)

print('Done')

