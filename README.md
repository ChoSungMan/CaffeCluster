# Caffe Cluster

Pytorch Caffe Cluster 

## Training Phase

1. Data Gathering [X]

cafe image data crawling. (cafe, 카페, カフェ, コーヒーショップ, 咖啡店)

<pre><code> python crawler.py </code></pre>

2. Corrupted Image Deleting [ ]

corrupted image may occur some error. so, we have to delete it

<pre><code> python find_corrupt.py --f 'your data folder' </code></pre>

3. Noise Deleting [ ]

using object detection algorithm.

if there is no chair, delete file.

4. Training Network using MobileNet v2 [ ]

softmax training. 

## Test Phase

1. Detect Object & Feature Extract[ ]

detect obejcts in input image and extract input image's feature. (parallel process)

2. Crawling [ ]

use the name of objects as a crawling keyword.

3. Calculate Similiarity [ ]

calculate the cosine similarity between the input image and the crawling images.

