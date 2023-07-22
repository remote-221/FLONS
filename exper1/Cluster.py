#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: zs
@Description: 描述
@time: 2022/10/15 14:25
@version: 1.0
"""

from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
from sklearn import manifold,datasets
from sklearn.decomposition import PCA

def cluster(data):

    n_components = 200
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=30)
    result = tsne.fit_transform(data)

    dbscan = DBSCAN(eps=20, min_samples=5)
    labels = dbscan.fit_predict(result)

    d = defaultdict(list)
    for i, x in enumerate(labels):
        d[x].append(i)

    result = list(d.values())
    for i in result:
        print(i)
