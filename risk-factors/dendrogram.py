from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os


from models import MultitaskEmbedding


os.sys.setrecursionlimit(100000)


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for
    # plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance,
                                      no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=12, p=50,
               **kwargs)


df = pd.read_csv('data/cervical_arranged_NORM.csv')

gt_labels = ['Biopsy']

y = df.Biopsy.values.astype(np.int32)
df = df.drop(gt_labels, 1)
X = df.as_matrix().astype(np.float64)

model = MultitaskEmbedding(embedding='zero', bypass=False,
                           alpha=0.1, width=10, depth=2)
model.fit(X, y)

domains = [np.unique(X[:, i]) for i in range(X.shape[1])]

impact = np.zeros((X.shape[1], 10))

for ft_orig in range(X.shape[1]):
    emb = model.get_hidden_representation(X)
    for value in domains[ft_orig]:
        Xnew = X.copy()
        Xnew[:, ft_orig] = value
        embnew = model.get_hidden_representation(Xnew)
        impact[ft_orig] = np.maximum(impact[ft_orig],
                                     np.mean(np.abs(emb - embnew), axis=0))

model = AgglomerativeClustering(n_clusters=1).fit(impact)
fig = plt.figure(figsize=(14, 12))
plot_dendrogram(model, labels=df.columns.values)
plt.yticks([])
plt.savefig('dendogram-embedding.pdf', bbox_inches='tight')
plt.clf()
