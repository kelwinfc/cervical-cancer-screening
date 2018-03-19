from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from models import MultitaskEmbedding


os.sys.setrecursionlimit(100000)

skf = StratifiedKFold(10, shuffle=False, random_state=42)
df = pd.read_csv('data/cervical_arranged_NORM.csv')

all_procedures = set(['Hinselmann', 'Schiller', 'Citology'])
gt_labels = ['Biopsy']

y = df.Biopsy.values.astype(np.int32)
X = df.drop(gt_labels, 1).as_matrix().astype(np.float64)

cv = 3
models = [
          ('Unsupervised',
           GridSearchCV(MultitaskEmbedding(alpha=0.0,
                                           bypass=False),
                        param_grid={'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        n_jobs=1)
           ),
          ('Semi',
           GridSearchCV(MultitaskEmbedding(alpha=0.01,
                                           bypass=False),
                        param_grid={'alpha': [0.01, 0.1],
                                    'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        n_jobs=-1)
           ),
          ('Sym-Semi',
           GridSearchCV(MultitaskEmbedding(embedding='symmetric',
                                           bypass=False),
                        param_grid={'alpha': [0.01, 0.1],
                                    'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        n_jobs=-1)
           ),
          ('Zero-Semi',
           GridSearchCV(MultitaskEmbedding(embedding='zero',
                                           bypass=False),
                        param_grid={'alpha': [0.01, 0.1],
                                    'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        n_jobs=-1)
           ),
          ]

for model_name, model in models:
    model.fit(X, y)
    model = model.best_estimator_

    manifold = TSNE(n_components=2, random_state=42)
    rep2d = manifold.fit_transform(model.get_hidden_representation(X))

    pos_rep2d = rep2d[y == 1]
    neg_rep2d = rep2d[y == 0]

    fig = plt.figure(figsize=(14, 12))

    plt.scatter(neg_rep2d[:, 0], neg_rep2d[:, 1], color='blue', marker='+',
                label='Negative', s=100)
    plt.scatter(pos_rep2d[:, 0], pos_rep2d[:, 1], color='red', marker='*',
                label='Positive', s=100)
    plt.legend(loc='best')
    plt.xticks([]), plt.yticks([])
    plt.savefig(model_name + '.pdf', bbox_inches='tight')
    plt.clf()
    print model_name
