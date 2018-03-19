from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools
import pickle
import copy
import os


from models import MultitaskEmbedding


print(os.sys.getrecursionlimit())
os.sys.setrecursionlimit(100000)

def aps(a, b):
    if not np.any(a == 1):
        return 1.

    ret = metrics.average_precision_score(a, b[:, 1])
    if np.isnan(ret):
        return -np.inf
    return ret


X = pd.read_csv(os.sys.argv[1]).as_matrix()
y = X[:, -1]
X = X[:, : -1]


skf = StratifiedKFold(10, shuffle=False, random_state=42)

cv = 3
models = [('Sup',
           GridSearchCV(MultitaskEmbedding(alpha=1.,
                                           embedding='raw'),
                        param_grid={'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        )),
           ('Semi',
            GridSearchCV(MultitaskEmbedding(alpha=0.01),
                        param_grid={'alpha': [0.01, 0.1],
                                    'depth': [1, 2, 3],
                                    'width': [10, 20],
                                    'bypass': [False, True],
                                    },
                        scoring='average_precision',
                        cv=cv,
                        )),
           ('Sym-Semi',
            GridSearchCV(MultitaskEmbedding(embedding='symmetric'),
                         param_grid={'alpha': [0.01, 0.1],
                                     'depth': [1, 2, 3],
                                     'width': [10, 20],
                                     'bypass': [False, True],
                                     },
                         scoring='average_precision',
                         cv=cv,
                         )),
           ('Zero-Semi',
            GridSearchCV(MultitaskEmbedding(embedding='zero'),
                         param_grid={'alpha': [0.01, 0.1],
                                     'depth': [1, 2, 3],
                                     'width': [10, 20],
                                     'bypass': [False, True],
                                     },
                         scoring='average_precision',
                         cv=cv,
                         ))]

results = {n: [] for n, _ in models}

for fold_id, (train_index, test_index) in enumerate(skf.split(X, y)):
    tr_X = X[train_index]
    tr_y = y[train_index]
    ts_X = X[test_index]
    ts_y = y[test_index]

    for model_name, model in models:
        next_model = make_pipeline(Imputer(), MinMaxScaler(),
                                   copy.deepcopy(model))
        next_model.fit(tr_X, tr_y)

        path = os.path.join('output', 'models', 'other-datasets',
                            str(fold_id))

        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, model_name + '.pckl')

        f = open(path, 'w')
        pickle.dump(next_model, f)
        f.close()

        preds = next_model.predict(ts_X)
        probs = next_model.predict_proba(ts_X)

        if probs.shape[1] == 1:
            probs = np.hstack((1 - probs, probs))

        pos_ratio = np.mean(ts_y == 1)
        sample_weight = pos_ratio * np.ones(ts_y.shape[0])
        sample_weight[ts_y == 1] = 1. - pos_ratio

        probs[:, 1] = np.minimum(1 - 1e-5, np.maximum(1e-5, probs[:, 1]))
        
        results[model_name].append(
            [metrics.average_precision_score(ts_y, probs[:, 1]),
             metrics.brier_score_loss(ts_y, probs[:, 1],
                                      sample_weight=sample_weight),
             metrics.log_loss(ts_y, probs[:, 1])
             ])
        print '%30s' % model_name, \
            ' '.join(['%8.4f' % x for x in np.mean(results[model_name],
                                                    axis=0)]), \
            (next_model.best_params_ if type(next_model) == GridSearchCV
                else '')
    print
print
print
print