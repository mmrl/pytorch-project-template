import numpy as np
import pandas as pd
from itertools import chain

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression


def temporal_pca(data, n_components=None, random_state=None, z_score=True):
    if z_score:
        data = data.groupby(['unit']).apply(
            lambda x: (x - x.mean())/x.std(ddof=1))

    X = data.groupby(level=['timestep', 'unit']).mean().values
    X = X.reshape(-1, len(data.index.unique(level='unit')))

    pca = PCA(
        n_components=n_components,
        copy=False, whiten=True,
        random_state=random_state
    ).fit(X=X)

    X_proj = pca.transform(X)

    components = range(1, pca.n_components_ + 1)    

    X_proj = pd.Series(
        name='activation projection',
        data=X_proj.reshape(-1),
        index=pd.MultiIndex.from_product(
            [data.index.unique(level='timestep'), components],
            names=['timestep', 'component']
        )
    )

    data = chain(pca.explained_variance_, pca.explained_variance_ratio_)
    expvar = pd.Series(
        name='value',
        data=list(data),
        index=pd.MultiIndex.from_product(
            [['expvar', 'expvar ratio'], components],
            names=['measurement', 'component']
        )
    )

    return X_proj, expvar, pca


def tsne_projection(data, n_components=2, perplexity=30.0):
    last_tstep = data.groupby(['input', 'unit']).last()
    units = sorted(data.index.unique(level='unit'))
    inputs = sorted(data.index.unique(level='input'))

    X = last_tstep.values.reshape(-1, len(units))
    X_proj = TSNE(n_components, perplexity).fit_transform(X)

    X_proj = pd.DataFrame(
        data=X_proj,
        index=pd.Index(inputs, names='inputs'),
        columns=['component #{}'.format(i + 1) for i in range(n_components)]
    )

    return X_proj
