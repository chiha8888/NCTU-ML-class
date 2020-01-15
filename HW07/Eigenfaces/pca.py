import numpy as np


def pca(X,num_dim=None):
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_center = X - X_mean

    # PCA
    eigenvalues, eigenvectors = np.linalg.eig(X_center.T @ X_center)
    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index], np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index=sort_index[:num_dim]

    eigenvalues=eigenvalues[sort_index]
    # from X.T@X eigenvector to X@X.T eigenvector
    eigenvectors=X_center@eigenvectors[:, sort_index]
    eigenvectors_norm=np.linalg.norm(eigenvectors,axis=0)
    eigenvectors=eigenvectors/eigenvectors_norm

    return eigenvalues,eigenvectors,X_mean