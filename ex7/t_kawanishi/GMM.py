"""Function about GMM."""
import numpy as np


def gauss(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """gaussian function 2-variable.

    Args:
        x (np.ndarray): x
        mean (np.ndarray): mean
        var (np.ndarray): variance

    Returns:
        np.ndarray: value (n_clusters,data)
    """
    # get data dimension and num of clusters
    dim = x.shape[1]
    n_clusters = mean.shape[0]

    # calculate data - each mean
    # x_m.shape = (n_clusters, data, dim)
    x_m = x - mean[:, np.newaxis]

    # calculate variance-covariance inverse
    # var_I.shape = (n_clusters, dim,dim)
    if dim == 1:
        var_I = 1 / (var**2)
        var_2 = (var**2).reshape(-1)
    else:
        var_I = np.linalg.inv(var)
        var_2 = np.linalg.det(var)

    # compute gaussian
    # N.shape = (n_clusters,data)
    b = np.exp(
        np.array([
            -1 / 2 * x_m[i] @ var_I[i] @ x_m[i].T
            for i in range(n_clusters)
            ])
    )
    a = 1 / np.sqrt(((2 * np.pi) ** dim) * var_2)
    N = a[:, np.newaxis, np.newaxis] * b
    N = np.diagonal(N, axis1=-2, axis2=-1)

    return N


def init(
    data: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate initial variable.

    Args:
        data (np.ndarray): dataset
        n_clusters (int): number of clusters

    Return:
        np.ndarray: weight of each gaussian (n_clusters,)
        np.ndarray: mean of each gaussian (n_clusters,dim)
        np.ndarray: variance of each gaussian (n_clusters,dim,dim)
    """
    dim = data.shape[1]
    weight = np.full(n_clusters, 1 / n_clusters)
    mean = np.random.randn(n_clusters, dim)
    var = np.array([np.identity(dim) for i in range(n_clusters)])
    return weight, mean, var


def update_gamma(
    data: np.ndarray, weight: np.ndarray, mean: np.ndarray, var: np.ndarray
) -> np.ndarray:
    """update gamma.

    Args:
        data (np.ndarray): dataset
        weight (np.ndarray): weight (n_clusters,)
        mean (np.ndarray): mean
        var (np.ndarray): variance

    Returns:
        np.ndarray: new gamma (n_cluster,data)
    """
    N = gauss(data, mean, var)  # (n_clusters, data)
    w_N = N * weight[:, np.newaxis]
    b = np.sum(w_N, axis=0)
    gamma = w_N / b
    return gamma


def new_para(
    data: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate new parameter

    Args:
        data (np.ndarray): data
        gamma (np.ndarray): gamma (n_clusters, data)

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]:
                            new mean, var and weight
    """
    N_k = np.sum(gamma, axis=1)  # (n_clusters,)
    a = (1 / N_k)[:, np.newaxis]  # (n_clusters,1)
    new_mean = a * np.sum(
        gamma[:, :, np.newaxis] * data,
        axis=1
        )  # (n_clusters,dim)
    x_mean = np.expand_dims(
        data - new_mean[:, np.newaxis], axis=-1
    )  # (n_clusters, data,dim,1)
    x_mean_T = x_mean.transpose(0, 1, 3, 2)
    b = np.sum(
        gamma[:, :, np.newaxis, np.newaxis] * (x_mean @ x_mean_T),
        axis=1
        )
    new_var = a.reshape(a.shape[0], 1, 1) * b
    new_weight = N_k / np.sum(N_k)
    return new_weight, new_mean, new_var


def log_likelihood(
    data: np.ndarray, weight: np.ndarray, mean: np.ndarray, var: np.ndarray
) -> float:
    """log-likelihood function.

    Args:
        data (np.ndarray): dataset
        weight (np.ndarray): weight
        mean (np.ndarray): mean
        var (np.ndarray): variance

    Returns:
        float: log-likelihood
    """
    N = gauss(data, mean, var)
    log_like = np.sum(np.log(np.sum(weight[:, np.newaxis] * N, axis=0)))
    return log_like


if __name__ == "__main__":
    pass
