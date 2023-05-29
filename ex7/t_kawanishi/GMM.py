"""Function about GMM."""
import numpy as np

def gauss(x:float, mean:float, var:float) -> float:
    """gaussian function.

    Args:
        x (float): value
        mean (float): mean
        var (float): variance

    Returns:
        float: value
    """
    return (1/(np.sqrt(2*np.pi)*var)) * np.exp(-(x-mean)**2/(2*var**2))

def gauss_2(x:np.ndarray,mean:np.ndarray,var:np.ndarray) ->np.ndarray:
    """gaussian function 2-variable

    Args:
        x (np.ndarray): x
        mean (np.ndarray): mean
        var (np.ndarray): variance

    Returns:
        np.ndarray: value
    """
    a = 1/(2*np.pi *np.sqrt(np.linalg.det(var)))
    return a * np.exp((-1/2)*(x-mean).T@var.I@(x-mean))

