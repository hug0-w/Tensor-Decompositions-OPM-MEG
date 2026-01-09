import numpy as np

def calculate_mse(original, reconstructed):
    """
    Calculates the Mean Squared Error (MSE) between two arrays.

    Parameters
    ----------
    original : np.ndarray
        The original array.
    reconstructed : np.ndarray
        The reconstructed array.

    Returns
    -------
    float
        The Mean Squared Error.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed arrays must have the same shape.")

    mse = np.mean((original - reconstructed) ** 2)
    return mse


def relative_error(original, reconstructed):
    


    f_norm = np.linalg.norm(original)
    r_norm = np.linalg.norm(original - reconstructed)

    relative_error = r_norm / f_norm

    return relative_error


def variance_explained(original, reconstructed):
     pass