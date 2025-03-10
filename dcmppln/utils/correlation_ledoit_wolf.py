###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import rbf_kernel

def ledoit_wolf_shrinkage(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Ledoit-Wolf shrinkage to denoise the covariance matrix adaptively.

    Parameters
    ----------
    cov_matrix: np.ndarray
        Input covariance matrix to be denoised.

    Returns
    -------
    shrunk_cov_matrix: np.ndarray
        The covariance matrix after applying Ledoit-Wolf shrinkage.
    """
    lw = LedoitWolf()
    lw.fit(cov_matrix)
    return lw.covariance_

def apply_rbf_kernel(data: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    Apply an RBF kernel to emphasize nonlinear correlations.

    Parameters
    ----------
    data: np.ndarray
        Input data matrix.
    gamma: float, optional
        Kernel coefficient for the RBF kernel. Default is 0.1.

    Returns
    -------
    kernel_matrix: np.ndarray
        The RBF kernel matrix.
    """
    return rbf_kernel(data, gamma=gamma)

def split_covariance_matrices(C: np.ndarray, gamma: float = 0.1):
    """
    Split the covariance matrix into C_Noise, C_Star, and C_Global using improved preprocessing.

    Parameters
    ----------
    C: np.ndarray
        Input covariance matrix.
    gamma: float, optional
        Kernel coefficient for the RBF kernel. Default is 0.1.

    Returns
    -------
    C_Noise: np.ndarray
        A split of the covariance matrix corresponding to noise.
    C_Star: np.ndarray
        A split of the covariance matrix corresponding to structured components.
    C_Global: np.ndarray
        A split of the covariance matrix corresponding to global components.
    """
    # Apply Ledoit-Wolf shrinkage
    C_shrunk = ledoit_wolf_shrinkage(C)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C_shrunk)
    eigenvalues = eigenvalues[::-1]  # Reverse order sorting
    eigenvectors = eigenvectors[:, ::-1]  # Reverse order sorting

    # Split eigenvalues and eigenvectors into regimes based on explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    cumulative_variance = np.cumsum(explained_variance_ratio)
    lambda_global = eigenvalues[0]

    # Noise regime (small eigenvalues)
    noise_threshold = 0.05
    eigenvalues_noise = eigenvalues[cumulative_variance <= noise_threshold]
    eigenvectors_noise = eigenvectors[:, cumulative_variance <= noise_threshold]

    # Structured regime (non-global but above noise)
    eigenvalues_star = eigenvalues[(cumulative_variance > noise_threshold) & (eigenvalues != lambda_global)]
    eigenvectors_star = eigenvectors[:, (cumulative_variance > noise_threshold) & (eigenvalues != lambda_global)]

    # Global regime (largest eigenvalue)
    eigenvalues_global = lambda_global
    eigenvectors_global = eigenvectors[:, 0]

    # Compute the covariance matrices for each regime
    C_Noise = eigenvectors_noise @ np.diag(eigenvalues_noise) @ eigenvectors_noise.T if eigenvalues_noise.size > 0 else np.zeros_like(C)
    C_Star = eigenvectors_star @ np.diag(eigenvalues_star) @ eigenvectors_star.T if eigenvalues_star.size > 0 else np.zeros_like(C)
    C_Global = eigenvectors_global[:, None] * eigenvalues_global * eigenvectors_global[None, :]

    return C_Noise, C_Star, C_Global

def improved_preprocessing(data, gamma=0.1):
    """
    Improved preprocessing with multi-level matrix cleaning.
    Combines shrinkage estimators and Gaussian RBF kernels.

    Parameters:
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    gamma : float
        Kernel parameter for RBF kernel.

    Returns:
    -------
    C_Noise, C_Star, C_Global : tuple of np.ndarray
        Cleaned covariance matrix components.
    """
    # Compute the raw covariance matrix
    raw_cov_matrix = LedoitWolf().fit(data).covariance_

    # Apply the RBF kernel on the transposed data to align dimensions
    kernel_matrix = apply_rbf_kernel(data.T, gamma=gamma)

    # Combine the kernel-enhanced covariance with the raw covariance
    hybrid_cov_matrix = (raw_cov_matrix + kernel_matrix) / 2

    # Split the covariance matrix into noise, structured, and global components
    C_Noise, C_Star, C_Global = split_covariance_matrices(hybrid_cov_matrix, data.shape[0] / data.shape[1])

    return C_Noise, C_Star, C_Global



if __name__=="__main__":
    # Example usage
    data = np.random.randn(100, 50)  # Example data (100 samples, 50 features)
    gamma = 0.1
    cleaned_cov_matrix = improved_preprocessing(data, gamma=gamma)
    print("Cleaned Covariance Matrix:")
    print(cleaned_cov_matrix)
