###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
import scipy.stats as stats

def bayesian_pca(data: np.ndarray, n_components: int, max_iter: int = 100, tol: float = 1e-4) -> tuple:
    """
    Perform Bayesian PCA using variational inference to estimate principal components and noise variance.

    Parameters
    ----------
    data: np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components: int
        Number of principal components to estimate.
    max_iter: int, optional
        Maximum number of iterations for convergence. Default is 100.
    tol: float, optional
        Convergence tolerance. Default is 1e-4.

    Returns
    -------
    W: np.ndarray
        Estimated factor loading matrix (n_features x n_components).
    sigma2: float
        Estimated noise variance.
    Z: np.ndarray
        Latent variables (n_samples x n_components).
    """
    n_samples, n_features = data.shape
    
    # Initialize parameters
    W = np.random.randn(n_features, n_components)  # Factor loadings
    Z = np.random.randn(n_samples, n_components)   # Latent variables
    alpha = np.ones(n_components) * 1e-6           # Precision of W (small for broad prior)
    sigma2 = 1.0                                   # Initial noise variance
    
    for iteration in range(max_iter):
        # E-step: Update latent variables Z
        M = W.T @ W + sigma2 * np.eye(n_components)
        M_inv = np.linalg.inv(M)
        Z = (data @ W @ M_inv).T  # Shape: (n_components, n_samples)
        
        # M-step: Update W and sigma2
        W_new = (data.T @ Z.T) @ np.linalg.inv(Z @ Z.T + sigma2 * np.diag(alpha))
        sigma2_new = np.mean((data - Z.T @ W.T) ** 2)
        
        # Update alpha (ARD for automatic relevance determination)
        alpha = n_features / (np.sum(W_new ** 2, axis=0) + 1e-10)
        
        # Check convergence
        if np.allclose(W, W_new, rtol=tol) and np.isclose(sigma2, sigma2_new, rtol=tol):
            break
        
        W = W_new
        sigma2 = sigma2_new
    
    return W, sigma2, Z.T

def split_covariance_matrices(C: np.ndarray, data: np.ndarray, n_components: int = None) -> tuple:
    """
    Split the covariance matrix into C_Noise, C_Star, and C_Global using Bayesian PCA.

    Parameters
    ----------
    C: np.ndarray
        Input covariance matrix.
    data: np.ndarray
        Original data matrix (n_samples, n_features) for Bayesian PCA.
    n_components: int, optional
        Number of components for Bayesian PCA. If None, estimated via ARD.

    Returns
    -------
    C_Noise: np.ndarray
        Noise component of the covariance matrix.
    C_Star: np.ndarray
        Structured component of the covariance matrix (signal minus global mode).
    C_Global: np.ndarray
        Global component of the covariance matrix (largest mode).
    """
    # Estimate number of components if not provided
    if n_components is None:
        n_components = min(data.shape) // 2  # Heuristic initial guess
    
    # Run Bayesian PCA directly on the data
    W, sigma2, Z = bayesian_pca(data, n_components)
    
    # Reconstruct signal: C_Signal = W @ W.T (low-rank approximation)
    C_Signal = W @ W.T
    
    # Noise component: C_Noise = sigma^2 * I (isotropic noise)
    C_Noise = sigma2 * np.eye(C.shape[0])
    
    # Compute eigenvalues and eigenvectors of the signal component
    eigenvalues, eigenvectors = np.linalg.eigh(C_Signal)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvectors = eigenvectors[:, ::-1]
    
    # Global component (largest eigenvalue)
    lambda_global = eigenvalues[0]
    C_Global = lambda_global * np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
    
    # Structured component (signal minus global)
    C_Star = C_Signal - C_Global
    
    return C_Noise, C_Star, C_Global

def improved_preprocessing(data: np.ndarray, n_components: int = None) -> tuple:
    """
    Improved preprocessing with Bayesian PCA for multi-level matrix cleaning.
    Replaces shrinkage estimators with a probabilistic approach.

    Parameters
    ----------
    data: np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_components: int, optional
        Number of components for Bayesian PCA. If None, estimated automatically.

    Returns
    -------
    C_Noise, C_Star, C_Global: tuple of np.ndarray
        Cleaned covariance matrix components.
    """
    # Compute the raw covariance matrix
    raw_cov_matrix = np.cov(data.T, bias=True)

    # Split the covariance matrix using Bayesian PCA
    C_Noise, C_Star, C_Global = split_covariance_matrices(raw_cov_matrix, data, n_components)

    return C_Noise, C_Star, C_Global

# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    data = np.random.randn(100, 50)  # 100 samples, 50 features
    
    # Run improved preprocessing with Bayesian PCA
    C_Noise, C_Star, C_Global = improved_preprocessing(data)
    
    print("Noise Component Shape:", C_Noise.shape)
    print("Structured Component Shape:", C_Star.shape)
    print("Global Component Shape:", C_Global.shape)