#Adaptive Robust Eigenvalue Filtering (AREF)
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import SparsePCA

def ledoit_wolf_shrinkage(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Ledoit-Wolf shrinkage to denoise the covariance matrix adaptively.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Input covariance matrix to be denoised.

    Returns
    -------
    shrunk_cov_matrix : np.ndarray
        The covariance matrix after applying Ledoit-Wolf shrinkage.
    """
    lw = LedoitWolf()
    lw.fit(cov_matrix)
    return lw.covariance_

def bootstrap_threshold(cov_matrix: np.ndarray, n_bootstraps: int = 100, percentile: float = 95) -> float:
    """
    Compute a dynamic noise threshold using bootstrapping on eigenvalues.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Input covariance matrix.
    n_bootstraps : int, optional
        Number of bootstrap samples. Default is 100.
    percentile : float, optional
        Percentile for threshold (0-100). Default is 95.

    Returns
    -------
    lambda_robust : float
        Dynamic threshold separating noise from signal.
    """
    eigenvalues = np.linalg.eigh(cov_matrix)[0]
    bootstrap_eigenvalues = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(eigenvalues), len(eigenvalues), replace=True)
        bootstrap_sample = eigenvalues[indices]
        bootstrap_eigenvalues.append(np.max(bootstrap_sample))
    
    return np.percentile(bootstrap_eigenvalues, percentile)

def sparsify_matrix(matrix: np.ndarray, n_components: int = 20, alpha: float = 1.0) -> np.ndarray:
    """
    Extract sparse structured components using Sparse PCA.

    Parameters
    ----------
    matrix : np.ndarray
        Input covariance matrix to sparsify.
    n_components : int, optional
        Number of sparse components to extract. Default is 20.
    alpha : float, optional
        Sparsity controlling parameter for Sparse PCA. Default is 1.0.

    Returns
    -------
    sparse_matrix : np.ndarray
        Sparsified structured component of the covariance matrix.
    """
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    spca.fit(matrix)
    components = spca.components_.T  # Shape: (n_features, n_components)
    eigenvalues = np.var(spca.transform(matrix), axis=0)  # Explained variance per component
    sparse_matrix = components @ np.diag(eigenvalues) @ components.T
    return sparse_matrix

def split_covariance_matrices(C_shrunk: np.ndarray, lambda_robust: float) -> tuple:
    """
    Split the shrunk covariance matrix into noise, structured, and global components using dynamic thresholding.

    Parameters
    ----------
    C_shrunk : np.ndarray
        Shrunk covariance matrix.
    lambda_robust : float
        Dynamic threshold for separating noise from signal.

    Returns
    -------
    C_Noise : np.ndarray
        Noise component of the covariance matrix.
    C_Star : np.ndarray
        Structured component of the covariance matrix.
    C_Global : np.ndarray
        Global component of the covariance matrix.
    """
    # Eigenvalue decomposition (ascending order by default)
    eigenvalues, eigenvectors = np.linalg.eigh(C_shrunk)
    # Sort in descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Noise regime: eigenvalues below the dynamic threshold
    noise_mask = eigenvalues < lambda_robust
    C_Noise = (eigenvectors[:, noise_mask] @ np.diag(eigenvalues[noise_mask]) @ eigenvectors[:, noise_mask].T
               if np.any(noise_mask) else np.zeros_like(C_shrunk))

    # Global regime: largest eigenvalue
    lambda_global = eigenvalues[0]
    v_global = eigenvectors[:, 0]
    C_Global = lambda_global * np.outer(v_global, v_global)

    # Structured regime: remaining eigenvalues >= lambda_robust, excluding the largest
    struct_mask = (eigenvalues >= lambda_robust) & (eigenvalues != lambda_global)
    C_Star_intermediate = (eigenvectors[:, struct_mask] @ np.diag(eigenvalues[struct_mask]) @ eigenvectors[:, struct_mask].T
                           if np.any(struct_mask) else np.zeros_like(C_shrunk))

    return C_Noise, C_Star_intermediate, C_Global

def improved_preprocessing(data: np.ndarray, n_bootstraps: int = 100, percentile: float = 95, sparsity_percentile: float = 95) -> tuple:
    """
    Preprocess the data to split the covariance matrix into noise, structured, and global components using Adaptive Robust Eigenvalue Filtering (AREF).

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (samples x features).
    n_bootstraps : int, optional
        Number of bootstrap iterations for threshold. Default is 100.
    percentile : float, optional
        Percentile for the dynamic noise threshold. Default is 95.
    sparsity_percentile : float, optional
        Controls sparsity via Sparse PCA (interpreted as alpha parameter here). Default is 95.

    Returns
    -------
    C_Noise : np.ndarray
        Noise component of the covariance matrix.
    C_Star : np.ndarray
        Structured component of the covariance matrix (sparsified via Sparse PCA).
    C_Global : np.ndarray
        Global component of the covariance matrix.
    """
    # Compute the shrunk covariance matrix
    C_shrunk = ledoit_wolf_shrinkage(np.cov(data.T))

    # Compute the dynamic noise threshold using bootstrapping
    lambda_robust = bootstrap_threshold(C_shrunk, n_bootstraps, percentile)

    # Split the covariance matrix into components
    C_Noise, C_Star_intermediate, C_Global = split_covariance_matrices(C_shrunk, lambda_robust)

    # Apply Sparse PCA to the structured component (interpreting sparsity_percentile as alpha)
    C_Star = sparsify_matrix(C_shrunk if np.all(C_Star_intermediate == 0) else C_Star_intermediate, 
                             n_components=20, alpha=sparsity_percentile / 10.0)  # Scale alpha for practical range

    return C_Noise, C_Star, C_Global

# Example usage
if __name__ == "__main__":
    # Generate sample data: 100 samples, 50 features
    data = np.random.randn(100, 50)
    C_Noise, C_Star, C_Global = improved_preprocessing(data)
    print("C_Noise shape:", C_Noise.shape)
    print("C_Star shape:", C_Star.shape)
    print("C_Global shape:", C_Global.shape)