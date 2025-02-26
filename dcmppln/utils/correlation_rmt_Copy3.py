import numpy as np
from sklearn.covariance import LedoitWolf

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

def permutation_test_threshold(data: np.ndarray, n_permutations: int = 100, percentile: float = 95) -> float:
    """
    Compute a dynamic noise threshold using a permutation test on maximum eigenvalues.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (samples x features).
    n_permutations : int, optional
        Number of permutations to perform. Default is 100.
    percentile : float, optional
        Percentile for the threshold (0-100). Default is 95.

    Returns
    -------
    lambda_robust : float
        Dynamic threshold separating noise from signal.
    """
    max_eigenvalues = []
    for _ in range(n_permutations):
        shuffled_data = np.copy(data)
        # Shuffle each feature independently
        for i in range(data.shape[1]):
            np.random.shuffle(shuffled_data[:, i])
        cov_shuffled = np.cov(shuffled_data.T)
        max_eigenvalue = np.max(np.linalg.eigvalsh(cov_shuffled))
        max_eigenvalues.append(max_eigenvalue)
    return np.percentile(max_eigenvalues, percentile)

def sparsify_matrix(matrix: np.ndarray, sparsity_percentile: float = 95) -> np.ndarray:
    """
    Sparsify the matrix by setting small entries to zero based on a percentile threshold.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to sparsify.
    sparsity_percentile : float, optional
        Percentile for thresholding (0-100). Default is 95.

    Returns
    -------
    sparse_matrix : np.ndarray
        Sparsified symmetric matrix.
    """
    abs_matrix = np.abs(matrix)
    # Compute threshold based on upper triangle (excluding diagonal) to avoid double-counting
    threshold = np.percentile(abs_matrix[np.triu_indices_from(abs_matrix, k=1)], sparsity_percentile)
    sparse_matrix = matrix.copy()
    sparse_matrix[abs_matrix < threshold] = 0
    # Ensure symmetry
    sparse_matrix = np.triu(sparse_matrix, k=0) + np.triu(sparse_matrix, k=1).T
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

    # Structured regime: eigenvalues >= lambda_robust, excluding the largest
    struct_mask = (eigenvalues >= lambda_robust) & (eigenvalues != lambda_global)
    C_Star = (eigenvectors[:, struct_mask] @ np.diag(eigenvalues[struct_mask]) @ eigenvectors[:, struct_mask].T
              if np.any(struct_mask) else np.zeros_like(C_shrunk))

    return C_Noise, C_Star, C_Global

def improved_preprocessing(data: np.ndarray, n_permutations: int = 100, percentile: float = 95, sparsity_percentile: float = 95) -> tuple:
    """
    Preprocess the data to split the covariance matrix into noise, structured, and global components with improved methodology.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix (samples x features).
    n_permutations : int, optional
        Number of permutations for the permutation test. Default is 100.
    percentile : float, optional
        Percentile for the dynamic noise threshold. Default is 95.
    sparsity_percentile : float, optional
        Percentile for sparsifying the structured component. Default is 95.

    Returns
    -------
    C_Noise : np.ndarray
        Noise component of the covariance matrix.
    C_Star : np.ndarray
        Structured component of the covariance matrix (sparsified).
    C_Global : np.ndarray
        Global component of the covariance matrix.
    """
    # Compute the shrunk covariance matrix
    C_shrunk = ledoit_wolf_shrinkage(np.cov(data.T))

    # Compute the dynamic noise threshold
    lambda_robust = permutation_test_threshold(data, n_permutations, percentile)

    # Split the covariance matrix into components
    C_Noise, C_Star, C_Global = split_covariance_matrices(C_shrunk, lambda_robust)

    # Apply sparsity to the structured component
    C_Star = sparsify_matrix(C_Star, sparsity_percentile)

    return C_Noise, C_Star, C_Global

# Example usage
if __name__ == "__main__":
    # Generate sample data: 100 samples, 50 features
    data = np.random.randn(100, 50)
    C_Noise, C_Star, C_Global = improved_preprocessing(data)
    print("C_Noise shape:", C_Noise.shape)
    print("C_Star shape:", C_Star.shape)
    print("C_Global shape:", C_Global.shape)