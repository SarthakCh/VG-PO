###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
from dcmppln.utils.utils import get_instance_non_private_attributes
from dcmppln.utils import correlation_rmt, correlation_ledoit_wolf, correlation_bayesian_PCA, correlation_pts, correlation_bootstrap
from dcmppln.utils.utils import timeit
from abc import ABC, abstractmethod

class Denoiser_Base(ABC):
    # use this structure to ease profile readability
    def __call__(self, C):
        return self.denoise(C)

    @abstractmethod
    def denoise(self, C : np.array)->np.array :
        pass

class Denoiser(Denoiser_Base):
    """
    Class to apply noise reduction before clustering
    """

    def __init__(self, active: bool = True, q: float = 0.5, q_fit: bool = True):
        """
        Parameters
        ----------
        active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        q: ratio of number of variables to number of observations. e.g. N_stocks/N_days
        q_fit: finds the best q
        """

        self.active = active
        self.q = q
        self.q_fit = q_fit

        self.params = get_instance_non_private_attributes(self)

    # Check with q=0.5, q_fit = False,  q=1.5
    @timeit
    def denoise(self, C: np.array) -> np.array:
        """
        function to calculate denoise, if the component is deactivated return the
        C else return

        Parameters
        ----------
        C: np.array input correlation matrix
        """

        if not self.active:
            return C

        C_1, C_2, C_3 = correlation_rmt.split_covariance_matrices(C, beta=self.q, q_fit=self.q_fit)
        return C_2

class Denoiser_Shrinkage(Denoiser_Base):
    """
    Class to apply noise reduction before clustering
    """

    def __init__(self, active: bool = True, gamma=1.):
        """
        Parameters
        ----------
        active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        q: ratio of number of variables to number of observations. e.g. N_stocks/N_days
        q_fit: finds the best q
        """

        self.active = active
        self.gamma = gamma

        self.params = get_instance_non_private_attributes(self)

    # Check with q=0.5, q_fit = False,  q=1.5
    @timeit
    def denoise(self, C: np.array) -> np.array:
        """
        function to calculate denoise, if the component is deactivated return the
        C else return

        Parameters
        ----------
        C: np.array input correlation matrix
        """

        if not self.active:
            return C

        C_1, C_2, C_3 = correlation_ledoit_wolf.split_covariance_matrices(C, gamma=self.gamma)
        return C_2

class Denoiser_Bayesian_PCA(Denoiser_Base):
    def __init__(self, raw_returns, active: bool=True, n_components: int=None):
        self.active = active
        self.n_components = n_components
        self.raw_returns = raw_returns
        self.params = get_instance_non_private_attributes(self)
    @timeit
    def denoise(self, C: np.array) -> np.array:
        if not self.active:
            return C

        C_1, C_2, C_3 = correlation_bayesian_PCA.split_covariance_matrices(C, data=self.raw_returns, n_components=self.n_components)
        return C_2

class Denoiser_PTSCDS(Denoiser_Base):
    def __init__(self, raw_returns, active: bool=True):
        self.active = active
        # Compute the shrunk covariance matrix from the data
        self.C_shrunk = correlation_pts.LedoitWolf().fit(raw_returns).covariance_
        # Compute the dynamic noise threshold
        self.lambda_robust = correlation_pts.permutation_test_threshold(raw_returns)
        # Compute the dynamic noise threshold
        self.params = get_instance_non_private_attributes(self)
    @timeit
    def denoise(self, C: np.array) -> np.array:
        if not self.active:
            return C

        C_1, C_2, C_3 = correlation_pts.split_covariance_matrices(C_shrunk=self.C_shrunk, lambda_robust=self.lambda_robust)
        return C_2

class Denoiser_Bootstrap(Denoiser_Base):
    def __init__(self, raw_returns, active: bool=True, lambda_robust: float=None):
        self.active = active
        self.C_shrunk = correlation_bootstrap.LedoitWolf().fit(raw_returns).covariance_

        # Compute the dynamic noise threshold
        self.lambda_robust = correlation_bootstrap.bootstrap_threshold(self.C_shrunk)
        self.params = get_instance_non_private_attributes(self)
    @timeit
    def denoise(self, C: np.array) -> np.array:
        if not self.active:
            return C

        C_1, C_2, C_3 = correlation_bootstrap.split_covariance_matrices(C_shrunk=self.C_shrunk, lambda_robust=self.lambda_robust)
        return C_2