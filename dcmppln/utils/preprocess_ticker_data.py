from typing import List
import pandas as pd
import numpy as np
from collections import namedtuple

PortfolioData = namedtuple('StockData', ['returns', 'mean_returns', 'covariances', 'correlations', 'log_returns'])

def calculate_returns_correlations(data : pd.DataFrame, log_returns=False):
    """ Given a dataframe of asset tickers and values over a period of time, computes the returns and correlations usable for the PO module.

    Parameters
    ----------
    data : pd.DataFrame
        m rows and n columns
    log_returns : bool
        Whether to return the log returns or not
    
    Returns
    -------
        namedtuple of np.array(m-1, n), np.array(n), np.array(n,n), np.array(n,n), bool
            Named tuple with fields returns, mean_returns, covariances, correlations, and log returns
    """
    returns = data/data.shift(1)
    if log_returns:
        returns = np.log(returns)
    # drop first row
    returns = returns.iloc[1:]

    mean_returns = returns.mean().to_numpy()
    # Compute covariance matrix of log returns
    covariances = returns.cov().to_numpy()

    # Compute correlation matrix of log returns
    correlations = returns.corr().to_numpy()

    portfolio_data = PortfolioData(returns=returns.to_numpy(), mean_returns=mean_returns, covariances=covariances, correlations=correlations, log_returns=log_returns)
    return portfolio_data
    
