"""This module contains a matrix factorization model."""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange

LOGGER = logging.getLogger(__name__)


class MatrixModel(nn.Module):
    """The PyTorch matrix factorization model.

    Parameters
    ----------
    n_peptides : int
        The number of peptides.
    n_runs : int
        The number of runs.
    n_factors: int
        The number of latent factors.
    """
    def __init__(self, n_peptides: int, n_runs: int, n_factors: int) -> None:
        """Initialize an ImputerModel."""
        super().__init__()
        self.n_peptides = n_peptides
        self.n_runs = n_runs
        self.n_factors = n_factors
        
        torch.manual_seed(0)

        # The model:
        self.peptide_factors = torch.randn(n_peptides, n_factors, requires_grad=True)
        self.run_factors = torch.randn(n_runs, n_factors, requires_grad=True)

    def forward(self, X, non_zero_mask) -> torch.Tensor:
        """Reconstruct the matrix and determine prediction error.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_peptides, n_runs)
            The retention time array. Missing peptides should be denoted as -1.
        non_zero_mask: torch.Tensor of shape (n_peptides, n_runs)
            Tensor of boolean values where missing values are False.

        Returns
        -------
        torch.Tensor of shape (1, 1)
            Difference between the expected and predicted RTs.
        """
        
        predicted = torch.mm(self.peptide_factors, self.run_factors.T)
        
        diff = (X - predicted)**2
        prediction_error = torch.sum(diff*non_zero_mask)
        
        return prediction_error
    
    def get_params(self): 
        """Return the peptide and run factors."""
        return self.peptide_factors, self.run_factors
    


class RTImputer:
    """A retention time prediction model.

    This is a PyTorch model wrapped in a sklearn-like API.

    Parameters
    ----------
    n_factors : int
        The number of latent factors.
    max_iter : int, optional
        The maximum number of training iterations
    tol : float, optional
        The percent improvement over the previous loss required to
        continue trianing. Used in conjuction with ``n_iter_no_change``
        to trigger early stopping. Set to ``None`` to disable.
    n_iter_no_change : int, optional
        The number of iterations to wait before triggering early
        stopping.
    device : str or torch.Device
        A valid PyTorch device on which to perform the optimization.

    """

    def __init__(
            self,
            n_factors: int,
            max_iter: int = 1000,
            tol: float = 1e-4,
            n_iter_no_change: int = 20,
            device: str | torch.device = "cpu",
    ):
        """Initialize the RTImputer"""
        # Parameters:
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.device = device

        # Set during fit:
        self._model = None
        self._history = None
        self._shape = None
        self._std = None
        self._means = None

    @property
    def model_(self) -> MatrixModel:
        """The underlying PyTorch model."""
        return self._model

    @property
    def history_(self) -> pd.DataFrame:
        """The training history."""
        return pd.DataFrame(
            self._history,
            columns=["iteration", "train_loss"],
        )
    
    def predict(self, only_missing: bool, non_zero_mask: torch.Tensor, X: torch.Tensor=None) -> torch.Tensor:
        """Reconstruct the matrix.

        Parameters
        ----------
        only_missing : bool
            True if predicted model should only predict missing values. 
            False if predicted model should return predicted values for all peptides/runs.
        non_zero_mask: torch.Tensor of shape (n_peptides, n_runs)
            Tensor of boolean values where missing values are False.
        X : torch.Tensor of shape (n_peptides, n_runs)
            The retention time array. Missing peptides should be denoted as -1.
                    
        Returns
        -------
        torch.Tensor of shape (n_peptides, n_runs)
            The predicted retention time matrix.
        """
        
        peptide_factors, run_factors = self._model.get_params()
        if only_missing:
            pred = torch.mm(peptide_factors, run_factors.T).detach().numpy()
            try:
                X = X.copy()
            except AttributeError:
                res = X.clone().numpy()
            res[~non_zero_mask] = pred[~non_zero_mask]
            return res
        
        return torch.mm(self.peptide_factors, self.run_factors.T).detach().numpy()

    
    def fit(self, X: np.ndarray, only_missing: bool=True) -> torch.Tensor:
        """Fit the RTImputer.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_peptides, n_runs)
            The retention time array. Missing peptides should be denoted as -1.
        only_missing : bool
            True if predicted model should only predict missing values. 
            False if predicted model should return predicted values for all peptides/runs.

        Returns
        -------
        torch.Tensor of shape (n_peptides, n_runs)
            The predicted retention time matrix.
        """
        
        LOGGER.info("Training retention time predictor...")
        
        # Set seed:
        torch.manual_seed(0)

        # Prepare the input and initialize model
        X = to_tensor(X)
        X = X.to(self.device, torch.float32)
        X[torch.isnan(X)] = -1
       
        self._shape = X.shape
        self._model = MatrixModel(*X.shape, self.n_factors).to(self.device)
        self._history = []
        
        # Get actual values to compare model against
        non_zero_mask = (X != -1)
        
        optimizer = torch.optim.RMSprop(self._model.get_params(), lr=.07)
        log_interval = max(1, self.max_iter // 20)
        LOGGER.info("Logging every %i iterations...", log_interval)

        # The main training loop:
        best_loss = np.inf
        early_stopping_counter = 0
        LOGGER.info("Iteration | Train Loss")
        LOGGER.info("----------------------")
        bar = trange(self.max_iter)
        for epoch in bar:
            optimizer.zero_grad()
            loss = self._model(X, non_zero_mask)
            loss.backward()
            optimizer.step()
            self._history.append((epoch, loss.item()))
            
            if not epoch % log_interval:
                LOGGER.info("%9i | %10.4f", epoch, loss.item())
                
            bar.set_postfix(loss=f'{loss:,.3f}')
            if self.tol is not None:
                if loss < best_loss:
                    best_loss = loss.item()
                    early_stopping_counter = 0
                    continue
                early_stopping_counter += 1
                if early_stopping_counter >= self.n_iter_no_change:
                    LOGGER.info("Stopping...")
                    break
                    
        LOGGER.info("DONE!")

        return self.predict(only_missing, non_zero_mask, X)
    
    
def to_tensor(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Transform an array into a PyTorch Tensor, copying the data.

    Parameters
    ----------
    array : numpy.ndarray or torch.Tensor
        The array to transform

    Returns
    -------
    torch.Tensor
        The converted PyTorch tensor.
    """
    if isinstance(array, torch.Tensor):
        return array.to("cpu").clone().detach()

    return torch.tensor(X)