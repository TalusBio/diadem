"""The retention time matrix factorization model."""
from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError
from tqdm import trange

LOGGER = logging.getLogger(__name__)


class MatrixFactorizationModel(nn.Module):
    """The PyTorch matrix factorization model.

    Parameters
    ----------
    n_peptides : int
        The number of peptides.
    n_runs : int
        The number of runs.
    n_factors: int
        The number of latent factors.
    rng : int | numpy.random.Generator | None
        The random number generator.
    """

    def __init__(
        self,
        n_peptides: int,
        n_runs: int,
        n_factors: int,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize an ImputerModel."""
        super().__init__()
        self.n_peptides = n_peptides
        self.n_runs = n_runs
        self.n_factors = n_factors
        self.rng = np.random.default_rng(rng)

        torch.manual_seed(self.rng.integers(1, 100000))

        # The model:
        self.peptide_factors = nn.Parameter(torch.randn(n_peptides, n_factors))
        self.run_factors = nn.Parameter(torch.randn(n_factors, n_runs))

    def forward(self) -> torch.Tensor:
        """Reconstruct the matrix.

        Returns
        -------
        torch.Tensor of shape (n_peptide, n_runs)
            The reconstructed matrix.
        """
        return torch.mm(self.peptide_factors, self.run_factors)


class MFImputer:
    """A matrix factorization imputation model.

    The MFImputer is a PyTorch model wrapped in a sklearn API.

    Parameters
    ----------
    n_factors : int | None, optional
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
    lr: float, optional
        The learning rate.
    device : str or torch.Device, optional
        A valid PyTorch device on which to perform the optimization.
    rng : int | np.random.Generator | None, optional
        The random number generator.
    task : str | None, optional
        A sting specifying the task. This is only used for logging.
    silent : bool, optional
        Suppress logging messages.
    """

    def __init__(
        self,
        n_factors: int = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
        n_iter_no_change: int = 20,
        lr: float = 0.1,
        device: str | torch.device = "cpu",
        rng: int | np.random.Generator | None = None,
        task: str | None = None,
        silent: bool = False,
    ) -> None:
        """Initialize the RTImputer."""
        # Parameters:
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.lr = lr
        self.rng = np.random.default_rng(rng)
        self.task = task
        self.silent = silent

        # Set during fit:
        self._model = None
        self._history = None
        self._shape = None
        self._std = None
        self._means = None

    @property
    def model_(self) -> MatrixFactorizationModel:
        """The underlying PyTorch model."""
        if self._model is None:
            raise NotFittedError("This model has not been fit yet.")

        return self._model

    @property
    def history_(self) -> pd.DataFrame:
        """The training history."""
        return pd.DataFrame(
            self._history,
            columns=["iteration", "train_loss"],
        )

    def _info(self, msg: str, *args: str | None) -> None:
        """Log at the info level.

        Parameters
        ----------
        msg : str
            The message to be logged.
        *args : str, optional
            Values to be formatted into the message.
        """
        if not self.silent:
            LOGGER.info(msg, *args)

    def transform(
        self,
        X: np.array | torch.Tensor | None = None,
    ) -> np.array:  # noqa: N803
        """Impute missing retention times.

        Parameters
        ----------
        X : array of shape (n_peptides, n_runs), optional
            The value matrix. Missing peptides should be denoted as np.nan.
            If ``none``, the full reconstruction will be returned.

        Returns
        -------
        np.array of shape (n_peptides, n_runs)
            The matrix with missing values imputed.
        """
        if X is None:
            return self.model_().to("cpu").detach().numpy()

        # Prepare the input and initialize model
        X = to_tensor(X)
        mask = torch.isnan(X)
        X_hat = self.model_().to("cpu").detach().type_as(X)
        X[mask] = X_hat[mask]
        return X.numpy()

    def fit(self, X: np.ndarray | torch.Tensor) -> MFImputer:
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_peptides, n_runs)
            The value matrix. Missing peptides should be denoted as np.nan.

        Returns
        -------
        self
        """
        if self.n_factors is None:
            raise ValueError(
                (
                    "n_factors must be specified using search_factors() "
                    "to find the best value."
                ),
            )

        if self.task:
            self._info("Training %s model...", self.task)

        # Prepare the input and initialize model
        X = to_tensor(X)
        mask = ~torch.isnan(X)
        X = X.to(self.device, torch.float32)

        self._shape = X.shape
        self._model = MatrixFactorizationModel(*X.shape, self.n_factors, self.rng)
        self._model.to(self.device)
        self._history = []
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        log_interval = max(1, self.max_iter // 20)
        self._info("Logging every %i iterations...", log_interval)

        # The main training loop:
        best_loss = np.inf
        early_stopping_counter = 0
        self._info("+-----------------------+")
        self._info("| Iteration | Train MSE |")
        self._info("+-----------------------+")
        bar = trange(self.max_iter, disable=self.silent)
        for iteration in bar:
            optimizer.zero_grad()
            X_hat = self._model()
            loss = ((X - X_hat)[mask] ** 2).mean()
            loss.backward()
            optimizer.step()
            self._history.append((iteration, loss.item()))

            if not iteration % log_interval:
                self._info("| %9i | %9.4f |", iteration, loss.item())

            bar.set_postfix(loss=f"{loss:,.3f}")
            if self.tol is not None:
                if loss < best_loss:
                    best_loss = loss.item()
                    early_stopping_counter = 0
                    continue
                early_stopping_counter += 1
                if early_stopping_counter >= self.n_iter_no_change:
                    break

        self._info("+-----------------------+")
        self._model.to("cpu")
        self._info("DONE!")
        return self

    def fit_transform(self, X: np.array | torch.Tensor) -> np.ndarray:
        """Fit and impute missing retention times.

        Parameters
        ----------
        X : array of shape (n_peptides, n_runs)
            The value matrix. Missing values should be denoted as np.nan.

        Returns
        -------
        np.array of shape (n_peptides, n_runs)
            The predicted retention time matrix.
        """
        return self.fit(X).transform(X)

    def search_factors(
        self,
        X: np.array | torch.Tensor,
        n_factors: Iterable[int],
        folds: int = 3,
    ) -> MFImputer:
        """Perform line search for the number of latent factors.

        Parameters
        ----------
        X : array of shape (n_peptides, n_runs)
            The value matrix. Missing values should be denoted as np.nan.
        n_factors : Iterable[int]
            The numbers of latent factors to try.
        folds: int, optional
            The number of cross-validation folds.

        Returns
        -------
        self
        """
        X = to_tensor(X)
        prev_silent = self.silent

        # Create CV splits
        indices = np.dstack(np.meshgrid(range(X.shape[0]), range(X.shape[1])))
        indices = indices.reshape(-1, 2)

        self.rng.shuffle(indices, axis=0)
        splits = torch.split(torch.tensor(indices), indices.shape[0] // folds)

        # Do the line search
        self._info("Searching for the best number of latent factors...")
        self.silent = True
        scores = []
        # This could be parallelized, but I don't think its worth it yet.
        for num in n_factors:
            self.n_factors = num
            split_scores = []
            for split in splits:
                split = tuple(split.T)
                train = X.clone()
                train[split] = np.nan
                pred = to_tensor(self.fit_transform(train))
                split_scores.append(((X[split] - pred[split]) ** 2).mean())

            scores.append(sum(split_scores).item())

        self.n_factors = n_factors[np.argmin(scores)]
        self.silent = prev_silent
        self._model = None
        self._info(" -> Chose %s factors", self.n_factors)
        return self


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

    return torch.tensor(array)
