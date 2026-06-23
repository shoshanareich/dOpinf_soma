import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import opinf
import pickle
from pathlib import Path


def weighted_lp_error(Q_true, Q_approx, weights=None, normalize=True):
    """
    Computes the weighted L2 error for spatial-temporal data.
    
    Parameters:
    - Q_true: 2D array (r, t)
    - Q_approx: 2D array (r, t)
    - weights: 1D array (r,) or None (defaults to ones).
    - normalize: If True, divides by the max weighted norm of the true data.
    """
    # If weights are not provided, use uniform weights (ones)
    if weights is None:
        weights = np.ones(Q_true.shape[0])

    w = weights[:, np.newaxis]
    
    # Compute weighted squared values along the spatial axis (axis 0)
    sq_err = np.sum(w * (Q_true - Q_approx)**2, axis=0)
    sq_true = np.sum(w * Q_true**2, axis=0)

    # L2 norms
    absolute_error = np.sqrt(sq_err)
    norm_of_data = np.sqrt(sq_true)

    if normalize:
        return absolute_error / norm_of_data.max()
    
    return absolute_error


class ShiftedTikhonovSolver(opinf.lstsq.TikhonovSolver):
    """Tikhonov solver that regularizes toward a nonzero operator matrix."""

    def __init__(self, regularizer=None, target_operator_matrix=None, method="lstsq"):
        super().__init__(regularizer, method=method)
        self.target_operator_matrix = None
        if target_operator_matrix is not None:
            self.target_operator_matrix = np.asarray(target_operator_matrix)

    def _check_target_shape(self):
        if self.target_operator_matrix is None:
            raise AttributeError("target_operator_matrix not set")
        expected = (self.r, self.d)
        if self.target_operator_matrix.shape != expected:
            raise ValueError(
                "target_operator_matrix.shape = "
                f"{self.target_operator_matrix.shape} != {expected}"
            )

    def fit(self, data_matrix, lhs_matrix):
        super().fit(data_matrix, lhs_matrix)
        self._check_target_shape()
        return self

    def solve(self):
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
        self._check_target_shape()

        target_t = self.target_operator_matrix.T
        if self.method == "lstsq":
            data_padded = np.vstack((self.data_matrix, self.regularizer))
            lhs_padded = np.vstack((self.lhs_matrix.T, self.regularizer @ target_t))
            return sla.lstsq(data_padded, lhs_padded, **self.options)[0].T

        regt_reg = self.regularizer.T @ self.regularizer
        lhs = self._DtZt + regt_reg @ target_t
        return sla.solve(self._DtD + regt_reg, lhs, assume_a="pos").T

    def regresidual(self, Ohat):
        residual = self.residual(Ohat)
        diff = (Ohat - self.target_operator_matrix).T
        return residual + np.sum((self.regularizer @ diff) ** 2, axis=0)


def _learned_operator_templates():
    return [
        opinf.operators.ConstantOperator(),
        opinf.operators.LinearOperator(),
        opinf.operators.QuadraticOperator(),
    ]


def _make_regularizer(reg_A, reg_H, r):
    reg_c = reg_A
    return opinf.lstsq.TikhonovSolver.get_operator_regularizer(
        operators=_learned_operator_templates(),
        regularization_parameters=[reg_c, reg_A, reg_H],
        state_dimension=r,
    )


def _make_solver(reg_A, reg_H, r, regularization_target=None):
    regularizer = _make_regularizer(reg_A, reg_H, r)
    if regularization_target is None:
        return opinf.lstsq.TikhonovSolver(regularizer, method="lstsq")
    return ShiftedTikhonovSolver(
        regularizer,
        target_operator_matrix=regularization_target,
        method="lstsq",
    )


def _make_model(B, C, solver):
    operators = [
        opinf.operators.InputOperator(entries=B),
        opinf.operators.ConstantOperator(entries=C),
        opinf.operators.ConstantOperator(),
        opinf.operators.LinearOperator(),
        opinf.operators.QuadraticOperator(),
    ]
    return opinf.ROM(model=opinf.models.DiscreteModel(operators, solver=solver))


def freeze_model(model):
    """Return a prediction-only ROM with all fitted operators fixed."""
    operators = [
        type(operator)(entries=np.array(operator.entries, copy=True))
        for operator in model.model.operators
    ]
    return opinf.ROM(model=opinf.models.DiscreteModel(operators))


def get_learned_operator_matrix(model):
    return np.array(model.model.operator_matrix, copy=True)




def fit_and_score_model(
    reg_A,
    reg_H,
    *,
    Qs_,
    inputs,
    B,
    C,
    r,
    nt,
    t,
    weights=None,
    regularization_target=None,
):
    """Fit ROM with given regularization and return (converged, error)."""

    if reg_A > reg_H:
        return False, np.inf

    rom = _make_model(
        B,
        C,
        solver=_make_solver(reg_A, reg_H, r, regularization_target),
    )
    model = rom.fit(states=Qs_, inputs=inputs)


   # # slice the big matrix to check how it performs on each scenario
   # num_scenarios = Qs_.shape[1] // nt
    scenario_errors = []

   # for i in range(num_scenarios):
    for i in range(len(Qs_)):
        # # Extract the ground truth and inputs for THIS scenario
        # start, end = i * nt, (i + 1) * nt
        # Q_true = Qs_[:, start:end]
        # u = inputs[:, start:end]

        # Predict starting from the first snapshot of this segment
       # Q_ROM_ = model.predict(Q_true[:, 0], niters=nt, inputs=u)
        Q_ROM_ = model.predict(Qs_[i][:, 0], niters=nt, inputs=inputs[i])

        if np.isnan(Q_ROM_).any():
            return False, np.inf
    
       # err = weighted_lp_error(Q_true, Q_ROM_, weights=weights, normalize=True)
        err = weighted_lp_error(Qs_[i], Q_ROM_, weights=weights, normalize=True)
        scenario_errors.append(la.norm(err))
    return True, max(scenario_errors)




class TikhonovSweep:
    def __init__(
        self,
        *,
        Qs_,
        inputs,
        B,
        C,
        r,
        nt,
        t,
        weights_A,
        weights_H,
        norm_weights=None,
        regularization_target=None,
    ):
        self.Qs_ = Qs_
        self.inputs = inputs
        self.B = B
        self.C = C
        self.r = r
        self.nt = nt
        self.t = t

        self.weights_A = weights_A
        self.weights_H = weights_H
        self.regularization_target = regularization_target
        #self.norm_weights = norm_weights
        if norm_weights is None:
            self.norm_weights = np.ones(r)
        else:
            self.norm_weights = norm_weights

        self.errors = np.full((len(weights_H), len(weights_A)), np.inf)
        self.converged = np.zeros_like(self.errors, dtype=bool)

        self.best_reg_A = None
        self.best_reg_H = None
        self.best_error = np.inf


    def run(self, verbose=True):
        'sweep'
        for i, reg_H in enumerate(self.weights_H):
            if verbose:
                print(f"iteration {i+1} / {len(self.weights_H)}", flush=True)

            for j, reg_A in enumerate(self.weights_A):
                converged, err = fit_and_score_model(
                    reg_A,
                    reg_H,
                    Qs_=self.Qs_,
                    inputs=self.inputs,
                    B=self.B,
                    C=self.C,
                    r=self.r,
                    nt=self.nt,
                    t=self.t,
                    weights=self.norm_weights,
                    regularization_target=self.regularization_target,
                )

                self.converged[i, j] = converged
                self.errors[i, j] = err

                if converged and err < self.best_error:
                    self.best_error = err
                    self.best_reg_A = reg_A
                    self.best_reg_H = reg_H

        return self.best_reg_A, self.best_reg_H, self.best_error


    def run_mpi(self, comm, verbose=True):
        """Distribute the regularization sweep over existing MPI ranks."""
        rank = comm.Get_rank()
        size = comm.Get_size()
        tasks = [
            (i, j, reg_A, reg_H)
            for i, reg_H in enumerate(self.weights_H)
            for j, reg_A in enumerate(self.weights_A)
        ]
        local_tasks = tasks[rank::size]

        if rank == 0 and verbose:
            print(
                f"running Tikhonov grid search with {size} MPI ranks "
                f"over {len(tasks)} candidates",
                flush=True,
            )

        local_results = []
        for i, j, reg_A, reg_H in local_tasks:
            try:
                converged, err = fit_and_score_model(
                    reg_A,
                    reg_H,
                    Qs_=self.Qs_,
                    inputs=self.inputs,
                    B=self.B,
                    C=self.C,
                    r=self.r,
                    nt=self.nt,
                    t=self.t,
                    weights=self.norm_weights,
                    regularization_target=self.regularization_target,
                )
                failure = None
            except Exception as exc:
                converged = False
                err = np.inf
                failure = f"rank {rank}: reg_A={reg_A:.3e}, reg_H={reg_H:.3e} failed with {exc!r}"

            local_results.append((i, j, converged, err, failure))

        gathered_results = comm.gather(local_results, root=0)
        best_tuple = None

        if rank == 0:
            failures = []
            for rank_results in gathered_results:
                for i, j, converged, err, failure in rank_results:
                    self.converged[i, j] = converged
                    self.errors[i, j] = err
                    if failure is not None:
                        failures.append(failure)

            for failure in failures[:10]:
                print(f"  -> {failure}", flush=True)
            if len(failures) > 10:
                print(f"  -> {len(failures) - 10} additional candidates failed", flush=True)

            self._set_best_from_errors()
            best_tuple = (self.best_reg_A, self.best_reg_H, self.best_error)

        best_tuple = comm.bcast(best_tuple, root=0)
        self.best_reg_A, self.best_reg_H, self.best_error = best_tuple
        return best_tuple


    def _set_best_from_errors(self):
        self.best_reg_A = None
        self.best_reg_H = None
        self.best_error = np.inf

        for i, reg_H in enumerate(self.weights_H):
            for j, reg_A in enumerate(self.weights_A):
                err = self.errors[i, j]
                if self.converged[i, j] and err < self.best_error:
                    self.best_error = err
                    self.best_reg_A = reg_A
                    self.best_reg_H = reg_H


    def save(self, path):
        """Save/load sweep result (not full data arrays)."""
        path = Path(path)
        payload = {
            "best_reg_A": self.best_reg_A,
            "best_reg_H": self.best_reg_H,
            "best_error": self.best_error,
            "weights_A": self.weights_A,
            "weights_H": self.weights_H,
            "r": self.r,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path, **runtime_kwargs):
        """Reload sweep result and attach runtime data."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            weights_A=data["weights_A"],
            weights_H=data["weights_H"],
            r=data["r"],
            **runtime_kwargs,  # Q_, inputs, B, C, nt, t
        )

        obj.best_reg_A = data["best_reg_A"]
        obj.best_reg_H = data["best_reg_H"]
        obj.best_error = data["best_error"]

        return obj


    def fit_best_model(self):
        """
        Retrain the model using the best regularization found during the sweep
        across ALL training scenarios and return the trained model object.
        """
        if self.best_reg_A is None:
            raise RuntimeError("run() or load() must be called first.")

        rom = _make_model(
            self.B,
            self.C,
            solver=_make_solver(
                self.best_reg_A,
                self.best_reg_H,
                self.r,
                self.regularization_target,
            ),
        )
        model = rom.fit(states=self.Qs_, inputs=self.inputs)

        return model


def fit_sequential_model(
    *,
    Qs_,
    inputs,
    B,
    C,
    r,
    n_year_train,
    n_days,
    block_years,
    weights_A,
    weights_H,
    comm,
    norm_weights=None,
    t=None,
    verbose=True,
):
    """Fit blocks of training years, anchoring each block to the previous fit."""
    if block_years is None:
        raise ValueError("block_years must be defined for sequential learning")
    block_years = int(block_years)
    if block_years <= 0:
        raise ValueError("block_years must be a positive integer")

    snapshots_per_year = int(360 / n_days)
    total_nt = snapshots_per_year * n_year_train
    block_nt = snapshots_per_year * block_years
    if block_nt < 2:
        raise ValueError("Each sequential block must contain at least two snapshots")

    rank = comm.Get_rank()
    target = None
    final_model = None

    for block_index, start in enumerate(range(0, total_nt, block_nt)):
        stop = min(start + block_nt, total_nt)
        current_nt = stop - start
        if current_nt < 2:
            if rank == 0 and verbose:
                print(
                    f"skipping final sequential block with {current_nt} snapshot",
                    flush=True,
                )
            continue

        states_block = [Q[:, start:stop] for Q in Qs_]
        inputs_block = [U[..., start:stop] for U in inputs]
        start_year = start / snapshots_per_year
        stop_year = stop / snapshots_per_year

        if rank == 0 and verbose:
            if target is None:
                anchor_msg = "zero"
            else:
                anchor_msg = "previous operators"
            print(
                "sequential block "
                f"{block_index + 1}: years {start_year:g}-{stop_year:g}, "
                f"regularizing toward {anchor_msg}",
                flush=True,
            )

        sweep = TikhonovSweep(
            Qs_=states_block,
            inputs=inputs_block,
            B=B,
            C=C,
            r=r,
            nt=current_nt,
            t=t,
            weights_A=weights_A,
            weights_H=weights_H,
            norm_weights=norm_weights,
            regularization_target=target,
        )
        reg_A, reg_H, best_err = sweep.run_mpi(comm, verbose=verbose)
        if reg_A is None:
            raise RuntimeError(
                f"No converged regularization candidate for sequential block {block_index + 1}"
            )

        if rank == 0 and verbose:
            print(
                f"Best sequential block {block_index + 1}: "
                f"{best_err} {reg_A} {reg_H}",
                flush=True,
            )

        if rank == 0:
            final_model = sweep.fit_best_model()
            target = get_learned_operator_matrix(final_model)
        target = comm.bcast(target, root=0)

    if rank == 0 and final_model is None:
        raise RuntimeError("Sequential learning did not fit any blocks")

    return freeze_model(final_model) if rank == 0 else None
