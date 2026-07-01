import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import opinf
import pickle
from pathlib import Path


def score_trajectory(Q_true, Q_ROM, *, weights=None, window_starts=None,
                     window_len=None, var_weight=0.0):
    """Scalar fidelity score for a single scenario rollout.

    - Default: L2-over-time of the normalized spatial error (original behavior).
    - window_starts (option 3): mean normalized-L2 over short windows instead of
      the whole trajectory (rewards short-horizon fidelity over a mean-collapsed
      long trajectory).
    - var_weight > 0 (option 2): add a per-mode std-deficit penalty so a damped
      (low-variance) rollout is penalized, not just a pointwise-wrong one.
    """
    if window_starts:
        errs = [
            la.norm(weighted_lp_error(
                Q_true[:, s:s + window_len], Q_ROM[:, s:s + window_len],
                weights=weights, normalize=True))
            for s in window_starts
        ]
        score = float(np.mean(errs))
    else:
        score = float(la.norm(weighted_lp_error(Q_true, Q_ROM, weights=weights, normalize=True)))

    if var_weight:
        std_true = Q_true.std(axis=1)
        std_rom = Q_ROM.std(axis=1)
        deficit = np.maximum(0.0, 1.0 - std_rom / np.maximum(std_true, 1e-12))
        score += var_weight * float(la.norm(deficit))

    return score


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


def _resolve_reg_N(reg_N, reg_A, reg_H):
    """Resolve the N regularization for one candidate.

    reg_N may be: None (-> tied to reg_H inside _make_regularizer), a number
    (fixed), or the string 'A'/'H' to tie N to the candidate's reg_A/reg_H.
    """
    if reg_N is None:
        return None
    if isinstance(reg_N, str):
        key = reg_N.strip().lower().lstrip("reg_")
        if key == "a":
            return reg_A
        if key == "h":
            return reg_H
        raise ValueError(f"reg_N string must be 'A' or 'H', got {reg_N!r}")
    return float(reg_N)


def _learned_operator_templates(learn_N=False):
    ops = [
        opinf.operators.ConstantOperator(),
        opinf.operators.LinearOperator(),
        opinf.operators.QuadraticOperator(),
    ]
    if learn_N:
        # Bilinear state-input operator N(u (x) q): makes the internal dynamics
        # depend on the forcing amplitude. Inferred (unlike imposed B).
        ops.append(opinf.operators.StateInputOperator())
    return ops


def _make_regularizer(reg_A, reg_H, r, *, learn_N=False, reg_N=None, m=None):
    reg_c = reg_A
    params = [reg_c, reg_A, reg_H]
    kwargs = dict(state_dimension=r)
    if learn_N:
        if m is None:
            raise ValueError("input dimension m is required when learn_N=True")
        params.append(reg_H if reg_N is None else reg_N)  # tie N to H by default
        kwargs["input_dimension"] = m
    return opinf.lstsq.TikhonovSolver.get_operator_regularizer(
        operators=_learned_operator_templates(learn_N=learn_N),
        regularization_parameters=params,
        **kwargs,
    )


def _make_solver(reg_A, reg_H, r, regularization_target=None, *, learn_N=False, reg_N=None, m=None):
    regularizer = _make_regularizer(reg_A, reg_H, r, learn_N=learn_N, reg_N=reg_N, m=m)
    if regularization_target is None:
        return opinf.lstsq.TikhonovSolver(regularizer, method="lstsq")
    return ShiftedTikhonovSolver(
        regularizer,
        target_operator_matrix=regularization_target,
        method="lstsq",
    )


def _make_model(B, C, solver, *, learn_N=False):
    operators = [
        opinf.operators.InputOperator(entries=B),
        opinf.operators.ConstantOperator(entries=C),
        opinf.operators.ConstantOperator(),
        opinf.operators.LinearOperator(),
        opinf.operators.QuadraticOperator(),
    ]
    if learn_N:
        operators.append(opinf.operators.StateInputOperator())
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
    stability_inputs=None,
    stability_niters=None,
    stability_bound=np.inf,
    learn_N=False,
    reg_N=None,
    m=None,
    var_weight=0.0,
    window_starts=None,
    window_len=None,
):
    """Fit ROM with given regularization and return (converged, error).

    If ``stability_inputs`` is given, each candidate is also rolled out over the
    full ``stability_niters``-step horizon (one rollout per input series, started
    from the first training IC). Candidates that go non-finite or exceed
    ``stability_bound`` anywhere in that horizon are rejected (converged=False),
    so only operators stable across the entire prediction window are selectable.
    """

    if reg_A > reg_H:
        return False, np.inf

    rom = _make_model(
        B,
        C,
        solver=_make_solver(reg_A, reg_H, r, regularization_target, learn_N=learn_N,
                            reg_N=_resolve_reg_N(reg_N, reg_A, reg_H), m=m),
        learn_N=learn_N,
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

        scenario_errors.append(score_trajectory(
            Qs_[i], Q_ROM_, weights=weights,
            window_starts=window_starts, window_len=window_len, var_weight=var_weight,
        ))

    # Full-horizon stability gate: reject operators that blow up over the entire
    # prediction window, not just the training window.
    if stability_inputs is not None and stability_niters:
        q0 = Qs_[0][:, 0]
        for u_full in stability_inputs:
            Q_full = model.predict(q0, niters=stability_niters, inputs=u_full)
            if not np.all(np.isfinite(Q_full)) or np.max(np.abs(Q_full)) > stability_bound:
                return False, np.inf

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
        stability_inputs=None,
        stability_niters=None,
        stability_bound=np.inf,
        learn_N=False,
        reg_N=None,
        var_weight=0.0,
        window_starts=None,
        window_len=None,
    ):
        self.Qs_ = Qs_
        self.inputs = inputs
        self.B = B
        self.C = C
        self.r = r
        self.nt = nt
        self.t = t
        self.learn_N = learn_N
        self.reg_N = reg_N  # independent reg for N; None -> tied to reg_H (inert when reg_H large)
        # input dimension m, needed to size the N regularizer (inputs: list of (m, nt))
        self.m = int(np.asarray(inputs[0]).shape[0]) if learn_N else None
        self.var_weight = var_weight          # option 2: variance-deficit penalty
        self.window_starts = window_starts    # option 3: short-window scoring
        self.window_len = window_len

        self.weights_A = weights_A
        self.weights_H = weights_H
        self.regularization_target = regularization_target
        self.stability_inputs = stability_inputs
        self.stability_niters = stability_niters
        self.stability_bound = stability_bound
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
                    stability_inputs=self.stability_inputs,
                    stability_niters=self.stability_niters,
                    stability_bound=self.stability_bound,
                    learn_N=self.learn_N,
                    reg_N=self.reg_N,
                    m=self.m,
                    var_weight=self.var_weight,
                    window_starts=self.window_starts,
                    window_len=self.window_len,
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
                    stability_inputs=self.stability_inputs,
                    stability_niters=self.stability_niters,
                    stability_bound=self.stability_bound,
                    learn_N=self.learn_N,
                    reg_N=self.reg_N,
                    m=self.m,
                    var_weight=self.var_weight,
                    window_starts=self.window_starts,
                    window_len=self.window_len,
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
            if self.stability_inputs is not None:
                raise RuntimeError(
                    "No regularization candidate was stable over the full "
                    f"{self.stability_niters}-step horizon (stability_bound="
                    f"{self.stability_bound:g}). Widen weights_H upward, raise "
                    "stability_bound, or shorten n_year_predict."
                )
            raise RuntimeError("run() or load() must be called first.")

        rom = _make_model(
            self.B,
            self.C,
            solver=_make_solver(
                self.best_reg_A,
                self.best_reg_H,
                self.r,
                self.regularization_target,
                learn_N=self.learn_N,
                reg_N=_resolve_reg_N(self.reg_N, self.best_reg_A, self.best_reg_H),
                m=self.m,
            ),
            learn_N=self.learn_N,
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
