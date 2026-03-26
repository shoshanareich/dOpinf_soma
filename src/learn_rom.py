import numpy as np
import numpy.linalg as la
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




def fit_and_score_model(reg_A, reg_H, *, Q_, inputs, B, C, r, nt, t, weights=None):
    """Fit ROM with given regularization and return (converged, error)."""

    if reg_A > reg_H:
        return False, np.inf

    reg_c = reg_A

    regularizer = opinf.lstsq.TikhonovSolver.get_operator_regularizer(
        operators=[
            opinf.operators.ConstantOperator(),
            opinf.operators.LinearOperator(),
            opinf.operators.QuadraticOperator(),
          #  opinf.operators.InputOperator(),
          #  opinf.operators.CubicOperator(),
        ],
        regularization_parameters=[reg_c, reg_A, reg_H],
        state_dimension=r,
        #input_dimension=2,
    )

    solver = opinf.lstsq.TikhonovSolver(regularizer, method="lstsq")

    operators = [
        opinf.operators.InputOperator(entries=B),
        opinf.operators.ConstantOperator(entries=C),
        opinf.operators.ConstantOperator(),
        opinf.operators.LinearOperator(),
        opinf.operators.QuadraticOperator(),
        #opinf.operators.InputOperator(),
       # opinf.operators.CubicOperator()
    ]

    # Check for NaNs or Infs
    if not np.isfinite(Q_).all():
        print("Warning: Q_hat contains NaNs or Infs!")
            
    model = opinf.models.DiscreteModel(operators, solver=solver)
    model = model.fit(states=Q_, inputs=inputs)

    Q_ROM_ = model.predict(Q_[:, 0], niters=nt, inputs=inputs)

    if np.isnan(Q_ROM_).any():
        return False, np.inf
    

    abs_l2err = weighted_lp_error(
        Q_[:, :len(t)], 
        Q_ROM_, 
        weights=weights,  # Ensure this matches the spatial dimension of Q_
        normalize=True
    )

    # abs_l2err, _ = opinf.post.lp_error(
    #     Q_[:, :len(t)], Q_ROM_, normalize=True
    # )

    return True, la.norm(abs_l2err)




class TikhonovSweep:
    def __init__(self, *, Q_, inputs, B, C, r, nt, t, weights_A, weights_H, norm_weights=None):
        self.Q_ = Q_
        self.inputs = inputs
        self.B = B
        self.C = C
        self.r = r
        self.nt = nt
        self.t = t

        self.weights_A = weights_A
        self.weights_H = weights_H
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
                print(f"iteration {i+1} / {len(self.weights_H)}")

            for j, reg_A in enumerate(self.weights_A):
                converged, err = fit_and_score_model(
                    reg_A,
                    reg_H,
                    Q_=self.Q_,
                    inputs=self.inputs,
                    B=self.B,
                    C=self.C,
                    r=self.r,
                    nt=self.nt,
                    t=self.t,
                    weights=self.norm_weights
                )

                self.converged[i, j] = converged
                self.errors[i, j] = err

                if converged and err < self.best_error:
                    self.best_error = err
                    self.best_reg_A = reg_A
                    self.best_reg_H = reg_H

        return self.best_reg_A, self.best_reg_H, self.best_error


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


    def fit_best_model(self, *, inputs_fit, niters):
        'fit using best regs'
        if self.best_reg_A is None:
            raise RuntimeError("run() or load() must be called first.")

        reg_A = self.best_reg_A
        reg_H = self.best_reg_H
        reg_c = reg_A

        regularizer = opinf.lstsq.TikhonovSolver.get_operator_regularizer(
            operators=[
                opinf.operators.ConstantOperator(),
                opinf.operators.LinearOperator(),
                opinf.operators.QuadraticOperator(),
                #opinf.operators.InputOperator(),
               # opinf.operators.CubicOperator(),
            ],
            regularization_parameters=[reg_c, reg_A, reg_H],
            state_dimension=self.r,
          #  input_dimension=2,
        )

        solver = opinf.lstsq.TikhonovSolver(regularizer)

        operators = [
            opinf.operators.InputOperator(entries=self.B),
            opinf.operators.ConstantOperator(entries=self.C),
            opinf.operators.ConstantOperator(),
            opinf.operators.LinearOperator(),
            opinf.operators.QuadraticOperator(),
           # opinf.operators.InputOperator(),
            #opinf.operators.CubicOperator(),
        ]

        model = opinf.models.DiscreteModel(operators, solver=solver)
        model = model.fit(states=self.Q_, inputs=inputs_fit)

        Q_ROM_ = model.predict(
            self.Q_[:, 0],
            niters=niters,
            inputs=inputs_fit,
        )

        return model, Q_ROM_
