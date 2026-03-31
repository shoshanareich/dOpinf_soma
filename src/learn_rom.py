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




def fit_and_score_model(reg_A, reg_H, *, Qs_, inputs, B, C, r, nt, t, weights=None):
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
       # input_dimension=len(inputs),
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
            
    #model = opinf.models.DiscreteModel(operators, solver=solver)
    #model = model.fit(states=Qs_, inputs=inputs)
    rom = opinf.ROM(model=opinf.models.DiscreteModel(operators, solver=solver))
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
    def __init__(self, *, Qs_, inputs, B, C, r, nt, t, weights_A, weights_H, norm_weights=None):
        self.Qs_ = Qs_
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
                    Qs_=self.Qs_,
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


    def fit_best_model(self):
        """
        Retrain the model using the best regularization found during the sweep
        across ALL training scenarios and return the trained model object.
        """
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
          #  input_dimension=len(self.inputs),
        )

        solver = opinf.lstsq.TikhonovSolver(regularizer, method='lstsq')

        operators = [
            opinf.operators.InputOperator(entries=self.B),
            opinf.operators.ConstantOperator(entries=self.C),
            opinf.operators.ConstantOperator(),
            opinf.operators.LinearOperator(),
            opinf.operators.QuadraticOperator(),
           # opinf.operators.InputOperator(),
            #opinf.operators.CubicOperator(),
        ]

        # model = opinf.models.DiscreteModel(operators, solver=solver)

        # # Fit with all training states and inputs
        # model = model.fit(states=self.Qs_, inputs=self.inputs)

        rom = opinf.ROM(model=opinf.models.DiscreteModel(operators, solver=solver))
        model = rom.fit(states=self.Qs_, inputs=self.inputs)

        return model
