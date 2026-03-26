"""Operator inference functions."""
import numpy as np
import scipy as sp
import opinf
import numpy.linalg as la




# Utils -----------------------------------------------------------------------

def kron(y1, y2):
    """
    Implements naive kronecker product between two
    vectors or matrices.

    TAKEN DIRECTLY FROM SHANE MCQUARRIE's OPINF PACKAGE.
    """
    ys = [y1[i] * y2 for i in range(y1.shape[0])]
    return np.concatenate(ys, axis=0)


def ckron(y):
    """
    Implements compact (no redundant terms) kronecker product between two
    vectors or matrices.

    TAKEN DIRECTLY FROM SHANE MCQUARRIE's OPINF PACKAGE.
    """
   # print('ckron')
   # print(y.shape)
    ys = [y[i] * y[:i+1] for i in range(y.shape[0])]
    return np.concatenate(ys, axis=0)




# # Dynamics --------------------------------------------------------------------

def rhs(t, q0, ops, forms, rs, ms, Y):
    """ dynamics rhs."""

  #  print(f'ms are: {ms}')
    # unpack ics and inputs
    ri, mi = 0, 0 # r x input dimension
    q0s, ys = [], []
    for rf, mf in zip(rs, ms):
        q0s.append(q0[ri:ri+rf])
        ys.append(Y(t)[mi:mi+mf])
      #  print('ys shape')
      #  print(Y(t)[mi:mi+mf].shape)
        ri += rf
        mi += mf

    # loop through domains
    qs = []
    for form, opsi, q0i, yi in zip(
        forms.values(), ops, q0s, ys,
    ):

       # print(form)
        # initialize rhs
        qi = np.zeros_like(q0i)

        # evaluate operators for this domain
        for op, optype in zip(opsi, form):
            
           # print(optype)
            # constant
            if optype == "C":
               # print(op.squeeze().shape)
                qi += op.squeeze()

              #  print('constant done')
            # linear
            elif optype == "A1":
               # print(op @ q0s[0])
                qi += op @ q0s[0]
            elif optype == "A2":
               # print(op @ q0s[1])
                qi += op @ q0s[1]
                
            elif optype == "A3":
               # print(op @ q0s[2])
                qi += op @ q0s[2]
     

            # quadratic
            elif optype == "H11":
                qoi_kron = ckron(q0s[0])
               # print(qoi_kron.shape)
                qi += op @ qoi_kron
               # print(qi.shape)
            elif optype == "H12":
                qoi_kron = kron(q0s[0], q0s[1])
                qi += op @ qoi_kron
                #print(qi.shape)
            elif optype == "H13":
                qoi_kron = kron(q0s[0], q0s[2])
                qi += op @ qoi_kron
               # print(qi.shape)
            elif optype == "H22":
                qoi_kron = ckron(q0s[1])
                qi += op @ qoi_kron
               # print(qi.shape)
            elif optype == "H23":
                qoi_kron = kron(q0s[1], q0s[2])
                qi += op @ qoi_kron
               # print(qi.shape)
            elif optype == "H33":
                qoi_kron = ckron(q0s[2])
                qi += op @ qoi_kron
               # print(qi.shape)

            # input
            elif optype.startswith("B"):
              #  print(f'B inputs shape: {yi.shape}')
                qi += op @ yi

        qs.append(qi)

    q = np.hstack(qs)

    return q



# Operator inference ----------------------------------------------------------


# def compute_compact_r2(r):
#     """Computes dimension of compact kronecker product output."""
#     return r * (r + 1) // 2


def compute_kmin(r, form, m=1):
    """Computes minimum number of snapshots needed for a given r, m."""
    kmin = 0
    if "c" in form:
        kmin += 1
    if "A" in form:
        kmin += r
    if "H" in form:
        kmin += quad_r(r)
    if "B" in form:
        kmin += m

    return kmin


def build_data_matrix(form, Q1_, Q2_, Q3_, y1, y3):
    """Assemble data matrix for operator inference least squares."""
    ktrain = Q1_.shape[1]
    # print(np.ones((1, ktrain)).T.shape)
    # print(Q1_.shape, Q2_.shape, Q3_.shape)
    # print(ckron(Q1_).shape)
    # Assemble matrix
    Dlist = []
    if "C" in form:
        Dlist.append(np.ones((1, ktrain)).T)

    if "A1" in form:
        Dlist.append(Q1_.T)
    if "A2" in form:
        Dlist.append(Q2_.T)
    if "A3" in form:
        Dlist.append(Q3_.T)
    if "H11" in form:
        Qkron_ = ckron(Q1_)
        Dlist.append(Qkron_.T)
    if "H12" in form:
        Qkron_ = kron(Q1_, Q2_)
        Dlist.append(Qkron_.T)
    if "H13" in form:
        Qkron_ = kron(Q1_, Q3_)
        Dlist.append(Qkron_.T)
    if "H22" in form:
        Qkron_ = ckron(Q2_)
        Dlist.append(Qkron_.T)
    if "H23" in form:
        Qkron_ = kron(Q2_, Q3_)
        Dlist.append(Qkron_.T)
    if "H33" in form:
        Qkron_ = ckron(Q3_)
        Dlist.append(Qkron_.T)
    if "B1" in form:
        Dlist.append(y1[np.newaxis,:].T)
    if "B3" in form:
        Dlist.append(y3[np.newaxis,:].T)
    D = np.hstack(Dlist)

    return D


def build_rhs_matrix(dQ_):
    """Assemble right-hand-side matrix for operator inference least squares."""

    # Assemble matrix
    R = dQ_.T

    return R


def quad_r(r):
    return r * (r + 1) // 2

def build_regularization_matrix(form, gammas, r1, r2, r3, m=1):
    """Assemble regularization matrix for operator inference least squares."""
   # r2 = r * (r + 1) // 2

    # Assemble matrix
    Plist = []
    if "C" in form:
        Plist.append(np.ones((1,))*gammas["C"])
    if "A1" in form:
        Plist.append(np.ones((r1,))*gammas["A"])
    if "A2" in form:
        Plist.append(np.ones((r2,))*gammas["A"])
    if "A3" in form:
        Plist.append(np.ones((r3,))*gammas["A"])

    if "H11" in form:
        Plist.append(np.ones((quad_r(r1),))*gammas["H"])
    if "H12" in form:
        Plist.append(np.ones((r1*r2,))*gammas["H"])
    if "H13" in form:
        Plist.append(np.ones((r1*r3,))*gammas["H"])
    if "H22" in form:
        Plist.append(np.ones((quad_r(r2),))*gammas["H"])
    if "H23" in form:
        Plist.append(np.ones((r2*r3,))*gammas["H"])
    if "H33" in form:
        Plist.append(np.ones((quad_r(r3),))*gammas["H"])
    
    if "B1" in form:
        Plist.append(np.ones((m,))*gammas["B"]) # m is input dimension
    if "B3" in form:
        Plist.append(np.ones((m,))*gammas["B"]) # m is input dimension
    if "G" in form:
        Plist.append(np.ones((r2*m,))*gammas["G"])
    
    P = np.diag(np.hstack(Plist))

    return P


def extract_operators(O, form, r1, r2, r3, m=1):
    """Unpack operators from least squares solution."""
   # r2 = r * (r + 1) // 2
   # r2m = r2 * m

    operators = []
    i = 0
    if "C" in form:
        c_ = O[:, i:i+1]
        operators.append(c_)
        i += 1
    if "A1" in form:
        A_ = O[:, i:i+r1]
        operators.append(A_)
        i += r1
    if "A2" in form:
        A_ = O[:, i:i+r2]
        operators.append(A_)
        i += r2
    if "A3" in form:
        A_ = O[:, i:i+r3]
        operators.append(A_)
        i += r3
    if "H11" in form:
        H_ = O[:, i:i+quad_r(r1)]
        operators.append(H_)
        i += quad_r(r1)
    if "H12" in form:
        H_ = O[:, i:i+r1*r2]
        operators.append(H_)
        i += r1*r2
    if "H13" in form:
        H_ = O[:, i:i+r1*r3]
        operators.append(H_)
        i += r1*r3
    if "H22" in form:
        H_ = O[:, i:i+quad_r(r2)]
        operators.append(H_)
        i += quad_r(r2)
    if "H23" in form:
        H_ = O[:, i:i+r2*r3]
        operators.append(H_)
        i += r2*r3
    if "H33" in form:
        H_ = O[:, i:i+quad_r(r3)]
        operators.append(H_)
        i += quad_r(r3)
    if "B1" in form:
        B_ = O[:, i:i+m]
        operators.append(B_)
        i += m
    if "B3" in form:
        B_ = O[:, i:i+m]
        operators.append(B_)
        i += m

    return operators


def lstsq(form, Q1_, Q2_, Q3_, dQ_, gammas, y1, y3):
    """Infer operators."""
    r1 = Q1_.shape[0]
    r2 = Q2_.shape[0]
    r3 = Q3_.shape[0]

    D = build_data_matrix(form, Q1_, Q2_, Q3_, y1, y3)
    R = build_rhs_matrix(dQ_)
    P = build_regularization_matrix(form, gammas, r1, r2, r3)

    # Infer operators
    Dpad = np.vstack((D, P))
    Rpad = np.vstack((R, np.zeros((P.shape[0], R.shape[1]))))
    O = sp.linalg.lstsq(Dpad, Rpad)[0].T

    #print(O.shape)

    operators = extract_operators(O, form, r1, r2, r3)
   # print('here')
    #for ops in operators:
       # print(ops.shape)

    return operators




def predict_continuous(operators, t, Q0_, forms, rs, ms, Y):
    """ Given operators, run model forward in time"""

    #rhs(q0, ops, forms, rs, ms, Y)
    # integrate
   # print('in predict')
   # print(Y.shape)
    input_func = sp.interpolate.CubicSpline(t, Y, axis=1)
    #ytest = input_func(t[1])
   # print(f'ytest shape: {ytest.shape}')

    result = sp.integrate.solve_ivp(rhs, (t[0], t[-1]), Q0_, t_eval=t, args=(operators, forms, rs, ms, input_func))
    print('ivp results')
    print(result)

    return result.y

def predict_discrete(operators, state0, niters, forms, rs, ms, inputs):
    """
    Given discrete operators, evolve the system forward: 
    q_{k+1} = f(q_k, u_k)
    """
    r_total = sum(rs)
    Q_ROM = np.zeros((r_total, niters))
    Q_ROM[:, 0] = state0

    # We reuse the 'rhs' function. 
    # In discrete mode, 't' is just an index for the input function.
    def input_func(t):
        # t is passed as a float by the logic, so we cast to int for indexing
        return inputs[:, int(t)]

    for k in range(niters - 1):
        # We pass k as 't' so the input_func grabs the correct column
        Q_ROM[:, k+1] = rhs(k, Q_ROM[:, k], operators, forms, rs, ms, input_func)

    return Q_ROM

    # """ Given operators, run model forward in time"""
    
    # # Create the solution array and fill in the initial condition.
    # states = np.empty((state0.shape[0], niters))
    # states[:, 0] = state0.copy()

    # # Validate shape of input, reshaping if input is 1d.
    # U = np.atleast_2d(inputs)
    # for j in range(niters - 1):
    #     states[:, j + 1] = self.rhs(states[:, j], U[:, j])

    # return states



def fit_and_score_model(reg_A, reg_H, Q1_, Q2_, Q3_, y1, y3, nt, t, model_type='discrete'):
    """Fit ROM with given regularization and return (converged, error)."""

    if reg_A > reg_H:
        return False, np.inf

    reg_c = reg_A

    gammas = {"C": reg_c, "A": reg_A, "H": reg_H, "B": reg_A}

    forms = {"1": ["C", "A1", "A2", "A3", "H11", "B1"],
             "2": ["C", "A1"],
             "3": ["C", "A3", "H13", "B3"]}
    
    time_derivate_estimator = opinf.ddt.UniformFiniteDifferencer(t, "ord6")
    _, dQ1_ = time_derivate_estimator.estimate(Q1_)
    _, dQ2_ = time_derivate_estimator.estimate(Q2_)
    _, dQ3_ = time_derivate_estimator.estimate(Q3_)

    # Check for NaNs or Infs
    if not np.isfinite(Q1_).all():
        print("Warning: Q1_hat contains NaNs or Infs!")
    if not np.isfinite(Q2_).all():
        print("Warning: Q2_hat contains NaNs or Infs!")
    if not np.isfinite(Q3_).all():
        print("Warning: Q3_hat contains NaNs or Infs!")
            
 #   model = opinf.models.DiscreteModel(operators, solver=solver)
#   model = model.fit(states=Q_, inputs=inputs)


    # learn operators FOR CONTINUOUS CASE
    if model_type == 'continuous':
        ops1 = lstsq(forms["1"], Q1_, Q2_, Q3_, dQ1_, gammas, y1, y3)
        ops2 = lstsq(forms["2"], Q1_, Q2_, Q3_, dQ2_, gammas, y1, y3)
        ops3 = lstsq(forms["3"], Q1_, Q2_, Q3_, dQ3_, gammas, y1, y3)
    
    elif model_type == 'discrete':
        # learn operators FOR DISCRETE CASE
        nextQ1_ = Q1_[:, 1:]
        nextQ2_ = Q2_[:, 1:]
        nextQ3_ = Q3_[:, 1:]
        #inputs = inputs[..., : states.shape[1]]
        ops1 = lstsq(forms["1"], Q1_[:,:-1], Q2_[:,:-1], Q3_[:,:-1], nextQ1_, gammas, y1[:-1], y3[:-1])
        ops2 = lstsq(forms["2"], Q1_[:,:-1], Q2_[:,:-1], Q3_[:,:-1], nextQ2_, gammas, y1[:-1], y3[:-1])
        ops3 = lstsq(forms["3"], Q1_[:,:-1], Q2_[:,:-1], Q3_[:,:-1], nextQ3_, gammas, y1[:-1], y3[:-1])
     #   print('fit discrete works')
        

    # for ops in ops1:
    #     print(ops.shape)

    # run for training period
    rs = [Q1_.shape[0], Q2_.shape[0], Q3_.shape[0]]
    ms = [1, 0, 1]
  #  print(f'ms {ms}')
  #  print(f'inpute y1 shape: {y1.shape[0]}')
    Q0_ = np.hstack([Q1_[:,0], Q2_[:,0], Q3_[:,0]])
  #  print(f'Q0_ shape: {Q0_.shape}')

    tmp = np.vstack([y1, np.zeros_like(y1), y3]).T
   # print('inputs shape')
  #  print(tmp.shape)

   # predict(operators, t, Q0_, forms, rs, ms, Y)
    if model_type == 'continuous':
        Q_ROM_ = predict_continuous([ops1, ops2, ops3], t, Q0_, forms, rs, ms,np.vstack([y1, np.zeros_like(y1), y3]))
        
    elif model_type == 'discrete':
        U_inputs = np.vstack([y1, np.zeros_like(y1), y3]) 
        Q_ROM_ = predict_discrete([ops1, ops2, ops3], Q0_, len(t), forms, rs, ms, U_inputs)

   # print('We got a rom prediction!')
   # print(Q_ROM_.shape)
    Q_ = np.vstack([Q1_, Q2_, Q3_])


    # get error     
    if np.isnan(Q_ROM_).any():
        return False, np.inf

    abs_l2err, _ = opinf.post.lp_error(
        Q_[:, :len(t)], Q_ROM_, normalize=True
    )


    
 #   la.norm()

 #   Q_ROM_ = model.predict(Q_[:, 0], niters=nt, inputs=inputs)

 

   # error = 

    return la.norm(abs_l2err)






class BlockTikhonovSweep:
    def __init__(self, *, Q1_, Q2_, Q3_, y1, y3, nt, t, weights_A, weights_H, model_type='discrete'):
        # Data snapshots for each block
        self.Q1_, self.Q2_, self.Q3_ = Q1_, Q2_, Q3_
        self.y1, self.y3 = y1, y3
        self.nt, self.t = nt, t
        self.model_type = model_type

        # Regularization grids
        self.weights_A = weights_A
        self.weights_H = weights_H

        # Results storage
        self.errors = np.full((len(weights_H), len(weights_A)), np.inf)
        self.best_reg_A = None
        self.best_reg_H = None
        self.best_error = np.inf

    def run(self, verbose=True):
        """Sweep through regularization parameters."""
        for i, reg_H in enumerate(self.weights_H):
            if verbose:
                print(f"Testing reg_H: {reg_H:.2e} ({i+1}/{len(self.weights_H)})")

            for j, reg_A in enumerate(self.weights_A):
                try:
                    # Execute the fit and score
                    err = fit_and_score_model(
                        reg_A, reg_H, 
                        self.Q1_, self.Q2_, self.Q3_, 
                        self.y1, self.y3, 
                        self.nt, self.t, 
                        model_type=self.model_type
                    )

                    # Handle case where fit_and_score might return a tuple or list
                    if isinstance(err, (tuple, list, np.ndarray)):
                        # If it's an array of errors over time, take the Frobenius norm
                        err = la.norm(err)
                    
                    # Check for stability (NaN or Inf)
                    if not np.isfinite(err):
                        err = 1e10 # High penalty for unstable models

                except (ValueError, TypeError, RuntimeWarning) as e:
                    # Catch the 'sequence' error or math errors and keep moving
                    if verbose:
                        print(f"  -> Model at reg_A={reg_A:.1e}, reg_H={reg_H:.1e} failed: {e}")
                    err = 1e10

                self.errors[i, j] = float(err)

                if err < self.best_error:
                    self.best_error = err
                    self.best_reg_A = reg_A
                    self.best_reg_H = reg_H

        if verbose:
            print(f"\nSweep Complete. Best Error: {self.best_error:.4e}")
            print(f"Best Reg_A: {self.best_reg_A:.2e}, Best Reg_H: {self.best_reg_H:.2e}")

        return self.best_reg_A, self.best_reg_H, self.best_error

    def fit_best_model(self):
        """Final fit using the best identified parameters to return operators."""
        if self.best_reg_A is None:
            raise RuntimeError("Run the sweep first!")

        # We essentially re-run the 'fit' logic of fit_and_score_model 
        # but return the actual operators instead of just a score.
        gammas = {
            "C": self.best_reg_A, 
            "A": self.best_reg_A, 
            "H": self.best_reg_H, 
            "B": self.best_reg_A
        }

        forms = {
            "1": ["C", "A1", "A2", "A3", "H11", "B1"],
            "2": ["C", "A1"],
            "3": ["C", "A3", "H13", "B3"]
        }

        if self.model_type == 'discrete':
            # Setup discrete training data
            ops1 = lstsq(forms["1"], self.Q1_[:,:-1], self.Q2_[:,:-1], self.Q3_[:,:-1], self.Q1_[:,1:], gammas, self.y1[:-1], self.y3[:-1])
            ops2 = lstsq(forms["2"], self.Q1_[:,:-1], self.Q2_[:,:-1], self.Q3_[:,:-1], self.Q2_[:,1:], gammas, self.y1[:-1], self.y3[:-1])
            ops3 = lstsq(forms["3"], self.Q1_[:,:-1], self.Q2_[:,:-1], self.Q3_[:,:-1], self.Q3_[:,1:], gammas, self.y1[:-1], self.y3[:-1])
        else:
            # Continuous logic (requires dQ calculation)
            # ... (implement similar to fit_and_score_model)
            pass

        return [ops1, ops2, ops3]