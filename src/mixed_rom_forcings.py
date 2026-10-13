import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
os.environ.setdefault("UCX_TLS", "rc,sm,self")

from mpi4py import MPI
import sys
import time
from datetime import datetime
import paths

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
script_start = time.perf_counter()


def log_timing(message, start_time=None):
    if rank != 0:
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time is None:
        print(f"[timing] {stamp} | {message}", flush=True)
    else:
        elapsed = time.perf_counter() - start_time
        print(f"[timing] {stamp} | {message} | elapsed {elapsed / 60:.2f} min ({elapsed:.1f} s)", flush=True)

log_timing(f"mixed_rom_forcings.py start; ranks initialized: {size}")
log_timing("Starting library imports")

import numpy as np
import xarray as xr
import netCDF4 as nc
import xmitgcm 
import json
import sys
from MITgcmutils import rdmds

from config_utils import resolve_forcing_config
from utils import *
from shiftscale import *
from svd import SVDDecomposition
from learn_rom import TikhonovSweep, fit_sequential_model
import gc # garbage cleanup

if MPI.COMM_WORLD.Get_rank() == 0:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    from plotting import ROMPlotter
    from forcing import *
    log_timing("Library imports finished; setting vars", script_start)


#root_dir = '/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/'
# Override with ROM_ROOT_DIR (e.g. when /scratch2 is down, point at /scratch).
root_dir = os.environ.get('ROM_ROOT_DIR', paths.RESULTS_ROOT)

## spatial vars
nx = 248
ny = nx
nz = 31

N = nx*ny*nz



### training config

# Read in config file on rank 0 only 
if rank == 0:
    config_path = os.environ.get("ROM_CONFIG_PATH", os.path.join(root_dir, "config.json"))
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = None

# Broadcast config to all ranks
config = comm.bcast(config, root=0)

# Extract training options
n_year_train = config['n_year_train']
n_year_predict = config['n_year_predict']
n_days = config['n_days']
center_opt = config['center_opt']
scale = config['scale']
dir_extension = config.get('dir_extension', 'default_extension')
block_years = config.get('block_years')
learn_N = bool(config.get('learn_N', False))  # option 1: bilinear state-input operator N(u⊗q)
reg_N = config.get('reg_N')  # independent reg for N (None -> tied to reg_H; ~1e3 works)
var_weight = float(config.get('var_weight', 0.0))  # option 2: variance-deficit penalty in the loss
n_windows = int(config.get('n_windows', 0))        # option 3: # of short scoring windows (0 -> off)
window_months = float(config.get('window_months', 3.0))  # option 3: window length in months
window_seed = int(config.get('window_seed', 0))    # option 3: RNG seed for window starts
run_label = config.get('run_label')  # optional suffix on the results dir to separate runs sharing center/r
forcing_config = resolve_forcing_config(config)
train_taus = forcing_config['train_taus']
test_taus = forcing_config['test_taus']
snapshot_dirs = forcing_config['snapshot_dirs']
test_dirs = forcing_config['test_dirs']

if any(tau is None for tau in train_taus):
    raise ValueError("Define 'train_taus' in config.json, or include tau values in each snapshot directory name")
if any(tau is None for tau in test_taus):
    raise ValueError("Define 'test_taus' in config.json, or include tau values in each test directory name")

test_dir = test_dirs[0] if test_dirs else ''

# check for target_ret_energy or r
target_ret_energy = config.get('target_ret_energy')
r = config.get('r')
svd_kwargs = {
    'gram_chunk_rows': config.get('svd_gram_chunk_rows'),
    'dense_block_gb': config.get('svd_dense_block_gb'),
}

if (target_ret_energy is None and r is None) or (target_ret_energy is not None and r is not None):
    raise ValueError("Must define either 'target_ret_energy' OR 'r' in config.json")

if not snapshot_dirs:
    raise ValueError("Must define 'snapshot_dirs' or train tau settings in config.json")

if not test_dirs:
    raise ValueError("Must define 'test_dir', 'test_dirs', or test tau settings in config.json")

if len(train_taus) != len(snapshot_dirs):
    raise ValueError("'train_taus' must have the same length as 'snapshot_dirs'")

if len(test_taus) != len(test_dirs):
    raise ValueError("'test_taus' must have the same length as test directories")

if block_years is not None:
    block_years = int(block_years)
    if block_years <= 0:
        raise ValueError("'block_years' must be a positive integer")

n_days_per_year = int(360 / n_days)
nt = n_days_per_year * n_year_train 
t = np.linspace(0, n_year_train * 360 * 24 * 3600, nt)

if rank == 0:
    print('Parameters loaded from config.json. Opening snapshots', flush=True)



preproc_dir = None
# Create the results directory
if rank == 0:
    preproc_dir = root_dir + 'save_roms/' + dir_extension + '/'
    os.makedirs(preproc_dir, exist_ok=True)

preproc_dir = comm.bcast(preproc_dir, root=0)



## split data and select training snapshots


# loop over training forcings
U_rank_list = []
V_rank_list = []
Eta_rank_list = []
T_rank_list = []
S_rank_list = []

i_chunk = max(1, nx // size)
time_chunk = min(360, 360 * n_year_train)

for snapshot_dir in snapshot_dirs:
    ds = xr.open_dataset(
        snapshot_dir + 'states_20yrs.nc',
        engine="netcdf4",
        decode_timedelta=False,
        decode_times=False,
        chunks={"time": time_chunk, "i": i_chunk, "i_g": i_chunk},
    )

    # Fail loudly if the backend dropped data variables (h5netcdf did this silently
    # for some files on the compute nodes -> cryptic KeyError later in reshape_data).
    missing = [v for v in ("U", "V", "Eta", "T", "S") if v not in ds.variables]
    if missing:
        raise KeyError(
            f"{snapshot_dir}states_20yrs.nc opened without {missing}; "
            f"present variables: {list(ds.variables)}"
        )

    U_rank, V_rank, Eta_rank, T_rank, S_rank = reshape_data(nx, n_days, n_year_train, ds, rank, size)
    
    U_rank_list.append(U_rank)
    V_rank_list.append(V_rank)
    Eta_rank_list.append(Eta_rank)
    T_rank_list.append(T_rank)
    S_rank_list.append(S_rank)

U_rank = np.hstack(U_rank_list)
V_rank = np.hstack(V_rank_list)
Eta_rank = np.hstack(Eta_rank_list)
T_rank = np.hstack(T_rank_list)
S_rank = np.hstack(S_rank_list)


shiftscale_start = time.perf_counter()
log_timing("Starting shiftscale")


def gather_global_alpha(local_alpha, actual_nz, nx, ny, comm):
    """Gather rank-local alpha vectors into the global flattened spatial order."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    alpha_arr = np.asarray(local_alpha)

    if alpha_arr.ndim == 0 or alpha_arr.size == 1:
        return float(alpha_arr.reshape(-1)[0]) if rank == 0 else None

    local_vec = np.ascontiguousarray(alpha_arr.reshape(-1), dtype=np.float64)
    all_pieces = comm.gather(local_vec, root=0)

    if rank != 0:
        return None

    global_alpha = np.zeros(actual_nz * ny * nx, dtype=np.float64)
    target = global_alpha.reshape((actual_nz, ny, nx))

    for r, piece in enumerate(all_pieces):
        i_start, i_end = slice_space(nx, r, size)
        ni_local = i_end - i_start
        target[:, :, i_start:i_end] = piece.reshape((actual_nz, ny, ni_local))

    return global_alpha


# center and scale data. immediately overwrite/delete the raw version
U_rank_transform, centerU, alphaU = shiftscale(U_rank, n_year_train, comm, center_type=center_opt, n_days=n_days, scale_type=scale, save_file=preproc_dir + f'centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
V_rank, centerV, alphaV = shiftscale(V_rank, n_year_train, comm, center_type=center_opt, n_days=n_days, scale_type=scale, save_file=preproc_dir + f'centerV_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
T_rank, centerT, alphaT = shiftscale(T_rank, n_year_train, comm, center_type=center_opt, n_days=n_days, scale_type=scale, save_file=preproc_dir + f'centerT_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
S_rank, centerS, alphaS = shiftscale(S_rank, n_year_train, comm, center_type=center_opt, n_days=n_days, scale_type=scale, save_file=preproc_dir + f'centerS_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, n_year_train, comm, center_type=center_opt, n_days=n_days, scale_type=scale, save_file=preproc_dir + f'centerEta_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )

## save scaling
alphas = {
    "U": gather_global_alpha(alphaU, nz, nx, ny, comm),
    "V": gather_global_alpha(alphaV, nz, nx, ny, comm),
    "T": gather_global_alpha(alphaT, nz, nx, ny, comm),
    "S": gather_global_alpha(alphaS, nz, nx, ny, comm),
    "Eta": gather_global_alpha(alphaEta, 1, nx, ny, comm),
}
if rank == 0:
    np.save(preproc_dir + f"alpha_{center_opt}_{scale}_{n_year_train}yrs.npy", alphas)
log_timing("Finished shiftscale", shiftscale_start)



## save surface fields for the imposed-B/C forcing projection.
## These must be CENTERED ONLY (not scaled): imposing B in the scaled ODE
## multiplies it by alpha, which exactly cancels the 1/alpha baked into the
## scaled basis -> alpha * Tr^T Q_scaled^T B = Tr^T Q_centered^T B. shiftscale
## returned centered-AND-scaled fields, so undo the per-point scaling here.
## (Multiplying the scaled field by alpha recovers Q - center for any center_opt,
## including time-varying seasonal/monthly centers.)
pts_per_surf = ny * (nx // size)

def _centered_surface(field_scaled, alpha_total):
    # alpha_total is a per-point column vector (shape (nspace,1)) for scale='var',
    # or a scalar (1) when scale is None; slice to the surface points either way.
    alpha_surf = alpha_total[:pts_per_surf] if np.ndim(alpha_total) else alpha_total
    return field_scaled[:pts_per_surf, :] * alpha_surf

SST_rank_surf = _centered_surface(T_rank, alphaT)
SSS_rank_surf = _centered_surface(S_rank, alphaS)
SSU_rank_surf = _centered_surface(U_rank_transform, alphaU)

# gather and save
if rank == 0:
    print("Saving transformed surface fields...", flush=True)
gather_and_save_surface(SST_rank_surf, f'SST_{center_opt}_{scale}_{n_year_train}yrs', preproc_dir, nx, ny, rank, comm)
gather_and_save_surface(SSS_rank_surf, f'SSS_{center_opt}_{scale}_{n_year_train}yrs', preproc_dir, nx, ny, rank, comm)
gather_and_save_surface(SSU_rank_surf, f'SSU_{center_opt}_{scale}_{n_year_train}yrs', preproc_dir, nx, ny, rank, comm)

if rank == 0:
    print("Surface save successful. Start stacking", flush=True)

# Preserve the stacked matrix math while avoiding a full explicit stack.
Q_rank = [U_rank_transform, V_rank, Eta_rank, T_rank, S_rank]

streamfunction_jobs = None
svd_start = time.perf_counter()
if rank ==0:
    log_timing("Starting SVD")
    sval_dir = root_dir + '/svals/' + dir_extension + '/'
    os.makedirs(sval_dir, exist_ok=True)

### SVD

if target_ret_energy is not None:
    svd = SVDDecomposition(Q_rank, comm, target_ret_energy=target_ret_energy, **svd_kwargs)
elif r is not None:
    svd = SVDDecomposition(Q_rank, comm, r=r, **svd_kwargs)
else:
    raise ValueError("Neither 'r' nor 'target_ret_energy' was defined.")
svd.compute()
log_timing("Finished SVD", svd_start)
del Q_rank
gc.collect()

###### parallel matrix multiplication to project ICs
# read in new IC and shiftscale based on training data
# centers = [centerU, centerV, centerEta, centerT, centerS]
# IC_rank = get_new_IC(IC_dir, centers, alphas, n_year_train, n_days, rank, size, nx=248)


# ## IC_ = Tr^T @ Q^T @ IC
# IC_rank_ = Q_rank.T @ IC_rank
# IC_rank_ = IC_rank_.compute()

# IC_global = np.zeros_like(IC_rank_)
# comm.Allreduce(IC_rank_, IC_global, op=MPI.SUM)
# q0_ = svd.Tr_global.T @ IC_global

## if using same IC for all:
q0_ = None
if rank == 0:
    q0_ = svd.Qhat_global[:, 0]



results_dir = None
# Create the results directory
if rank == 0:
    run_suffix = f'_{run_label}' if run_label else ''
    results_dir = root_dir + 'save_roms/' + dir_extension + '/' +f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}{run_suffix}/'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created: {results_dir}", flush=True)
    
    # Get Slurm Array ID and Task ID, default to PID if not in a Slurm environment
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid())
    pointer_dir = os.environ.get("ROM_POINTER_DIR", os.getcwd())
    
    temp_file = os.path.join(pointer_dir, f".results_path_{job_id}_{task_id}")
    
    with open(temp_file, "w") as f:
        f.write(results_dir)
    
    # Print the identifier so the bash script can grep it easily
    print(f"PATH_POINTER_FILE:{temp_file}")


results_dir = comm.bcast(results_dir, root=0)



rom_learning_payload = None
if rank == 0:

  #  W_ = svd.Tr_global.T @ W_
   # np.save(root_dir + f'save_roms/varweights_{center_opt}_r{svd.r}_{n_days}days_{n_year_train}yrs.npy', W_)

    print(f'reduced dimension r = {svd.r}')
    print(f'retained energy = {svd.ret_energy[svd.r - 1]}')
  #  svd.save(root_dir , prefix=f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}') ## saves Tr and Qhat
    svd.save(results_dir) ## saves Tr and Qhat

    svd.plot_decay(sval_dir + f'svals_{center_opt}_{scale}_{n_year_train}trainyrs_r{svd.r}.png')


    ## compute reduced forcings
    print('compute forcings')
    forcings_start = time.perf_counter()

    F = OceanForcing(
        grid_dir=paths.GRID_DIR,
        nx=nx, ny=ny,
        xo=0, yo=14,
        xc=31, yc=45,
        dx=0.25, dy=0.25
    )

    sstar = F.compute_salinity_forcing() ## SSS relaxation
    Tmax, c_theta, b_theta = F.compute_temp_forcing() ## SST relaxation
    g = F.compute_wind_forcing() ## time-independent spatially-varying component of wind forcing
    tau_train = [
        F.compute_wind_seasonal_amplitude(max_val=float(tau))
        for tau in train_taus
    ] ## wind stress inputs for each training scenario
    tau_test = [
        F.compute_wind_seasonal_amplitude(max_val=float(tau))
        for tau in test_taus
    ] ## wind stress inputs for each test scenario

    forcings = {
        's_star': sstar.ravel(), #* alphaS,
        'c_theta': c_theta, #* alphaT,
        'b_theta': b_theta, #* alphaT,
        'g': g.ravel(), #* alphaU,
    }


    ### transform forcings
    
    SST_transformed_global = np.load(preproc_dir + f'SST_{center_opt}_{scale}_{n_year_train}yrs_transformed_surface.npy')
    SSS_transformed_global = np.load(preproc_dir + f'SSS_{center_opt}_{scale}_{n_year_train}yrs_transformed_surface.npy')
    SSU_transformed_global = np.load(preproc_dir + f'SSU_{center_opt}_{scale}_{n_year_train}yrs_transformed_surface.npy')

    B, C = project_surface_forcings_multiple(SST_transformed_global, SSS_transformed_global, SSU_transformed_global, svd.Tr_global, forcings)
    log_timing('Finished compute forcings', forcings_start)

    ## ROM inputs

    inputs_list = [
        build_rom_inputs(Tmax, tau, n_year_train, n_days)
        for tau in tau_train
    ]

    ## Standardize the ROM inputs when learning the bilinear N(u(x)q) operator.
    ## N's data columns are u_k * q_i, so a single scalar reg_N is only meaningful
    ## if the input channels share a common O(1) scale -- otherwise the large SST
    ## channel (Tmax ~ 27) swamps the small wind channel (tau4 ~ 0.2) and N's wind
    ## sensitivity is regularized to ~0. Centering also removes the mean-input part
    ## of N, which is collinear with A (so A carries the mean dynamics and N carries
    ## the forcing-fluctuation-dependent dynamics). The imposed B/C are
    ## reparameterized EXACTLY, leaving the physical forcing pathway unchanged:
    ##   B u + C == (B diag(sigma)) u~ + (C + B ubar),   u~ = (u - ubar) / sigma
    ## With learn_N=False this is an identity (ubar=0, sigma=1): behavior is
    ## bit-identical to before.
    if learn_N:
        all_train_inputs = np.hstack(inputs_list)          # (m, num_scenarios*nt)
        input_mean = all_train_inputs.mean(axis=1)          # (m,) grand mean per channel
        input_scale = all_train_inputs.std(axis=1)          # (m,) equalizes N-block col energy
        input_scale[input_scale == 0.0] = 1.0               # guard flat channels (e.g. tau0.1)
    else:
        m_in = inputs_list[0].shape[0]
        input_mean = np.zeros(m_in)
        input_scale = np.ones(m_in)

    def standardize_inputs(u):
        return (np.asarray(u, dtype=float) - input_mean[:, None]) / input_scale[:, None]

    C = C + B @ input_mean            # fold mean forcing into the imposed constant
    B = B * input_scale[None, :]      # scale imposed B columns to match standardized inputs
    inputs_list = [standardize_inputs(u) for u in inputs_list]

    np.save(results_dir + 'input_mean.npy', input_mean)
    np.save(results_dir + 'input_scale.npy', input_scale)


    ## slice concatenated Qhat into a list
    num_scenarios = svd.Qhat_global.shape[1] // nt
    states_list = [svd.Qhat_global[:, i*nt : (i+1)*nt] for i in range(num_scenarios)]

    # Full-horizon stability gate (OPT-IN via config 'enforce_full_horizon_stability').
    # When enabled, the selected operator must stay bounded over the entire
    # train+predict window for every scenario we roll out (train + test taus), not just
    # the training window; candidates that blow up are rejected in the Tikhonov sweep.
    # Left OFF for the overdamping study (taus_2_4 etc.), which deliberately explores
    # marginally-stable/vibrant operators and must NOT auto-reject them.
    if bool(config.get('enforce_full_horizon_stability', False)):
        niters_predict = int((n_year_train + n_year_predict) * 360 / n_days)
        stability_inputs = [
            standardize_inputs(build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days))
            for tau in (tau_train + tau_test)
        ]
        stability_bound_factor = float(config.get('stability_bound_factor', 10.0))
        stability_bound = stability_bound_factor * float(np.abs(svd.Qhat_global).max())
    else:
        niters_predict = None
        stability_inputs = None
        stability_bound = np.inf

    # Option 3: short-window scoring. Pick n_windows random windows once (seeded,
    # shared across every candidate so the comparison is fair) and score on their
    # mean error instead of the whole trajectory. Off when n_windows == 0.
    if n_windows > 0:
        window_len = int(round(window_months * 30 / n_days))
        max_start = nt - window_len
        if max_start < 1:
            raise ValueError(f"window_len {window_len} >= nt {nt}; reduce window_months")
        rng = np.random.default_rng(window_seed)
        window_starts = sorted(int(s) for s in rng.integers(0, max_start + 1, size=n_windows))
        print(f"option 3: scoring on {n_windows} windows of {window_len} steps; starts={window_starts}", flush=True)
    else:
        window_len = None
        window_starts = None

    # learn ROM!
    n_train_H = 20
    n_train_A = 21

    weights_H = np.logspace(-2, 6, n_train_H)
    weights_A = np.logspace(-10, 1, n_train_A)
  #  weights_H = np.logspace(-2, 10, n_train_H)
  #  weights_A = np.logspace(-8, 3, n_train_A)

    # cond = np.linalg.cond(svd.Qhat_global)
    # print(f'condition number: {cond}')

    if block_years is None:
        print('learn ROM', flush=True)
    else:
        print(f'learn ROM sequentially with {block_years}-year blocks', flush=True)
    rom_learning_start = time.perf_counter()
    rom_learning_payload = {
        "states_list": states_list,
        "inputs_list": inputs_list,
        "B": B,
        "C": C,
        "weights_A": weights_A,
        "weights_H": weights_H,
        "stability_inputs": stability_inputs,
        "stability_niters": niters_predict,
        "stability_bound": stability_bound,
        "window_starts": window_starts,
        "window_len": window_len,
    }
else:
    rom_learning_start = None

rom_learning_payload = comm.bcast(rom_learning_payload, root=0)
states_list = rom_learning_payload["states_list"]
inputs_list = rom_learning_payload["inputs_list"]
B = rom_learning_payload["B"]
C = rom_learning_payload["C"]
weights_A = rom_learning_payload["weights_A"]
weights_H = rom_learning_payload["weights_H"]
stability_inputs = rom_learning_payload["stability_inputs"]
stability_niters = rom_learning_payload["stability_niters"]
stability_bound = rom_learning_payload["stability_bound"]
window_starts = rom_learning_payload["window_starts"]
window_len = rom_learning_payload["window_len"]

if block_years is None:
    ## regularization parameter search 
    sweep = TikhonovSweep(
        Qs_=states_list,
        inputs=inputs_list,
        B=B,
        C=C,
        r=svd.r,
        nt=nt,
        t=t,
        weights_A=weights_A,
        weights_H=weights_H,
        # norm_weights=abs(W_),
        stability_inputs=stability_inputs,
        stability_niters=stability_niters,
        stability_bound=stability_bound,
        learn_N=learn_N,
        reg_N=reg_N,
        var_weight=var_weight,
        window_starts=window_starts,
        window_len=window_len,
    )


    # grid search for best regularization parameters
    reg_A, reg_H, best_err = sweep.run_mpi(comm)

    # No candidate converged (every rollout blew up / was rejected). Fail fast on
    # ALL ranks -- otherwise rank 0 raises in fit_best_model while the others sit
    # in a collective and the job deadlocks.
    if reg_A is None:
        if rank == 0:
            print("ERROR: Tikhonov sweep found no valid (finite) model for any reg "
                  "combination -- every candidate blew up over the training window. "
                  "Regularize more (raise reg_N, e.g. reg_N='H' or a larger fixed "
                  "value; or raise weights_H).", flush=True)
        comm.Barrier()
        sys.exit(1)
        
    if rank == 0:
        print("Best:", best_err, reg_A, reg_H)

        # build final model with best reg params
        model = sweep.fit_best_model()
        model.model.save(results_dir + 'model.h5', overwrite=True)
        log_timing('Finished ROM learning', rom_learning_start)
else:
    model = fit_sequential_model(
        Qs_=states_list,
        inputs=inputs_list,
        B=B,
        C=C,
        r=svd.r,
        n_year_train=n_year_train,
        n_days=n_days,
        block_years=block_years,
        weights_A=weights_A,
        weights_H=weights_H,
        comm=comm,
        t=t,
    )

    if rank == 0:
        model.model.save(results_dir + 'model.h5', overwrite=True)
        log_timing('Finished sequential ROM learning', rom_learning_start)

if rank == 0:
    # Generate scenario labels from snapshot directory names
    # Extract meaningful names from full paths (e.g., 'run_10yrspinup_tau0.1/' -> 'tau0.1')
    scenario_labels = []
    for snap_dir in snapshot_dirs:
        # Clean up path and extract last component
        clean_path = snap_dir.rstrip('/')
        dir_name = clean_path.split('/')[-1]
        scenario_labels.append(dir_name)
    
    test_labels = [path.rstrip('/').split('/')[-1] for path in test_dirs]

    # predict for all training scenarios
    inputs_list_predict = [
        standardize_inputs(build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days))
        for tau in tau_train
    ]
    
    streamfunction_jobs = []
    for i, (Q_, u_in) in enumerate(zip(states_list, inputs_list_predict)):
        Q0 = Q_[:, 0]

        prediction_start = time.perf_counter() if i == 0 else None
        Q_rec = model.predict(Q0, niters= int((n_year_train + n_year_predict)* 360 / n_days), inputs=u_in)
        if i == 0:
            log_timing('Finished one ROM prediction', prediction_start)
        
        # Save each reconstruction with descriptive name
        scenario_name = scenario_labels[i] if i < len(scenario_labels) else f'scenario_{i}'
        q_rom_stem = f'Q_ROM_train_{scenario_name}'
        np.save(results_dir + f'{q_rom_stem}.npy', Q_rec)
        streamfunction_jobs.append((q_rom_stem, Q_rec))
        print(f"Reconstructed and saved training scenario {i} ({scenario_name})")

    # predict for unseen cases
    for test_label, tau in zip(test_labels, tau_test):
        inputs_new = standardize_inputs(build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days))

        ## using projected q0 
        # OR can use Q0 from training scenario, looking at how well we can drift ##
        Q_ROM_ = model.predict(q0_, niters = int((n_year_train + n_year_predict)* 360 / n_days), inputs=inputs_new)
        q_rom_stem = f'Q_ROM_test_{test_label}'
        np.save(results_dir + f'{q_rom_stem}.npy', Q_ROM_)
        streamfunction_jobs.append((q_rom_stem, Q_ROM_))


comm.Barrier()

# --- compute u_bt for barotropic streamfunction computation ---
#if rank ==0:
 #   print('read grid')

grid = read_grid(comm, paths.GRID_DIR)

if rank ==0:
    streamfunction_total_start = time.perf_counter()
    log_timing('Starting barotropic streamfunction for each scenario')
else:
    streamfunction_total_start = None

streamfunction_labels = comm.bcast(
    [label for label, _ in streamfunction_jobs] if rank == 0 else None,
    root=0,
)

for job_idx, output_label in enumerate(streamfunction_labels):
    streamfunction_scenario_start = time.perf_counter()
    if rank == 0:
        log_timing(f'Starting barotropic streamfunction: {output_label}')
    Q_ROM_val = streamfunction_jobs[job_idx][1] if rank == 0 else None

    compute_barotropic_streamfunction(
        comm=comm,
        grid=grid,
        U_rank=U_rank,
        centerU=centerU,
        svd=svd,
        Q_ROM_val=Q_ROM_val,
        nx=nx,
        ny=ny,
        nz=nz,
        root_dir=results_dir,
        center_opt=center_opt,
        scale_type=scale,
        n_year_train=n_year_train,
        output_label=output_label,
    )
    log_timing(f'Finished barotropic streamfunction: {output_label}', streamfunction_scenario_start)

log_timing('Finished all barotropic streamfunctions', streamfunction_total_start)
log_timing('mixed_rom_forcings.py finished', script_start)
