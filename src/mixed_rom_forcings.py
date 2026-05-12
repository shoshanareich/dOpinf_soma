from mpi4py import MPI
import sys
import os
import time
from datetime import datetime

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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["UCX_MEMTYPE_CACHE"] = "n"

log_timing(f"mixed_rom_forcings.py start; ranks initialized: {size}")
log_timing("Starting library imports")

import numpy as np
import xarray as xr
import netCDF4 as nc
import xmitgcm 
import json
import sys
sys.path.append('/home/shoshi/MITgcm_obsfit/MITgcm/utils/python/MITgcmutils/')
from MITgcmutils import rdmds

from config_utils import resolve_forcing_config
from utils import *
from shiftscale import *
from svd import SVDDecomposition
import gc # garbage cleanup

if MPI.COMM_WORLD.Get_rank() == 0:
    import opinf
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    from plotting import ROMPlotter
    from forcing import *
    from learn_rom import TikhonovSweep
    log_timing("Library imports finished; setting vars", script_start)


#root_dir = '/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/' 
root_dir = '/scratch/shoshi/soma4/dOpInf_results/'

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

for snapshot_dir in snapshot_dirs:
    ds = xr.open_dataset(
        snapshot_dir + 'states_20yrs.nc',
        engine="h5netcdf",
        decode_timedelta=False,
        decode_times=False,
        chunks={"time": min(360, 360 * n_year_train), "i": 8, "i_g": 8},
    )  # , backend_kwargs={'driver': 'mpio'})

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
U_rank_transform, centerU, alphaU = shiftscale(U_rank, n_year_train, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
V_rank, centerV, alphaV = shiftscale(V_rank, n_year_train, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerV_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
T_rank, centerT, alphaT = shiftscale(T_rank, n_year_train, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerT_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
S_rank, centerS, alphaS = shiftscale(S_rank, n_year_train, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerS_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, n_year_train, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerEta_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )

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



## save transformed surface fields for forcing projection:

pts_per_surf = ny * (nx // size) 
SST_rank_surf = T_rank[:pts_per_surf, :] 
SSS_rank_surf = S_rank[:pts_per_surf, :] 
SSU_rank_surf = U_rank_transform[:pts_per_surf, :] 

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
    svd = SVDDecomposition(Q_rank, comm, target_ret_energy=target_ret_energy)
elif r is not None:
    svd = SVDDecomposition(Q_rank, comm, r=r)
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
    results_dir = root_dir + 'save_roms/' + dir_extension + '/' +f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}/'
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



if rank ==0:

  #  W_ = svd.Tr_global.T @ W_
   # np.save(root_dir + f'save_roms/varweights_{center_opt}_r{svd.r}_{n_days}days_{n_year_train}yrs.npy', W_)

    print(f'reduced dimension r = {svd.r}')
    print(f'retained energy = {svd.ret_energy[svd.r]}')
  #  svd.save(root_dir , prefix=f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}') ## saves Tr and Qhat
    svd.save(results_dir) ## saves Tr and Qhat

    svd.plot_decay(sval_dir + f'svals_{center_opt}_{scale}_{n_year_train}trainyrs_r{svd.r}.png')


    ## compute reduced forcings
    print('compute forcings')
    forcings_start = time.perf_counter()

    F = OceanForcing(
        grid_dir='/scratch/shoshi/soma4/grid/',
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
                                

    ## slice concatenated Qhat into a list 
    num_scenarios = svd.Qhat_global.shape[1] // nt
    states_list = [svd.Qhat_global[:, i*nt : (i+1)*nt] for i in range(num_scenarios)]


    print('learn ROM', flush=True)
    rom_learning_start = time.perf_counter()

    # learn ROM!
    n_train_H = 20
    n_train_A = 21

    weights_H = np.logspace(-2, 6, n_train_H)
    weights_A = np.logspace(-10, 1, n_train_A)
  #  weights_H = np.logspace(-2, 10, n_train_H)
  #  weights_A = np.logspace(-8, 3, n_train_A)

    # cond = np.linalg.cond(svd.Qhat_global)
    # print(f'condition number: {cond}')

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
        weights_H=weights_H#,
      #  norm_weights=abs(W_)
    )


    # grid search for best regularization parameters
    reg_A, reg_H, best_err = sweep.run()
    print("Best:", best_err, reg_A, reg_H)

    # build final model with best reg params
    model = sweep.fit_best_model()
    model.model.save(results_dir + 'model.h5', overwrite=True)
    log_timing('Finished ROM learning', rom_learning_start)

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
        build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days)
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
        inputs_new = build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days)

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

grid = read_grid(comm, '/scratch/shoshi/soma4/grid/')

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
