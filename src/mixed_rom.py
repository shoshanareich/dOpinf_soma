from mpi4py import MPI
import sys
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["UCX_MEMTYPE_CACHE"] = "n"

if rank == 0:
    print(f"Ranks initialized: {size}. Starting library imports", flush=True)

import numpy as np
import xarray as xr
import netCDF4 as nc
import xmitgcm 
import json
import sys
sys.path.append('/home/shoshi/MITgcm_obsfit/MITgcm/utils/python/MITgcmutils/')
from MITgcmutils import rdmds

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
    print("All modules loaded. Setting vars... ", flush=True)


#root_dir = '/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/' 
root_dir = '/scratch/shoshi/soma4/dOpInf_results/'
snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'

## spatial vars
nx = 248
ny = nx
nz = 31

N = nx*ny*nz



#### training config

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

# check for target_ret_energy or r
target_ret_energy = config.get('target_ret_energy')
r = config.get('r')

if (target_ret_energy is None and r is None) or (target_ret_energy is not None and r is not None):
    raise ValueError("Must define either 'target_ret_energy' OR 'r' in config.json")

n_days_per_year = int(360 / n_days)
nt = n_days_per_year * n_year_train 
t = np.linspace(0, n_year_train * 360 * 24 * 3600, nt)

if rank == 0:
    print('Parameters loaded from config.json. Opening snapshots', flush=True)






## split data and select training snapshots

ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="h5netcdf",  
                    decode_timedelta=False, decode_times=False, chunks={})#, backend_kwargs={'driver': 'mpio'})

U_rank, V_rank, Eta_rank, T_rank, S_rank = reshape_data(nx, n_days, n_year_train, ds, rank, size)


# # every rank opens the file, but the 'netCDF4' engine is often faster at "opening" than the xarray wrapper.
# # small stagger avoids initial collision.
# import time
# time.sleep(rank * 0.02) 

# f = nc.Dataset(snapshot_dir + 'states_20yrs.nc', mode='r')
# U_rank, V_rank, Eta_rank, T_rank, S_rank = get_Q_rank(f, nx, n_year_train, n_days, rank, size)

if rank == 0:
    print("Training snapshots loaded. Compute variance for weights ", flush=True)

# Wu_rank = 1 / scalefactor(U_rank, 'var', comm)
# Wv_rank = 1 / scalefactor(V_rank, 'var', comm)
# Weta_rank = 1 / scalefactor(Eta_rank, 'var', comm)
# WT_rank = 1 / scalefactor(T_rank, 'var', comm)
# WS_rank = 1 / scalefactor(S_rank, 'var', comm)

# W_rank = np.concatenate([Wu_rank, Wv_rank, Weta_rank, WT_rank, WS_rank])


if rank == 0:
    print("Training snapshots loaded. Starting shiftscale... ", flush=True)

# center and scale data. immediately overwrite/delete the raw version
U_rank_transform, centerU, alphaU = shiftscale(U_rank, comm, center_type=center_opt, scale_type=scale, save_file=root_dir + f'preproc/centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc')
V_rank, centerV, alphaV = shiftscale(V_rank, comm, center_type=center_opt, scale_type=scale, save_file=root_dir + f'preproc/centerV_{center_opt}_{n_days}days_{n_year_train}yrs.nc')
T_rank, centerT, alphaT = shiftscale(T_rank, comm, center_type=center_opt, scale_type=scale, save_file=root_dir + f'preproc/centerT_{center_opt}_{n_days}days_{n_year_train}yrs.nc')
S_rank, centerS, alphaS = shiftscale(S_rank, comm, center_type=center_opt, scale_type=scale, save_file=root_dir + f'preproc/centerS_{center_opt}_{n_days}days_{n_year_train}yrs.nc')
Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, comm, center_type=center_opt, scale_type=scale, save_file=root_dir + f'preproc/centerEta_{center_opt}_{n_days}days_{n_year_train}yrs.nc')



# # Shift u, v independently 
# U_shifted, centerU, _ = shiftscale(U_rank, comm, center_type=center_opt, scale_type=None, save_file=root_dir + f'preproc/centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc')
# V_shifted, centerV, _ = shiftscale(V_rank, comm, center_type=center_opt, scale_type=None, save_file=root_dir + f'preproc/centerV_{center_opt}_{n_days}days_{n_year_train}yrs.nc')

# alphaU = scalefactor(U_shifted, scale_type = 'maxabs',comm=comm)
# alphaV = scalefactor(V_shifted, scale_type = 'maxabs',comm=comm)

# alphaUV = np.nanmax([alphaU, alphaV])

# # Compute a joint scale factor for u, v
# # Energy = u^2 + v^2
# local_sq_sum = ((U_shifted**2).sum(axis=0) + (V_shifted**2).sum(axis=0)).compute()

# global_sq_sum = np.zeros_like(local_sq_sum)
# comm.Allreduce(local_sq_sum, global_sq_sum, op=MPI.SUM)
# alphaUV = np.sqrt(global_sq_sum).max()

# # Apply the shared scale
# U_rank = U_shifted / alphaUV
# V_rank = V_shifted / alphaUV

if rank == 0:
    alphas = {
        "U": alphaU,
        "V": alphaV,
        "T": alphaT,
        "S": alphaS,
        "Eta": alphaEta,
    }
    np.save(root_dir + f"preproc/alpha_{center_opt}_{scale}_{n_year_train}yrs.npy", alphas)



if rank ==0:
    print('start stacking')#

Q_rank = np.vstack([U_rank_transform, V_rank, Eta_rank, T_rank, S_rank])

## clean up 
del U_rank_transform, V_rank, Eta_rank, T_rank, S_rank
gc.collect()

if rank ==0:
    print('snapshot matrix compiled. starting SVD...')

### SVD

if 'target_ret_energy' in locals():
    svd = SVDDecomposition(Q_rank, comm, target_ret_energy=target_ret_energy)
elif 'r' in locals():
    svd = SVDDecomposition(Q_rank, comm, r=r)
else:
    raise ValueError("Neither 'r' nor 'target_ret_energy' was defined.")
svd.compute()

sval_dir = root_dir + '/svals/'

### project weights (distributed mat-vec multiplication)
# W_ = Tr.T @ Q.T @ W
#    = Tr.T @ sum(Q_rank.T @ W_rank)


# W_rank_ = Q_rank.T.dot(W_rank).compute()
# W_ = np.zeros(int(n_year_train * n_days_per_year))
# comm.Allreduce(W_rank_, W_, op=MPI.SUM)

results_dir = None
# Create the results directory
if rank == 0:
    results_dir = root_dir + 'save_roms/' +f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}/'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created: {results_dir}", flush=True)
    
    # Get Slurm Array ID and Task ID, default to PID if not in a Slurm environment
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid())
    
    temp_file = f".results_path_{job_id}_{task_id}"
    
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
    tau4 = F.compute_wind_seasonal_amplitude() ## wind stress inputs

    forcings = {
        's_star': sstar.ravel(), #* alphaS,
        'c_theta': c_theta, #* alphaT,
        'b_theta': b_theta, #* alphaT,
        'g': g.ravel(), #* alphaU,
    }

    ## ROM inputs

    inputs = build_rom_inputs(Tmax, tau4, n_year_train, n_days)

    ### transform forcings

    ds_surface = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="h5netcdf",  
                        decode_timedelta=False, chunks={}).isel(k=0, time=slice(0, 360*n_year_train, n_days))

    # Load global centers 
    centers = {
        'U': xr.open_dataset(root_dir + f'preproc/centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc').center.values,
        'T': xr.open_dataset(root_dir + f'preproc/centerT_{center_opt}_{n_days}days_{n_year_train}yrs.nc').center.values,
        'S': xr.open_dataset(root_dir + f'preproc/centerS_{center_opt}_{n_days}days_{n_year_train}yrs.nc').center.values
    }
    alphas  = {'T': alphaT,  'S': alphaS,  'U': alphaU}
    B, C = project_surface_forcings(ds_surface, svd.Tr_global,centers,alphas,forcings,nx, ny, nz, [center_opt, center_opt], n_year_train)

    # np.save(root_dir + f'for_nicole/projected_input_operator_r{svd.r}_{center_opt}_{n_year_train}trainyrs.npy', B)
    # np.save(root_dir + f'for_nicole/projected_constant_operator_r{svd.r}_{center_opt}_{n_year_train}trainyrs.npy', C)
    # np.save(root_dir + f'for_nicole/inputs_r{svd.r}_{center_opt}_{n_year_train}trainyrs.npy', inputs)
 
 #   SST_transformed, _, _ = shiftscale(ds_surface.T.stack(space=('j', 'i')).T, center_type=center_opt)
 #   SSS_transformed, _, _ = shiftscale(ds_surface.S.stack(space=('j', 'i')).T, center_type=center_opt)
 #   SSU_transformed, _, _ = shiftscale(ds_surface.U.stack(space=('j', 'i_g')).T, center_type=center_opt)

    print('learn ROM')

    # learn ROM!
    n_train_H = 20
    n_train_A = 21

    weights_H = np.logspace(-2, 6, n_train_H)
    weights_A = np.logspace(-10, 1, n_train_A)
  #  weights_H = np.logspace(-2, 10, n_train_H)
  #  weights_A = np.logspace(-8, 3, n_train_A)
    # #weights_A = np.logspace(-4, 1, n_train_A)

    # weights_A = np.array([1e-16])
    # weights_H = np.array([1e-16])

    cond = np.linalg.cond(svd.Qhat_global)
    print(f'condition number: {cond}')

    ## regularization parameter search 
    sweep = TikhonovSweep(
        Q_=svd.Qhat_global,
        inputs=inputs,
        B=B,
        C=C,
        r=svd.r,
        nt=nt,
        t=t,
        weights_A=weights_A,
        weights_H=weights_H#,
      #  norm_weights=abs(W_)
    )

    reg_A, reg_H, best_err = sweep.run()
    print("Best:", best_err, reg_A, reg_H)

    # ### readding in best to do prediction:
    # sweep = TikhonovSweep.load(
    # "/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/roms/"
    # f"sweep_r{r}_{center_opt}_{n_year_train}yrs.pkl",
    # Q_=svd.Qhat_global,
    # inputs=inputs,
    # B=B,
    # C=C,
    # nt=nt,
    # t=t,
    # )

    # save best so easily can predict again
   # save_dir = root_dir + 'save_roms/'
 #   sweep.save(save_dir + f'Qrom_sweep_r{r}_{center_opt}_{n_year_train}trainingyrs.pkl')

    inputs = build_rom_inputs(Tmax, tau4, n_year_train + n_year_predict, n_days)

    model, Q_ROM_ = sweep.fit_best_model(
        inputs_fit=inputs,
        niters=(n_year_train + n_year_predict) * int(360 / n_days) ,
    )

    # save rom for all predict years
    np.save(results_dir + f'Q_ROM_.npy', Q_ROM_)
    model.save(results_dir + 'model.h5', overwrite=True)


comm.Barrier()

# --- compute u_bt for barotropic streamfunction computation ---
#if rank ==0:
 #   print('read grid')

grid = read_grid(comm, '/scratch/shoshi/soma4/grid/')

if rank ==0:
    print('compute barotropic streamfunction')

Q_ROM_val = Q_ROM_ if rank == 0 else None

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
    scale_type = scale,
    n_year_train=n_year_train
)




    # read in FOM for comparison
#    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', decode_timedelta=False, chunks={}) 
 #   Eta = ds['Eta'].isel(time=slice(0, 360*(n_year_train + n_year_predict)+1, n_days))

#     Eta_train = Eta.isel(time=slice(0, int(360/n_days)*n_year_train+1))

#     #S_Eta = Eta_train.stack(space=('j', 'i')).T
#     S_Eta_train = Eta_train.stack(space=('j', 'i')).T
#     S_Eta = Eta.stack(space=('j', 'i')).T

#     # alpha = abs(S_Eta).max().values

#     # if center_opt == 'mean':
#     #     center = S_Eta.mean(dim='time').values
#     # elif center_opt == 'IC':
#     #     center = S_Eta.isel(time=0).values
#     # elif center_opt == 'seasonal':
#     #     center = S_Eta.rolling(time=30, center=True, min_periods=1).mean()
#     # elif center_opt == 'monthly':
#     #     # build synthetic month index: 0..11
#     #     month = ((S_Eta['time'] % (360*24*3600)) // (30*24*3600)).astype(int)
#     #     # attach month as a coordinate
#     #     Etam = S_Eta.assign_coords(month=('time', month.data))
#     #     # monthly climatology (mean across all years)
#     #     monthly_mean = Etam.groupby('month').mean(dim='time')
#     #     # broadcast back to full time axis
#     #     center = monthly_mean.sel(month=Etam['month'])
#     # else:
#     #     center = np.zeros_like(S_Eta.isel(time=0).values)

#     ## compared processed FOM and ROM
#    # S_Eta, _, _ = shiftscale(S_Eta, center_type=center_opt)

#     center_train = S_Eta_train.rolling(time=30, center=True, min_periods=1).mean()

#     doy_train = (center_train['time'] % (360*24*3600)) // (24*3600) + 1
#     Ctrain = center_train.assign_coords(dayofyear=('time', doy_train.data))

#     center_clim = Ctrain.groupby('dayofyear').mean('time')
#     doy_full = (S_Eta['time'] % (360*24*3600)) // (24*3600) + 1
#     center_full = center_clim.sel(dayofyear=doy_full).drop_vars('dayofyear')


#     ## get centering and scaling of training data to transform back
#    # _, center, alphaEta = shiftscale(S_Eta_train, comm=None, center_type=center_opt)



# # project back to high-dimensional space and de-transform data (centering and scaling)
# Eta_transformed = gather_distributed_array(Eta_rank_transformed.values, comm, root=0)
# centerEta_global = gather_distributed_array(centerEta.values, comm, root=0)

# if rank == 0:
#     print(f"Reconstruction complete. Global shape: {Eta_transformed.shape}")

#     nt_total = (n_year_train + n_year_predict)*int(360/n_days)+1

#     Vr = np.matmul(Eta_transformed, svd.Tr_global)
#     Eta_ROM = Vr @ Q_ROM_ * alphaEta + centerEta_global

#     Eta_ROM_3d = Eta_ROM.values.T.reshape(Eta_ROM.shape[1], nx, ny) 

#     abs_err, norm_err = opinf.post.lp_error(S_Eta, Eta_ROM, normalize=True)
#     ymax= np.nanmax(norm_err)

    
    

#     # plots!
#     Eta_ROM_3d_predict = Eta_ROM_3d[n_year_train*int(360/n_days)+1:]
#     Eta_ROM_monthlymean = Eta_ROM_3d_predict.reshape(n_year_predict*12, int(30/n_days), nx, nx).mean(axis=1)  # monthly mean


#     Eta_FOM_monthlymean = (Eta[n_year_train*int(360/n_days)+1:].values).reshape(n_year_predict*12, int(30/n_days), nx, nx).mean(axis=1)
    

#     save_dir = '/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/plots/'

#     plotter = ROMPlotter(Q_fom=Eta, Q_rom=Eta_ROM_3d, r=svd.r,
#                      n_year_train=n_year_train, n_year_predict=n_year_predict,
#                      n_days=n_days, n_days_per_year=n_days_per_year)

#     plotter.plot_rel_l2_error(save_path=save_dir+f"rel_l2_error_r{svd.r}_{center_opt}_{n_year_train}yrs.png")
#     plotter.plot_timeseries_locations(save_path=save_dir+f"ssh_timeseries_r{svd.r}_{center_opt}_{n_year_train}yrs.png")
#     plotter.plot_monthly_avg(Eta_ROM_monthlymean, Eta_FOM_monthlymean, month_idx=12, save_path=save_dir+f"monthly_avg_r{svd.r}_{center_opt}_{n_year_train}yrs.png")

#     vid = plotter.animate_monthly(Eta_ROM_monthlymean, Eta_FOM_monthlymean, timesteps=range(Eta_FOM_monthlymean.shape[0]))
#     HTML(vid.to_jshtml())
#     vid.save(save_dir +f'monthly_avg_r{svd.r}_{center_opt}_{n_year_train}yrs.gif')
