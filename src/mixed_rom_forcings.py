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
snapshot_dirs = ['/scratch/shoshi/soma4/run_10yrspinup_tau0.1/',
                    '/scratch/shoshi/soma4/run_10yrspinup_tau0.5/']

#IC_dir = '/scratch/shoshi/soma4/run_20yrs_tau0.3/'
test_dir = '/scratch/shoshi/soma4/run_10yrspinup_tau0.3/'

## spatial vars
nx = 248
ny = nx
nz = 31

N = nx*ny*nz



#### training config

# # Read in config file on rank 0 only 
# if rank == 0:
#     config_path = os.environ.get("ROM_CONFIG_PATH", os.path.join(root_dir, "config.json"))
#     with open(config_path, 'r') as f:
#         config = json.load(f)
# else:
#     config = None

# # Broadcast config to all ranks
# config = comm.bcast(config, root=0)

# # Extract training options
# n_year_train = config['n_year_train']
# n_year_predict = config['n_year_predict']
# n_days = config['n_days']
# center_opt = config['center_opt']
# scale = config['scale']

# # check for target_ret_energy or r
# target_ret_energy = config.get('target_ret_energy')
# r = config.get('r')

n_year_train = 5
n_year_predict = 0
n_days = 1
center_opt = 'mean'
scale = 'maxabs'

# check for target_ret_energy or r
#target_ret_energy = 0.85
target_ret_energy = None
r = 20

if (target_ret_energy is None and r is None) or (target_ret_energy is not None and r is not None):
    raise ValueError("Must define either 'target_ret_energy' OR 'r' in config.json")

n_days_per_year = int(360 / n_days)
nt = n_days_per_year * n_year_train 
t = np.linspace(0, n_year_train * 360 * 24 * 3600, nt)

if rank == 0:
    print('Parameters loaded from config.json. Opening snapshots', flush=True)



preproc_dir = None
# Create the results directory
if rank == 0:
    preproc_dir = root_dir + 'save_roms/taus_1_5_sameIC/'
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
    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="h5netcdf",  
                        decode_timedelta=False, decode_times=False, chunks={})#, backend_kwargs={'driver': 'mpio'})

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


if rank == 0:
    print("Training snapshots loaded. Starting shiftscale... ", flush=True)

# center and scale data. immediately overwrite/delete the raw version
U_rank_transform, centerU, alphaU = shiftscale(U_rank, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerU_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
V_rank, centerV, alphaV = shiftscale(V_rank, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerV_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
T_rank, centerT, alphaT = shiftscale(T_rank, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerT_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
S_rank, centerS, alphaS = shiftscale(S_rank, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerS_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )
Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, comm, center_type=center_opt, scale_type=scale, save_file=preproc_dir + f'centerEta_{center_opt}_{n_days}days_{n_year_train}yrs.nc' )


## save scaling
alphas = {
        "U": alphaU,
        "V": alphaV,
        "T": alphaT,
        "S": alphaS,
        "Eta": alphaEta,
    }
if rank == 0:
    np.save(preproc_dir + f"alpha_{center_opt}_{scale}_{n_year_train}yrs.npy", alphas)



## save transformed surface fields for forcing projection:

pts_per_surf = ny * (nx // size) 
SST_rank_surf = T_rank[:pts_per_surf, :] 
SSS_rank_surf = S_rank[:pts_per_surf, :] 
SSU_rank_surf = U_rank_transform[:pts_per_surf, :] 

# gather and save
gather_and_save_surface(SST_rank_surf, 'SST', preproc_dir, nx, ny, rank, comm)
gather_and_save_surface(SSS_rank_surf, 'SSS', preproc_dir, nx, ny, rank, comm)
gather_and_save_surface(SSU_rank_surf, 'SSU', preproc_dir, nx, ny, rank, comm)

if rank == 0:
    print("Surface save successful. Start stacking")


Q_rank = np.vstack([U_rank_transform, V_rank, Eta_rank, T_rank, S_rank])

## clean up 
del U_rank_transform, V_rank, Eta_rank, T_rank, S_rank
gc.collect()

if rank ==0:
    print('snapshot matrix compiled. starting SVD...')
    sval_dir = root_dir + '/svals/taus_1_5_sameIC/'
    os.makedirs(sval_dir, exist_ok=True)

### SVD

if 'target_ret_energy' in locals():
    svd = SVDDecomposition(Q_rank, comm, target_ret_energy=target_ret_energy)
elif 'r' in locals():
    svd = SVDDecomposition(Q_rank, comm, r=45)
else:
    raise ValueError("Neither 'r' nor 'target_ret_energy' was defined.")
svd.compute()


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
q0_ = svd.Qhat_global[:,0]



results_dir = None
# Create the results directory
if rank == 0:
    results_dir = root_dir + 'save_roms/taus_1_5_sameIC/' +f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{svd.r}/'
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
    tau1 = F.compute_wind_seasonal_amplitude(max_val=0.1) ## wind stress inputs

    forcings = {
        's_star': sstar.ravel(), #* alphaS,
        'c_theta': c_theta, #* alphaT,
        'b_theta': b_theta, #* alphaT,
        'g': g.ravel(), #* alphaU,
    }


    ### transform forcings
    
    SST_transformed_global = np.load(preproc_dir + 'SST_transformed_surface.npy')
    SSS_transformed_global = np.load(preproc_dir + 'SSS_transformed_surface.npy')
    SSU_transformed_global = np.load(preproc_dir + 'SSU_transformed_surface.npy')

    B, C = project_surface_forcings_multiple(SST_transformed_global, SSS_transformed_global, SSU_transformed_global, svd.Tr_global, forcings)

    ## ROM inputs

    inputs_list = [build_rom_inputs(Tmax, tau1, n_year_train, n_days),
                    build_rom_inputs(Tmax, tau4, n_year_train, n_days)]
    
    inputs_stacked = np.hstack([build_rom_inputs(Tmax, tau1, n_year_train, n_days),
                                build_rom_inputs(Tmax, tau4, n_year_train, n_days)])

    ## slice concatenated Qhat into a list 
    num_scenarios = svd.Qhat_global.shape[1] // nt
    states_list = [svd.Qhat_global[:, i*nt : (i+1)*nt] for i in range(num_scenarios)]


    print('learn ROM')

    # learn ROM!
    n_train_H = 20
    n_train_A = 21

    weights_H = np.logspace(-2, 6, n_train_H)
    weights_A = np.logspace(-10, 1, n_train_A)
  #  weights_H = np.logspace(-2, 10, n_train_H)
  #  weights_A = np.logspace(-8, 3, n_train_A)

    cond = np.linalg.cond(svd.Qhat_global)
    print(f'condition number: {cond}')

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

    # predict for all training scenarios
    inputs_list_predict = [build_rom_inputs(Tmax, tau1, n_year_train + n_year_predict, n_days),
                    build_rom_inputs(Tmax, tau4, n_year_train + n_year_predict, n_days)]
    
    for i, (Q_, u_in) in enumerate(zip(states_list, inputs_list_predict)):
        Q0 = Q_[:, 0]
        
        Q_rec = model.predict(Q0, niters= int((n_year_train + n_year_predict)* 360 / n_days), inputs=u_in)
        
        # Save each reconstruction
        np.save(results_dir + f'Q_ROM_train_scenario_{i}.npy', Q_rec)
        print(f"Reconstructed and saved training scenario {i}")

    # predict for unseen case
    tau = F.compute_wind_seasonal_amplitude(max_val=0.3)
    inputs_new = build_rom_inputs(Tmax, tau, n_year_train + n_year_predict, n_days)


    ## using projected q0 
    # OR can use Q0 from training scenario, looking at how well we can drift ##
    Q_ROM_ = model.predict(q0_[:,0], niters = int((n_year_train + n_year_predict)* 360 / n_days), inputs=inputs_new)
    np.save(results_dir + f'Q_ROM_test3.npy', Q_ROM_)


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




#region OLD PLOT CODE


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


#endregion