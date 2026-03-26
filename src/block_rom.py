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
   # from learn_rom import TikhonovSweep
    from learn_blockrom import *
    print("All modules loaded. Setting vars... ", flush=True)


root_dir = '/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/'
snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'

## spatial vars
nx = 248
ny = nx
nz = 31

N = nx*ny*nz

## training options
n_year_train = 2
n_year_predict = 5
n_days = 1
n_days_per_year = int(360/n_days)

center_opt = 'mean' ## how to center data
#center_opt2 = 'global_mean'
target_ret_energy = 0.8 # define target retained energy for the dOpInf ROM
#r = 20

# number of training snapshots
nt = int(360/n_days)*n_year_train 
t = np.linspace(0, n_year_train*360*24*3600, n_year_train*n_days_per_year)

if rank ==0:
    print('parameters set. Opening snapshots')


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
    print("Training snapshots loaded. Starting shiftscale... ", flush=True)

# center and scale data. immediately overwrite/delete the raw version
U_rank_transform, centerU, alphaU = shiftscale(U_rank, comm, center_type=center_opt, save_file=root_dir + f'save_for_later/centerU_{center_opt}_{n_year_train}yrs.nc')
V_rank, centerV, alphaV = shiftscale(V_rank, comm, center_type=center_opt, save_file=root_dir + f'save_for_later/centerV_{center_opt}_{n_year_train}yrs.nc')
T_rank, centerT, alphaT = shiftscale(T_rank, comm, center_type=center_opt, save_file=root_dir + f'save_for_later/centerT_{center_opt}_{n_year_train}yrs.nc')
S_rank, centerS, alphaS = shiftscale(S_rank, comm, center_type=center_opt, save_file=root_dir + f'save_for_later/centerS_{center_opt}_{n_year_train}yrs.nc')
Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, comm, center_type=center_opt, save_file=root_dir + f'save_for_later/centerEta_{center_opt}_{n_year_train}yrs.nc')

# U_rank, centerU, alphaU = shiftscale(U_rank, comm, center_type=center_opt)
# V_rank, centerV, alphaV = shiftscale(V_rank, comm, center_type=center_opt)
# T_rank, centerT, alphaT = shiftscale(T_rank, comm, center_type=center_opt)
# S_rank, centerS, alphaS = shiftscale(S_rank, comm, center_type=center_opt)
# Eta_rank, centerEta, alphaEta = shiftscale(Eta_rank, comm, center_type=center_opt)


if rank ==0:
    print('start stacking')

Q1_rank = np.vstack([U_rank_transform, V_rank])
Q2_rank = np.vstack([Eta_rank])
Q3_rank = np.vstack([T_rank, S_rank])

## clean up 
del U_rank_transform, V_rank, Eta_rank, T_rank, S_rank
gc.collect()

if rank ==0:
    print('snapshot matrices compiled. starting SVD...')

### SVD

if 'target_ret_energy' in locals():
    svd1 = SVDDecomposition(Q1_rank, comm, target_ret_energy=target_ret_energy)
    svd2 = SVDDecomposition(Q2_rank, comm, target_ret_energy=target_ret_energy)
    svd3 = SVDDecomposition(Q3_rank, comm, target_ret_energy=target_ret_energy)
elif 'r' in locals():
    svd1 = SVDDecomposition(Q1_rank, comm, r=r)
    svd2 = SVDDecomposition(Q2_rank, comm, r=r)
    svd3 = SVDDecomposition(Q3_rank, comm, r=r)
else:
    raise ValueError("Neither 'r' nor 'target_ret_energy' was defined.")
svd1.compute()
svd2.compute()
svd3.compute()

sval_dir = root_dir + '/svals/'

if rank ==0:
    print(f'reduced dimension r1 = {svd1.r}')
    print(f'retained energy 1 = {svd1.ret_energy[svd1.r]}')
    svd1.save(root_dir + 'save_for_later/', prefix=f'UV_{center_opt}_r{svd1.r}_{n_year_train}trainingyrs') ## saves Tr and Qhat
    svd1.plot_decay(sval_dir + f'UVsvals_{center_opt}_{n_year_train}trainyrs_r{svd1.r}.png')

    print(f'reduced dimension r2 = {svd2.r}')
    print(f'retained energy 2 = {svd2.ret_energy[svd2.r]}')
    svd2.save(root_dir + 'save_for_later/', prefix=f'Eta_{center_opt}_r{svd2.r}_{n_year_train}trainingyrs') ## saves Tr and Qhat
    svd2.plot_decay(sval_dir + f'Etasvals_{center_opt}_{n_year_train}trainyrs_r{svd2.r}.png')

    print(f'reduced dimension r3 = {svd3.r}')
    print(f'retained energy 3 = {svd3.ret_energy[svd3.r]}')
    svd3.save(root_dir + 'save_for_later/', prefix=f'TS_{center_opt}_r{svd3.r}_{n_year_train}trainingyrs') ## saves Tr and Qhat
    svd3.plot_decay(sval_dir + f'TSsvals_{center_opt}_{n_year_train}trainyrs_r{svd3.r}.png')


    ## compute reduced forcings
    #print('compute forcings')
    print('compute forcing inputs')

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

    inputsT, inputsU = build_rom_inputs(Tmax, tau4, n_year_train, n_days, block=True)

    # ### transform forcings

    # ds_surface = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="h5netcdf",  
    #                     decode_timedelta=False, chunks={}).isel(k=0, time=slice(0, 360*n_year_train, n_days))

    # # Load global centers 
    # centers = {
    #     'U': xr.open_dataset(root_dir + f'save_for_later/centerU_{center_opt}_{n_year_train}yrs.nc').center.values,
    #     'T': xr.open_dataset(root_dir + f'save_for_later/centerT_{center_opt}_{n_year_train}yrs.nc').center.values,
    #     'S': xr.open_dataset(root_dir + f'save_for_later/centerS_{center_opt}_{n_year_train}yrs.nc').center.values
    # }
    # alphas  = {'T': alphaT,  'S': alphaS,  'U': alphaU}
    # B, C = project_surface_forcings(ds_surface, svd.Tr_global,centers,alphas,forcings,nx, ny, nz, [center_opt, center_opt], n_year_train)

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
    weights_A = np.logspace(-8, 1, n_train_A)



    # ## regularization parameter search 
    # sweep = TikhonovSweep(
    #     Q_=svd1.Qhat_global,
    #     inputs=inputs,
    #     B=B,
    #     C=C,
    #     r=svd.r,
    #     nt=nt,
    #     t=t,
    #     weights_A=weights_A,
    #     weights_H=weights_H,
    # )
'''
    reg_A, reg_H, best_err = sweep.run()
    print("Best:", best_err, reg_A, reg_H)



    # save best so easily can predict again
    save_dir = root_dir + 'roms/'
 #   sweep.save(save_dir + f'Qrom_sweep_r{r}_{center_opt}_{n_year_train}trainingyrs.pkl')

    inputs = build_rom_inputs(Tmax, tau4, n_year_train + n_year_predict, n_days)

    model, Q_ROM_ = sweep.fit_best_model(
        inputs_fit=inputs,
        niters=(n_year_train + n_year_predict) * int(360 / n_days) ,
    )

    # save rom for all predict years
    np.save(save_dir + f'Q_ROM_r{svd.r}_{center_opt}_{n_year_train}trainingyrs.npy', Q_ROM_)


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
    root_dir=root_dir,
    center_opt=center_opt,
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
'''