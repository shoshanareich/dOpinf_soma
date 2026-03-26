import numpy as np
import xarray as xr
import opinf
import xmitgcm
import json
import cmocean as cmo
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import sys
sys.path.append('/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/src')
from utils import *
from plotting import *





#### training config


root_dir = '/scratch/shoshi/soma4/dOpInf_results/'
snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'


nx = 248
ny = nx
nz = 31

## training options

r = sys.argv[1]
config_path = sys.argv[2]

with open(config_path, 'r') as f:
    config = json.load(f)

# Extract training options
n_year_train = config['n_year_train']
n_year_predict = config['n_year_predict']
n_days = config['n_days']
center_opt = config['center_opt']
scale = config['scale']

n_days_per_year = int(360/n_days)

# check for target_ret_energy or r
target_ret_energy = config.get('target_ret_energy')


#r = config.get('r')

#target_ret_energy = 0.85 # define target retained energy for the dOpInf ROM
#r = 31

data_dir = root_dir + 'save_roms/' + f'{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}/'

grid = xmitgcm.open_mdsdataset('/scratch/shoshi/soma4/grid/', iters=None)




### read in learned ROM
Q_ROM_ = np.load(data_dir + 'Q_ROM_.npy')

### need Tr to project back up
Tr = np.load(data_dir + 'Tr.npy')



### tranform ROM  surface fields

SST_ROM, SST_ROM_3d = transform_and_project_k('T', n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, k=0,anom=False) 
SSH_ROM, SSH_ROM_3d = transform_and_project_k('Eta', n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, k=0,anom=False) 
SSU_ROM, _ = transform_and_project_k('U', n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, k=0,anom=False) 
SSV_ROM, _ = transform_and_project_k('V', n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, k=0,anom=False) 


## read in FOM
SST_FOM, SST_FOM_3d = load_var_fom_k('T', n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,k=0, anom=False)
SSH_FOM, SSH_FOM_3d = load_var_fom_k('Eta', n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,k=0, anom=False)
SSU_FOM, _ = load_var_fom_k('U', n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,k=0, anom=False)
SSV_FOM, _ = load_var_fom_k('V', n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,k=0, anom=False)

speed_ROM = np.sqrt(SSU_ROM**2 + SSV_ROM**2)
speed_val = np.sqrt(SSU_FOM.values**2 + SSV_FOM.values**2)
speed_FOM = xr.DataArray(
    speed_val, 
    coords=SSU_FOM.coords, 
    dims=SSU_FOM.dims, 
    name="Surface_Speed"
)





def plot_vertical_profile(Q_fom, Q_rom, n_year_train, n_days_per_year, time_idx=None, save_path=None):
        """Plots Depth Profile (Time-Averaged or Instantaneous)."""
        # Select data
        if time_idx is None:
            # Time-mean over prediction period
            train_steps = n_year_train * n_days_per_year
            fom_plt = Q_fom.isel(time=slice(train_steps, None)).mean(dim='time')
            rom_plt = Q_rom[train_steps:].mean(axis=0) # Assuming numpy
            main_title = f'Vertical Profile of Time-Mean Temperature'
        else:
            fom_plt = Q_fom.isel(time=time_idx)
            rom_plt = Q_rom[time_idx]
            main_title = f'Vertical Profile of Temperature at an Instant'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plotting logic
        fom_plt.plot(y='Z', ax=axes[1], cmap=cmo.cm.thermal)
        axes[1].set_title("FOM")
        
        # For ROM, we wrap back to DataArray to use xarray plot logic
        rom_da = xr.DataArray(rom_plt, coords=fom_plt.coords, dims=fom_plt.dims)
        rom_da.plot(y='Z', ax=axes[0], cmap=cmo.cm.thermal)
        axes[0].set_title("ROM")
        
        (fom_plt - rom_da).plot(y='Z', ax=axes[2], cmap='RdBu_r')
        axes[2].set_title("FOM - ROM")
        
        fig.suptitle(main_title, size=18)
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.show()



### vertical cross-section


i=124

var = 'T'

T_ROM, T_ROM_3d = transform_and_project_lon(var, n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, i=i,anom=False) 

T_FOM, T_FOM_3d = load_var_fom_lon(var, n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,i=i, anom=False)

grid['T_ROM_3d'] = xr.DataArray(T_ROM_3d, dims=('time', 'Z', 'YC'))
grid['T_FOM_3d'] = xr.DataArray(T_FOM_3d.values, dims=('time', 'Z', 'YC'))
grid['T_FOM_3d'] = grid['T_FOM_3d'].where(grid['T_FOM_3d'] !=0)
grid['T_ROM_3d'] = grid['T_ROM_3d'].where(grid['T_ROM_3d'] !=0)


plot_vertical_profile(grid['T_FOM_3d'], grid['T_ROM_3d'], n_year_train, n_days_per_year, time_idx=1800, save_path=data_dir + f'Tvert_prof_instant_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png')
plot_vertical_profile(grid['T_FOM_3d'], grid['T_ROM_3d'], n_year_train, n_days_per_year, save_path=data_dir + f'Tvert_prof_timemean_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png')


### plot T at different depths

var = 'T'
k = 19

var_ROM, var_ROM_3d = transform_and_project_k(var, n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, k=k,anom=False) 
var_FOM, var_FOM_3d = load_var_fom_k(var, n_year_train, n_year_predict, n_days, root_dir, center_opt, scale ,k=k, anom=False)

plot_timeseries_atdepth(SST_FOM, SST_ROM, var_FOM, var_ROM, n_year_train, n_year_predict, save_path=data_dir + f'Tdepth_timeseries_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png')



### FNO plots

sst_plotter = ROMPlotter(Q_fom=SST_FOM_3d, Q_rom=SST_ROM_3d, r=r,
                 n_year_train=n_year_train, n_year_predict=n_year_predict,
                 n_days=n_days, n_days_per_year=n_days_per_year, k=k, var='T')

ssh_plotter = ROMPlotter(Q_fom=SSH_FOM_3d, Q_rom=SSH_ROM_3d, r=r,
                 n_year_train=n_year_train, n_year_predict=n_year_predict,
                 n_days=n_days, n_days_per_year=n_days_per_year, k=k, var='Eta')

speed_FOM_3d = speed_FOM.unstack('space')
speed_ROM_3d = speed_ROM.T.reshape(speed_ROM.T.shape[0], ny, nx)
speed_plotter = ROMPlotter(Q_fom=speed_FOM_3d, Q_rom=speed_ROM_3d, r=r,
                 n_year_train=n_year_train, n_year_predict=n_year_predict,
                 n_days=n_days, n_days_per_year=n_days_per_year, k=k, var='U')

labels = [r'Surface Speed ($\frac{m}{s}$)', r'SST ($^\circ$C)', 'SSH (m)']
plot_surface_rmse_summary([speed_plotter, sst_plotter, ssh_plotter], labels, save_path=data_dir+f"rmse_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")


### SST plots
sst_plotter.plot_timeseries_locations(save_path=data_dir+f"sst_timeseries_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")


## spatial plots
ssh_plotter.plot_timeseries_locations(save_path=data_dir+f"ssh_timeseries_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")

SSH_ROM_3d_predict = SSH_ROM_3d[n_year_train*n_days_per_year:]
SSH_ROM_monthlymean = SSH_ROM_3d_predict.reshape(n_year_predict*12, int(30/n_days), nx, nx).mean(axis=1)  # monthly mean

SSH_FOM_monthlymean = (SSH_FOM_3d[n_year_train*n_days_per_year:].values).reshape(n_year_predict*12, int(30/n_days), nx, nx).mean(axis=1)

ssh_plotter.plot_monthly_avg(SSH_ROM_monthlymean, SSH_FOM_monthlymean, month_idx=13, save_path=data_dir+f"ssh_monthly_avg_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")

vid = ssh_plotter.animate_monthly(SSH_ROM_monthlymean, SSH_FOM_monthlymean, timesteps=range(SSH_FOM_monthlymean.shape[0]))
HTML(vid.to_jshtml())
vid.save(data_dir +f'ssh_monthly_avg_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.gif')


ssh_rom_da = xr.DataArray(
    SSH_ROM_3d, 
    coords=SSH_FOM_3d.coords, 
    dims=SSH_FOM_3d.dims,
    name="SSH_ROM"
)


SSH_ROM_3d_train = ssh_rom_da[:n_year_train*n_days_per_year]
SSH_FOM_3d_train = SSH_FOM_3d.isel(time=slice(0, n_year_train*n_days_per_year))

SSH_ROM_3d_predict = ssh_rom_da[n_year_train*n_days_per_year:(n_year_train+1)*n_days_per_year]
SSH_FOM_3d_predict = SSH_FOM_3d.isel(time=slice(n_year_train*n_days_per_year, (n_year_train+1)*n_days_per_year))

pearson_map_train = xr.corr(SSH_FOM_3d_train, SSH_ROM_3d_train, dim='time')
pearson_map_predict = xr.corr(SSH_FOM_3d_predict, SSH_ROM_3d_predict, dim='time')

sst_plotter.plot_pearson_maps(save_path=data_dir+f"pearson_sst_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")




## barotropic streamfunction


psi = xr.open_dataset(snapshot_dir + 'psi_20yrs.nc').isel(time=slice(0, 360*(n_year_train + n_year_predict), n_days))
psi_rom = xr.open_dataset(data_dir + f'psi_rom_{center_opt}_{scale}_r{r}_{n_year_train}trainingyrs.nc').isel(time=slice(0, 360*(n_year_train + n_year_predict), n_days))


plotter = PsiPlotter(psi, psi_rom, n_year_train, n_days, root_dir)

plotter.plot_comparison_summary(save_path=data_dir+f"psi_{center_opt}_{scale}_{n_days}days_{n_year_train}yrs_r{r}.png")

# Generate the GIF (slower, e.g., 500ms duration per frame)
plotter.generate_gif(fps=2)

