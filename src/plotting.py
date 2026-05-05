import numpy as np
import xarray as xr
import cmocean as cmo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import opinf
from IPython.display import HTML


def animate_psi(fld_rom, fld_fom, timesteps, vmin=-90, vmax=90, cmap='RdBu_r', title='\psi'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        mesh1 = axes[0].pcolormesh(fld_fom[0], vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(mesh1, ax=axes[0])
        axes[0].set_title(fr'${{{title}}}_{{MITgcm}}$ Month 1', size=20)

        mesh2 = axes[1].pcolormesh(fld_rom[0], vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(mesh2, ax=axes[1])
        axes[1].set_title(fr'${{{title}}}_{{OpInf}}$ Month 1', size=20)

        error = np.abs(fld_fom - fld_rom)
        error[error == 0] = np.nan
        # Use data-driven vmax for error to see detail
        mesh3 = axes[2].pcolormesh(error[0], vmin=0, vmax=np.nanmax(error), cmap='Reds')
        fig.colorbar(mesh3, ax=axes[2])
        axes[2].set_title(f'Absolute Error Month 1')

        def update_plot(t):
            mesh1.set_array(fld_fom[t].ravel())
            axes[0].set_title(fr'${{{title}}}_{{MITgcm}}$ Month {t+1}')
            mesh2.set_array(fld_rom[t].ravel())
            axes[1].set_title(fr'${{{title}}}_{{OpInf}}$ Month {t+1}')
            mesh3.set_array(error[t].ravel())
            axes[2].set_title(f'Absolute Error Month {t+1}')
            return mesh1, mesh2, mesh3

        ani = animation.FuncAnimation(fig, update_plot, frames=timesteps, repeat=False, blit=False)
        return ani



# --- Global Configuration for Variables ---
VAR_CONFIG = {
    "Eta": {
        "cmap": "RdBu_r",
        "vmin": -1,       # Set to a float like -0.5 for fixed scales
        "vmax": 1,       # Set to a float like 0.5 for fixed scales
        "label": "Sea Surface Height Anomaly [m]",
        "units": "m",
        "err_cmap": "Reds",
        "title": "SSH"
    },
    "T": {
        "cmap": "inferno",
        "vmin": -2,
        "vmax": 30,
        "label": "Temperature [°C]",
        "units": "°C",
        "err_cmap": "Purples",
        "title": "Temp"
    },
    "S": {
        "cmap": "blues",
        "vmin": 30,
        "vmax": 38,
        "label": "Salinity [ppt]",
        "units": "ppt",
        "err_cmap": "Purples",
        "title": "Salinity"
    },
    "U": {
        "cmap": "RdBu_r",
        "vmin": -1,
        "vmax": 1,
        "label": "Zonal Velocity [m/s]",
        "units": "m/s",
        "err_cmap": "Reds",
         "title": "U"
    },
    "V": {
        "cmap": "RdBu_r",
        "vmin": -1,
        "vmax": 1,
        "label": "Meridional Velocity [m/s]",
        "units": "m/s",
        "err_cmap": "Reds",
        "title": "V"
    }
}

class ROMPlotter:
    def __init__(self, Q_fom=None, Q_rom=None, Q_rom2=None, r=None, n_year_train=None, n_year_predict=None,
                 n_days=None, n_days_per_year=None, k = None, var="SSH"):
        self.Q_fom = Q_fom
        self.Q_rom = Q_rom
        self.Q_rom2 = Q_rom2
        self.r = r
        self.n_year_train = n_year_train
        self.n_year_predict = n_year_predict
        self.n_days = n_days
        self.n_days_per_year = n_days_per_year
        self.k = k
       #self.title = title

        # Load variable-specific settings
        config = VAR_CONFIG.get(var, VAR_CONFIG[var])
        self.cmap = config["cmap"]
        self.fixed_vmin = config["vmin"]
        self.fixed_vmax = config["vmax"]
        self.y_label = config["label"]
        self.units = config["units"]
        self.err_cmap = config.get("err_cmap", "Reds")
        self.title = config.get("title")

    def _get_bounds(self, data):
        """Helper to determine vmin/vmax based on fixed config or data limits."""
        if self.fixed_vmax is not None:
            return self.fixed_vmin, self.fixed_vmax
        
        # If no fixed bounds, center the colormap around zero
        limit = np.nanmax(np.abs(data))
        return -limit, limit
    
    def compute_rmse_stats(self):
        """Calculates RMSE relative to ground truth, climatology, and persistence."""
        # Ensure data is flattened to (space, time) if not already
        fom = self.Q_fom.values.reshape(self.Q_fom.shape[0], -1).T
        rom = self.Q_rom.reshape(self.Q_rom.shape[0], -1).T
        
        # 1. Prediction RMSE
        rmse = np.sqrt(np.nansum((rom - fom)**2, axis=0) / rom.shape[0])

        # 2. Climatology RMSE (Mean of training period)
        train_steps = self.n_year_train * self.n_days_per_year
        clim = np.nanmean(fom[:, :train_steps], axis=1)
        rmse_clim = np.sqrt(np.nansum((fom - clim[:, np.newaxis])**2, axis=0) / fom.shape[0])

        # 3. Persistence RMSE (Initial Condition of the series)
        IC = fom[:, 0]
        rmse_pers = np.sqrt(np.nansum((fom - IC[:, np.newaxis])**2, axis=0) / fom.shape[0])

        return rmse, rmse_clim, rmse_pers

    def plot_rel_l2_error(self, save_path=None):
        fom = self.Q_fom.values.reshape(self.Q_fom.shape[0], -1).T
        rom = self.Q_rom.reshape(self.Q_rom.shape[0], -1).T
        abs_err, norm_err = opinf.post.lp_error(fom, rom, normalize=True)
        ymax = np.nanmax(norm_err)
        nt_total = (self.n_year_train + self.n_year_predict) * int(360 / self.n_days) + 1

        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(np.arange(norm_err.size), norm_err, color="red", linewidth=2, label=f"r = {self.r}")
        ax.vlines(self.n_year_train * 360, ymin=0, ymax=1.1*ymax, lw=2, color="gray", ls="--")
        ax.set_xlabel("Year", fontsize=15)
        ax.set_xticks(range(0, nt_total, 360))
        ax.set_xticklabels(range(0, self.n_year_train + self.n_year_predict + 1))
        ax.set_ylabel("Relative Error", fontsize=15)
        ax.set_ylim(0, 1.1*ymax)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(loc="lower right", fontsize=15)
        ax.set_title(fr"Relative $\ell^2$ error k = {self.k}", fontsize=18)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return norm_err

    def plot_timeseries_locations(self, loc_indices=None, save_path=None):
        if loc_indices is None:
            loc_indices = [(58*4,124), (34*4,124), (20*4,124)]
        nt_total = self.n_days_per_year * (self.n_year_train + self.n_year_predict)
        fig, axes = plt.subplots(len(loc_indices), 1, figsize=(15, 6))
        axes = axes.ravel()
        lat_labels = ['High Latitude', 'Mid Latitude', 'Low Latitude']

        i = 0
        for ax, (lat_idx, lon_idx), label_text in zip(axes, loc_indices, lat_labels):
            ax.plot(self.Q_fom[:nt_total, lat_idx, lon_idx], label='MITgcm', lw=2)
            ymin, ymax = ax.get_ylim()
            ax.plot(self.Q_rom[:nt_total, lat_idx, lon_idx], label='ROM', lw=2)
           # ax.plot(self.Q_rom2[:nt_total, lat_idx, lon_idx], label='ROMvar', lw=2)
            ax.set_xticks(range(0, nt_total+1, self.n_days_per_year))
            ax.set_xticklabels(range(0, self.n_year_train + self.n_year_predict + 1))
            ax.set_ylim(ymin-0.03, ymax+0.03)
            ax.set_yticks(np.linspace(np.round(ymin+0.01, 2), np.round(ymax-0.01, 2), 3))
            ax.vlines(self.n_year_train * self.n_days_per_year, ymin-0.015, ymax+0.015, lw=2, color='gray', ls='--')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.text(0.07, 0.87, label_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=18)
            ax.grid(True)
            if i == 0:
                ax.legend(loc='upper right', prop={'size': 14})
            i += 1

        # Dynamic label from config
        fig.text(0.05, 0.5, self.y_label, va='center', rotation='vertical', size=20)
        plt.subplots_adjust(hspace=0.07)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_timeseries_zonalavg(self, loc_indices=None, save_path=None, title=None):
        if loc_indices is None:
            loc_indices = [56*4, 36*4, 20*4]
        nt_total = self.n_days_per_year * (self.n_year_train + self.n_year_predict)
        fig, axes = plt.subplots(len(loc_indices), 1, figsize=(15, 6))
        axes = axes.ravel()
        lat_labels = ['High Latitude', 'Mid Latitude', 'Low Latitude']

        i = 0
        for ax,  lat_idx, label_text in zip(axes, loc_indices, lat_labels):
            ax.plot(self.Q_fom[:nt_total, lat_idx, :].mean(dim='i'), label='MITgcm', lw=2)
            ax.plot(np.nanmean(self.Q_rom[1:nt_total, lat_idx, :], axis=-1), label='ROM', lw=2)
            ymin, ymax = ax.get_ylim()
            ax.set_xticks(range(0, nt_total+1, self.n_days_per_year))
            ax.set_xticklabels(range(0, self.n_year_train + self.n_year_predict + 1))
            ax.set_ylim(ymin-0.03, ymax+0.03)
            ax.set_yticks(np.linspace(np.round(ymin+0.01, 2), np.round(ymax-0.01, 2), 3))
            ax.vlines(self.n_year_train * self.n_days_per_year, ymin-0.015, ymax+0.015, lw=2, color='gray', ls='--')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.text(0.07, 0.87, label_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=18)
            
            ax.grid(True)
            if i == 0:
                ax.legend(loc='upper right', prop={'size': 14})
            i += 1

        # Dynamic label from config
        fig.text(0.08, 0.9, self.units, va='center',  size=16)
        fig.suptitle(f'Zonally-Averaged {title}', size=20)
       # fig.text(0.05, 0.5, self.y_label, va='center', rotation='vertical', size=20)
        plt.subplots_adjust(hspace=0.07)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_monthly_avg(self, Eta_ROM_monthlymean, Eta_FOM_monthlymean, month_idx=12, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        rom_plt = Eta_ROM_monthlymean[month_idx]
        fom_plt = Eta_FOM_monthlymean[month_idx]

        vmin, vmax = self._get_bounds(fom_plt) 

        # FOM
        ax = axes[0]
        mesh = ax.pcolormesh(fom_plt, cmap=self.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax).ax.tick_params(labelsize=14)
        ax.set_title(fr'${{{self.title}}}_{{MITgcm}}$ Month {month_idx}', size=20)

        # ROM
        ax = axes[1]
        mesh = ax.pcolormesh(rom_plt, cmap=self.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax).ax.tick_params(labelsize=14)
        ax.set_title(fr'${{{self.title}}}_{{OpInf}}$ Month {month_idx}', size=20)

        # Absolute error
        abs_error = np.abs(fom_plt - rom_plt)
        ax = axes[2]
        mesh = ax.pcolormesh(abs_error, cmap=self.err_cmap, vmin=0, vmax=np.nanmax(abs_error))
        fig.colorbar(mesh, ax=ax).ax.tick_params(labelsize=14)
        ax.set_title('Absolute error', size=20)

        for ax in axes:
            ax.set_yticks([4, 86, 170, 244], [r'$15^\circ$N', r'$35^\circ$N', r'$55^\circ$N', r'$75^\circ$N'], size=12)
            ax.set_xticks([4, 86, 170, 244], [r'$0^\circ$E', r'$20^\circ$E', r'$40^\circ$E', r'$60^\circ$E'], size=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def animate_monthly(self, fld_rom, fld_fom, timesteps, vmax_err=1):
        vmin, vmax = self._get_bounds(fld_fom)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        mesh1 = axes[0].pcolormesh(fld_fom[0], vmin=vmin, vmax=vmax, cmap=self.cmap)
        fig.colorbar(mesh1, ax=axes[0])
        axes[0].set_title(fr'${{{self.title}}}_{{MITgcm}}$ Month 1', size=20)

        mesh2 = axes[1].pcolormesh(fld_rom[0], vmin=vmin, vmax=vmax, cmap=self.cmap)
        fig.colorbar(mesh2, ax=axes[1])
        axes[1].set_title(fr'${{{self.title}}}_{{OpInf}}$ Month 1', size=20)

        error = np.abs(fld_fom - fld_rom)
        error[error == 0] = np.nan
        # Use data-driven vmax for error to see detail
        mesh3 = axes[2].pcolormesh(error[0], vmin=0, vmax=vmax_err, cmap=self.err_cmap)
        fig.colorbar(mesh3, ax=axes[2])
        axes[2].set_title(f'Absolute Error Month 1')

        def update_plot(t):
            mesh1.set_array(fld_fom[t].ravel())
            axes[0].set_title(fr'${{{self.title}}}_{{MITgcm}}$ Month {t+1}')
            mesh2.set_array(fld_rom[t].ravel())
            axes[1].set_title(fr'${{{self.title}}}_{{OpInf}}$ Month {t+1}')
            mesh3.set_array(error[t].ravel())
            axes[2].set_title(f'Absolute Error Month {t+1}')
            return mesh1, mesh2, mesh3

        ani = animation.FuncAnimation(fig, update_plot, frames=timesteps, repeat=False, blit=False)
        return ani
    
    def plot_pearson_maps(self, save_path=None):
        """Plots Training and Prediction Pearson Correlation Maps."""
        train_steps = self.n_year_train * self.n_days_per_year
        
        # Create DataArray for ROM to use xarray.corr
        rom_da = xr.DataArray(self.Q_rom, coords=self.Q_fom.coords, dims=self.Q_fom.dims)
        
        map_train = xr.corr(self.Q_fom[:train_steps], rom_da[:train_steps], dim='time')
        map_predict = xr.corr(self.Q_fom[train_steps:], rom_da[train_steps:], dim='time')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        
        for ax, data, label in zip(axes, [map_train, map_predict], ['Training', 'Prediction']):
            mesh = ax.pcolormesh(data, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set_title(label, size=16)
        
        fig.colorbar(mesh, ax=axes.ravel().tolist())
        fig.suptitle(f'Pearson Correlation Coefficient of {self.title}', size=18, y=1.05)
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.show()



def plot_surface_rmse_summary(plotters, labels, save_path=None):
    """
    Plots a 3-panel RMSE comparison for a list of ROMPlotter instances.
    """
    fig, axes = plt.subplots(len(plotters), 1, figsize=(15, 3 * len(plotters)))
    
    for i, (ax, p, label) in enumerate(zip(axes, plotters, labels)):
        rmse, r_clim, r_pers = p.compute_rmse_stats()
        
        ax.plot(rmse, color='red', label='Prediction' if i == 0 else "")
        ax.plot(r_clim, color='black', label='Climatology' if i == 0 else "")
        ax.plot(r_pers, color='blue', label='Persistence' if i == 0 else "")
        
        # Formatting
        ymax = np.nanmax(r_pers)
        ax.vlines(p.n_year_train * 360, ymin=0, ymax=1.2*ymax, lw=2, color="gray", ls="--")
        ax.set_ylim(0, 1.2*ymax)
        ax.set_xticks(range(0, rmse.shape[0]+1, 360))
        
        if i == len(plotters) - 1:
            ax.set_xticklabels(range(0, p.n_year_train + p.n_year_predict + 1))
            ax.set_xlabel("Year", fontsize=15)
        else:
            ax.set_xticklabels([])

        # Vertical centered labels
        ax.text(-0.08, 0.5, label, rotation=90, va='center', ha='center', 
                transform=ax.transAxes, size=16)
        ax.grid(True)
        if i == 0: ax.legend(loc='lower right', prop={'size': 14})

    fig.suptitle('RMSE of Surface Fields', size=20)
    plt.subplots_adjust(hspace=0.1)
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()





import cmocean as cmo

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
        
        (fom_plt - rom_da).plot(y='Z', ax=axes[2], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title("FOM - ROM")
        
        fig.suptitle(main_title, size=18)
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.show()



def plot_timeseries_atdepth(SST_FOM, SST_ROM, var_FOM, var_ROM,  n_year_train, n_year_predict, depth=0, save_path=None):

    fig, axes = plt.subplots(2, 1, figsize=(15, 4))
    axes = axes.ravel()


    ax = axes[0]
    ax.plot(np.nanmean(SST_FOM, axis=0), label = 'MITgcm')
    ax.plot(np.nanmean(SST_ROM, axis=0), label = 'ROM')
    #ymin, ymax = ax.get_ylim()
    #ymin = ymax -1
    ymin=9.5
    ymax=10.8
    ax.vlines(n_year_train * 360, ymin=ymin, ymax=ymax, lw=2, color="gray", ls="--")
    ax.set_ylim(ymin, ymax)
    #ax.set_xlabel("Year", fontsize=15)
    ax.set_xticks(range(0, SST_ROM.shape[1]+1, 360))
    #ax.set_xticklabels(range(0, n_year_train + n_year_predict +1))
    #ax.set_ylabel(label)
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=14)
    #  ax.text(0.07, 0.87, label_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=18)
    ax.grid(True)
    ax.legend(loc='upper right', prop={'size': 14})
    ax.set_title('Surface Temperature', size=16)


    ax = axes[1]
    ax.plot(np.nanmean(var_FOM, axis=0), label='MITgcm')
    ax.plot(np.nanmean(var_ROM, axis=0), label='ROM')
    ymin, ymax = ax.get_ylim()
   # ymin = 4.63
   # ymax=4.7
    ax.vlines(n_year_train * 360, ymin=ymin, ymax=ymax, lw=2, color="gray", ls="--")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Year", fontsize=15)
    ax.set_xticks(range(0, SST_ROM.shape[1]+1, 360))
    ax.set_xticklabels(range(0, n_year_train + n_year_predict +1))
    #ax.set_ylabel(label)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #  ax.text(0.07, 0.87, label_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=18)
    ax.grid(True)
    #ax.legend(loc='lower right', prop={'size': 14})
    ax.set_title(f'Temperature at {depth} m', size=16)

    # Dynamic label from config
    fig.text(0.08, 0.9, r'$^\circ$ C', va='center',  size=16)
    plt.subplots_adjust(hspace=0.6)

    if save_path: plt.savefig(save_path)
    plt.show()
        













import ipywidgets
from IPython.display import display
import os
import sys
sys.path.append('/home/goldberg/dinocean/dinocean/neverworld2/')
from gif import *

class PsiPlotter:
    def __init__(self, psi_fom_ds, psi_rom_ds, n_year_train, n_days, root_dir):
        """
        Initializes the plotter and pre-calculates means.
        
        Args:
            psi_fom_ds: xarray Dataset containing FOM psi
            psi_rom_ds: xarray Dataset containing ROM psi
            n_year_train: Number of training years
            n_days: Stride/Sampling interval
            root_dir: Base directory for saving outputs
        """
        self.root_dir = root_dir
        self.n_year_train = n_year_train
        self.psi_fom = psi_fom_ds.psi_bt
        self.psi_rom = psi_rom_ds.psi
        
        # 1. Compute Monthly Means
        # 30 days / n_days gives the number of snapshots in a month
        snapshots_per_month = int(30 / n_days)
        self.fom_monthly = self.psi_fom.coarsen(time=snapshots_per_month, boundary='trim').mean()
        self.rom_monthly = self.psi_rom.coarsen(time=snapshots_per_month, boundary='trim').mean()
        
        # 2. Compute Time Averages
        self.fom_time_mean = self.fom_monthly.mean(dim='time')
        self.rom_time_mean = self.rom_monthly.mean(dim='time')

    def plot_comparison_summary(self, month_offset=-1, save_path=None):
        """Creates the 2x2 plot of Instantaneous vs Time-Averaged Psi."""
        # Calculate specific instant (default is 6 months into prediction)
        t_idx = int(self.n_year_train * 12 + month_offset)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        axes = axes.ravel()
        levels = np.arange(-90, 95, 5)
        cmap = 'RdBu_r'

        # Row 1: Instantaneous
        m0 = axes[0].contourf(self.fom_monthly.isel(time=t_idx), cmap=cmap, levels=levels, extend='both')
        fig.colorbar(m0, ax=axes[0])
        axes[0].set_title(r'$\psi_{MITgcm}$ at Month ' + str(t_idx))

        m1 = axes[1].contourf(self.rom_monthly.isel(time=t_idx), cmap=cmap, levels=levels, extend='both')
        fig.colorbar(m1, ax=axes[1])
        axes[1].set_title(r'$\psi_{ROM}$ at Month ' + str(t_idx))

        # Row 2: Time Averages
        m2 = axes[2].contourf(self.fom_time_mean, cmap=cmap, levels=levels, extend='both')
        fig.colorbar(m2, ax=axes[2])
        axes[2].set_title(r'Time-Averaged $\psi_{MITgcm}$')

        m3 = axes[3].contourf(self.rom_time_mean, cmap=cmap, levels=levels, extend='both')
        fig.colorbar(m3, ax=axes[3])
        axes[3].set_title(r'Time-Averaged $\psi_{ROM}$')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def generate_gif(self, prefix="psi_evolution", fps=2):
        """Generates a GIF of the prediction period."""
        # Slice data to start after training
        start_idx = 12 * self.n_year_train
        fld_fom = self.fom_monthly.values[start_idx:]
        fld_rom = self.rom_monthly.values[start_idx:]
        
        # Pre-calculate error for performance
        fld_err = np.abs(fld_fom - fld_rom)
        fld_err[fld_err == 0] = np.nan
        
        def make_frame(t):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            levels = np.arange(-90, 95, 15)
            
            # FOM
            m1 = axes[0].contourf(fld_fom[t], cmap='RdBu_r', levels=levels, extend='both')
            fig.colorbar(m1, ax=axes[0])
            axes[0].set_title(fr'$\psi_{{MITgcm}}$ Month {t}')
            
            # ROM
            m2 = axes[1].contourf(fld_rom[t], cmap='RdBu_r', levels=levels, extend='both')
            fig.colorbar(m2, ax=axes[1])
            axes[1].set_title(fr'$\psi_{{OpInf}}$ Month {t}')
            
            # Error
            m3 = axes[2].contourf(fld_err[t], cmap='RdBu_r', levels=levels, extend='both')
            fig.colorbar(m3, ax=axes[2])
            axes[2].set_title(f'Absolute Error Month {t}')
            
            return fig, axes

        # Initialize GIFmaker
        gif_dir = os.path.join(self.root_dir, 'for_gif/')
        os.makedirs(gif_dir, exist_ok=True)
        gm = GIFmaker(gif_dir, prefix=prefix, dpi=150)
        
        # Create frames
        gm.make_frames(
            nt=fld_fom.shape[0],
            frame_func=make_frame,
            stride=1
        )
        
        # Compile to GIF
        gm.to_gif(fps=fps)
        print(f"GIF saved with prefix {prefix} in {self.root_dir}for_gif/")