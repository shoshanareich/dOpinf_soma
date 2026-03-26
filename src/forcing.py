import numpy as np
from MITgcmutils import rdmds


def build_rom_inputs(Tmax, tau4, n_year_train, n_days, block=False):
    """
    Build ROM input time series for training period.
    Returns inputs array of shape (n_inputs, nt)
    """
    reps = int(30 / n_days)

    Tmax_train = np.tile(np.repeat(Tmax, reps), n_year_train)
    tau4_train = np.tile(np.repeat(tau4, reps), n_year_train)

  # Tmax_train = np.append(Tmax_train, Tmax[0])
  #  tau4_train = np.append(tau4_train, tau4[0])

    if block:
        return Tmax_train, tau4_train
    else:
        return np.vstack((Tmax_train, tau4_train))
    

def apply_center_and_scale(field, centers, alphas, var_name, nxy, center_opt, n_year_train):
    """
    Center and scale a surface forcing field (SST, SSS, SSU, etc.)

    Parameters
    ----------
    field : np.ndarray or xarray.DataArray
        The raw field, shape (space, time)
    centers : dict of np.ndarray / DataArray
        Centers for each variable, e.g., centers['T'], centers['S'], centers['U']
    alphas : dict
        Scaling factors for each variable
    var_name : str
        Which variable to process ('T', 'S', 'U')
    nxy : int
        Number of spatial points to select
    center_opt : str
        One of ['mean', 'IC', 'seasonal', 'monthly']
    n_year_train : int
        Number of years to tile seasonal/monthly centers
    """
    

    if center_opt == 'global_mean':
        center_rep = centers[var_name] # scalar
    elif center_opt in ['mean', 'IC']:
        # just add singleton axis for broadcasting along time
        center_sel = centers[var_name][:nxy]  # select first nxy spatial points
        center_rep = center_sel[:, None]  # (space, 1)
    else:
        # tile seasonal/monthly pattern along time for n_year_train
        center_sel = centers[var_name][:nxy]  # select first nxy spatial points
        cycle_len = center_sel.shape[1]
    
        # Calculate how many full repeats (e.g., 1801 // 360 = 5)
        num_repeats = field.shape[1] // cycle_len
        
        # Calculate the remainder (e.g., 1801 % 360 = 1)
        remainder = field.shape[1] % cycle_len
        
        # Tile the full years
        full_tiles = np.tile(center_sel, (1, num_repeats))
        
        #  Append the partial cycle to hit exactly n_time
        if remainder > 0:
            leftover_cols = center_sel[:, :remainder]
            center_rep = np.hstack([full_tiles, leftover_cols])
        else:
            center_rep = full_tiles


    # center and scale
    return (field - center_rep) #/ alphas[var_name]



def project_surface_forcings(ds_surface, Tr_global, centers, alphas, forcings, nx, ny, nz, center_opt, n_year_train, block=False):
    """
        Project surface forcings onto the POD basis
        F_project,sst = V_sst^T @ F
        where V_sst = QTr = [(SST - center)/alpha]Tr
        Then scale forcings since data is scaled (no need to center)
        """
    nxy = nx * ny      # 2D size (Surface)
    nxyz = nxy * nz    # 3D size (Full volume)

    # --- Define slicing based on [U, V, Eta, T, S] ---
    u_start   = 0
    v_start   = nxyz
    eta_start = 2 * nxyz
    t_start   = 2 * nxyz + nxy   
    s_start   = 3 * nxyz + nxy


    # --- Transform Surface Snapshots ---
    SST = ds_surface.T.stack(space=('j', 'i')).T.values
    SSS = ds_surface.S.stack(space=('j', 'i')).T.values
    SSU = ds_surface.U.stack(space=('j', 'i_g')).T.values
        

    # --- apply training centering & scaling ---
    SSTc = apply_center_and_scale(SST, centers, alphas, 'T', nxy, center_opt[0], n_year_train)
    SSSc = apply_center_and_scale(SSS, centers, alphas, 'S', nxy, center_opt[0], n_year_train)
    SSUc = apply_center_and_scale(SSU, centers, alphas, 'U', nxy, center_opt[1], n_year_train)


    # --- project ---
    if block:
        # Tr_global is list where Tr_global = [Tr_uve, Tr_TS]
        V_sst = SSTc @ Tr_global[1]
        V_sss = SSSc @ Tr_global[1]
        V_ssu = SSUc @ Tr_global[0]

        r_uve = V_ssu.shape[1]
        r_ts = V_sst.shape[1]
        r_total = r_uve + r_ts
        C = np.zeros((r_total))
        B = np.zeros((r_total, 2))
        C[-r_ts:] = (
            V_sst.T @ (forcings['c_theta'] )#* alphas['T'])
        + V_sss.T @ (forcings['s_star'] )#* alphas['S'])
        )
        B[-r_ts:,0] = V_sst.T @ forcings['b_theta'] 
        B[:r_uve, 1] = V_ssu.T @ forcings['g']

    
    else:
        V_sst = SSTc @ Tr_global
        V_sss = SSSc @ Tr_global
        V_ssu = SSUc @ Tr_global
    
        # --- assemble ROM forcing matrices ---
        C = (
            V_sst.T @ (forcings['c_theta'] )#* alphas['T'])
        + V_sss.T @ (forcings['s_star'] )#* alphas['S'])
        )

        B = np.column_stack((
            V_sst.T @ (forcings['b_theta'] ),#* alphas['T']),
            V_ssu.T @ (forcings['g'] ),#* alphas['U']),
        ))

    return B, C





def project_surface_forcings_multiple(SSTc, SSSc, SSUc, Tr_global, forcings):
    """
        Project surface forcings onto the POD basis
        F_project,sst = V_sst^T @ F
        where V_sst = QTr = [(SST - center)/alpha]Tr
        Then scale forcings since data is scaled (no need to center)
        """

    V_sst = SSTc @ Tr_global
    V_sss = SSSc @ Tr_global
    V_ssu = SSUc @ Tr_global

    # --- assemble ROM forcing matrices ---
    C = (
        V_sst.T @ (forcings['c_theta'] )#* alphas['T'])
    + V_sss.T @ (forcings['s_star'] )#* alphas['S'])
    )

    B = np.column_stack((
        V_sst.T @ (forcings['b_theta'] ),#* alphas['T']),
        V_ssu.T @ (forcings['g'] ),#* alphas['U']),
    ))

    return B, C













class OceanForcing:
    def __init__(self, grid_dir, nx, ny, xo, yo, xc, yc, dx, dy):
        """
        Container for: 
        - grid information
        - salinity forcing
        - temperature forcing
        - wind forcing
        """
        self.grid_dir = grid_dir

        # Store parameters
        self.nx = nx
        self.ny = ny
        self.xo = xo
        self.yo = yo
        self.xc = xc
        self.yc = yc
        self.dx = dx
        self.dy = dy

        # Domain extents
        self.xeast = xo + (nx - 2/dx) * dx
        self.ynorth = yo + (ny - 2/dx) * dy
        self.L = (ny - 2/dx) / dy

        # Build grid
        self.build_grid()

        # Load hFacC (C-grid mask)
        self.hfacc = rdmds(grid_dir + 'hFacC')[0]

    def build_grid(self):
        '''grid'''
        self.x = np.linspace(self.xo, self.xeast, self.nx)
        self.y = np.linspace(self.yo, self.ynorth, self.ny)

        self.Y, self.X = np.meshgrid(self.y, self.x, indexing='ij')

    def compute_salinity_forcing(self, Smin=35, Smax=36, tau_s=15552000):
        """
        Compute time-independent salinity restoring field.
        """
        Srest = (Smax - Smin) / (self.ny - 2) / self.dy * (self.ynorth - self.Y) + Smin

        self.s_star_const = Srest * self.hfacc / tau_s
        return self.s_star_const

    def compute_temp_forcing(self, Tmin=-1.8, Tmax_low=25, Tmax_high=30, tau_theta=2592000):
        """
        Compute seasonal temperature forcing coefficients.
        
        Tmax is a 12-month cosine cycle:
            Tmax = - (T_high - T_low)/2 * cos(month) + (T_high + T_low)/2
        """
        # --- 12-month seasonal Tmax ---
        self.Tmax = - (Tmax_high - Tmax_low) / 2 * np.cos(np.linspace(0, 2*np.pi, 12)) \
                    + (Tmax_high + Tmax_low) / 2

        # --- spatial coefficients ---
        c_theta = self.hfacc * (-Tmin * (self.ynorth - self.Y) / ((self.ny - 8) * self.dy) + Tmin) / tau_theta
        b_theta = self.hfacc * ((self.ynorth - self.Y) / ((self.ny - 8) * self.dy)) / tau_theta

        self.c_theta = c_theta.ravel()
        self.b_theta = b_theta.ravel()

        return self.Tmax, self.c_theta, self.b_theta

        

    def compute_wind_forcing(self, xi=0.5):
        """
        Compute time-independent wind forcing g(phi).
        """

        # XG, YC staggered grid
        x = np.linspace(self.xo - self.dx, self.xeast, self.nx)
        y = np.linspace(self.yo - self.dy, self.ynorth, self.ny) + self.dy / 2

        Y, X = np.meshgrid(y, x, indexing='ij')

        delY = (Y - self.yc) / 30

        g = ((1 - xi * delY) *
             np.exp(-np.square(delY)) *
             np.cos(np.pi * delY)) * self.hfacc

        self.wind_g = g
        return self.wind_g

    def compute_wind_seasonal_amplitude(self, min_val=0.1, max_val=0.4):
        """
        Compute seasonal (time-dependent) cos cycle for wind stress (tau)
        """
        mo = np.linspace(0, 2 * np.pi, 12)

        k = (max_val - min_val) / 2
        a = (max_val - min_val) / 2

        tau4 = a * np.cos(mo) + (min_val + k)

        self.tau4 = tau4
        return self.tau4
