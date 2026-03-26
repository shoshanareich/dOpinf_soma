from mpi4py import MPI
import numpy as np
import xarray as xr
import xmitgcm
import zarr
import os
import shutil
import gc
import sys
sys.path.append('/home/shoshi/MITgcm_obsfit/MITgcm/utils/python/MITgcmutils/')
from MITgcmutils import rdmds

def slice_space(n, rank, size):
    '''slice along space for this rank'''
    i_start = rank * (n // size)
    i_end = (rank + 1) * (n // size)
    if rank == size - 1:
        i_end = n  # last rank takes remainder

    return i_start, i_end

# def reshape_data(nx, n_days, n_year_train, ds, rank, size):
#     ''' split data, select training snapshots, and reshape for opinf '''
#     i_start, i_end = slice_space(nx, rank, size)

#     U_rank = ds['U'].isel(i_g=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
#     V_rank = ds['V'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
#     Eta_rank = ds['Eta'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
#     T_rank = ds['T'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
#     S_rank = ds['S'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))

#     ## reshape for Q[space_stacked, time]
#     U_rank = U_rank.stack(space=('k', 'j', 'i_g')).T.data
#     V_rank = V_rank.stack(space=('k', 'j_g', 'i')).T.data
#     Eta_rank = Eta_rank.stack(space=('j', 'i')).T.data
#     T_rank = T_rank.stack(space=('k', 'j', 'i')).T.data
#     S_rank = S_rank.stack(space=('k', 'j', 'i')).T.data

#     return U_rank, V_rank, Eta_rank, T_rank, S_rank

def reshape_data(nx, n_days, n_year_train, ds, rank, size, zero_zone_end=0): #previously used 55
    ''' split data, select training snapshots, and reshape for opinf '''
    i_start, i_end = slice_space(nx, rank, size)

    # 1. Selection
    U_rank = ds['U'].isel(i_g=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
    V_rank = ds['V'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
    Eta_rank = ds['Eta'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
    T_rank = ds['T'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))
    S_rank = ds['S'].isel(i=slice(i_start, i_end), time=slice(0, 360*n_year_train, n_days))

    # 2. Zero-out the Western Boundary Zone (0 to zero_zone_end)
    # We apply the mask globally to the coordinates; Xarray handles the local slicing.
    # .where(condition, other) keeps values where condition is True, else replaces with other.
    U_rank = U_rank.where(U_rank.i_g >= zero_zone_end, 0)
    V_rank = V_rank.where(V_rank.i >= zero_zone_end, 0)
    Eta_rank = Eta_rank.where(Eta_rank.i >= zero_zone_end, 0)
    T_rank = T_rank.where(T_rank.i >= zero_zone_end, 0)
    S_rank = S_rank.where(S_rank.i >= zero_zone_end, 0)

    ## 3. Reshape for Q[space_stacked, time]
    # Note: Using .data at the end converts to the underlying Dask array
    U_rank = U_rank.stack(space=('k', 'j', 'i_g')).T.data
    V_rank = V_rank.stack(space=('k', 'j_g', 'i')).T.data
    Eta_rank = Eta_rank.stack(space=('j', 'i')).T.data
    T_rank = T_rank.stack(space=('k', 'j', 'i')).T.data
    S_rank = S_rank.stack(space=('k', 'j', 'i')).T.data

    return U_rank, V_rank, Eta_rank, T_rank, S_rank


def get_variable_rank_chunk(ds, var_name, nx, n_days, n_year_train, rank, size):
    """
    Directly pulls a slice of a variable and flattens it to (Space, Time).
    Bypasses xarray.stack() for speed and memory efficiency.
    """
    i_start, i_end = slice_space(nx, rank, size)
    t_slice = slice(0, 360 * n_year_train , n_days)
    
    # Pull the raw data slice as a NumPy array immediately
    if var_name == 'Eta':
        data = ds[var_name].variable.data[t_slice, :, i_start:i_end]
    elif var_name == 'U':
        data = ds[var_name].variable.data[t_slice, :, :, i_start:i_end]
    elif var_name == 'V':
        data = ds[var_name].variable.data[t_slice, :, :, i_start:i_end]
    else:
        data = ds[var_name].variable.data[t_slice, :, :, i_start:i_end]

    # Convert to real NumPy array (this triggers the actual disk read)
    data_np = np.array(data) 
    
    # Reshape: (Time, Dim1, Dim2, Dim3) -> (Time, Flattened_Space) -> (Flattened_Space, Time)
    nt = data_np.shape[0]
    reshaped = data_np.reshape(nt, -1).T
    
    return reshaped


def gather_distributed_array(local_array, comm, root=0):
    """
    Gathers a distributed array stacked along the first dimension (rows) 
    to the root process.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get the shape of the local chunk
    local_rows = local_array.shape[0]
    # Handle both 1D vectors and 2D matrices
    cols = local_array.shape[1] if local_array.ndim > 1 else 1
    
    # Gather the size of each chunk on the root
    # rows_counts is a list of the number of rows from each rank
    rows_counts = comm.gather(local_rows, root=root)
    
    if rank == root:
        # Gatherv expects counts in terms of flat elements (rows * cols)
        total_rows = sum(rows_counts)
        elem_counts = [r * cols for r in rows_counts]
        displacements = [sum(elem_counts[:i]) for i in range(size)]
        
        # Allocate the global buffer
        if local_array.ndim > 1:
            global_array = np.empty((total_rows, cols), dtype=local_array.dtype)
        else:
            global_array = np.empty(total_rows, dtype=local_array.dtype)
            
        # Gather data
        comm.Gatherv(local_array, [global_array, elem_counts, displacements, MPI.DOUBLE], root=root)
        return global_array
    else:
        # Non-root ranks send their data
        comm.Gatherv(local_array, None, root=root)
        return None
    










def get_Q_rank(f, nx, n_year_train, n_days, rank, size):
    i_start, i_end = slice_space(nx, rank, size)
    t_idx = slice(0, 360 * n_year_train + 1, n_days)
    
    # We load variables one by one to keep memory low
    # and use the raw [:] slicing which returns a numpy array
    def load_and_flatten(var_name, is_2d=False):
        if rank == 0: print(f"  Loading {var_name}...", flush=True)
        
        # Accessing via f.variables[name] skips all Xarray overhead
        if is_2d:
            # Shape: (time, j, i) -> (time, j, i_slice)
            data = f.variables[var_name][t_idx, :, i_start:i_end]
        else:
            # Shape: (time, k, j, i) -> (time, k, j, i_slice)
            data = f.variables[var_name][t_idx, :, :, i_start:i_end]
        
        # Reshape to (Space, Time)
        nt = data.shape[0]
        return data.reshape(nt, -1).T

    # Process variables
    U = load_and_flatten('U')
    V = load_and_flatten('V')
    Eta = load_and_flatten('Eta', is_2d=True)
    T = load_and_flatten('T')
    S = load_and_flatten('S')
    
    return U, V, Eta, T, S




def gather_surface(local_surf, rank, comm, root=0):
    """Gathers all rank-local surface chunks into a single global 2D array."""
    all_chunks = comm.gather(local_surf, root=root)
    if rank == root:
        # np.vstack combines (N_local, Time) into (N_global, Time)
        return np.vstack(all_chunks)
    return None



def extract_surface(rank_data, rank, size, nx=248, ny=248, nz=31):
    """
    Extracts the surface layer (k=0) by carefully matching the stack order.
    """
    i_start, i_end = slice_space(nx, rank, size)
    nx_local = i_end - i_start

    pts_per_surface_layer = ny * nx_local
    return rank_data[:pts_per_surface_layer, :]



def gather_and_save_surface(local_surf_dask, name, preproc_dir, nx, ny, rank, comm):
    # Compute LOCALLY first
    local_surf_np = local_surf_dask.compute() 
    
    # gather the NumPy chunks to Rank 0
    all_chunks = comm.gather(local_surf_np, root=0)

    if rank == 0:
        # Reshape each chunk to (ny, nx_local, ntime) 
        ntime = all_chunks[0].shape[1]
        size = comm.Get_size()
        
        # reconstruct the horizontal map by stacking along the i-dimension
        # Each chunk is (ny * nx_local, ntime) -> reshape to (ny, nx_local, ntime)
        reshaped_chunks = []
        for i, chunk in enumerate(all_chunks):
            nx_local = chunk.shape[0] // ny
            reshaped_chunks.append(chunk.reshape((ny, nx_local, ntime)))

        # Concatenate along the 'i' axis (axis 1)
        # Result shape: (ny, nx, ntime)
        global_surf_3d = np.concatenate(reshaped_chunks, axis=1)

        # Flatten to (ny*nx, ntime) to match your OpInf expectations
        global_surf_2d = global_surf_3d.reshape((-1, ntime))

        #  Save as .npy file
        save_path = f"{preproc_dir}{name}_transformed_surface.npy"
        np.save(save_path, global_surf_2d)
        print(f"Successfully saved {name} surface. Final shape: {global_surf_2d.shape}")





def transform_and_project_k(var, n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, nx=248, ny=248, k=0, anom=False):
    '''
    undo center/scale and project
    S_rom = [S_fom,train - center] Tr Q_ROM_ + center
    '''

    snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'
    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="netcdf4",  
                    decode_timedelta=False, decode_times=False, chunks={})

   # zero_zone_end = 55
    zero_zone_end = 0
    if var == 'Eta':
        S_FOM_train = ds[var].isel(time=slice(0, 360*n_year_train, n_days)).stack(space=('j', 'i')).T.values
     #   S_FOM_train = S_FOM_train.where(S_FOM_train.i >= zero_zone_end, 0)
     #   S_FOM_train = S_FOM_train.values

    elif var =='U':
        S_FOM_train = ds[var].isel(k=k, time=slice(0, 360*n_year_train, n_days)).stack(space=('j', 'i_g')).T.values
       # S_FOM_train = S_FOM_train.where(S_FOM_train.i_g >= zero_zone_end, 0)
    elif var =='V':
        S_FOM_train = ds[var].isel(k=k, time=slice(0, 360*n_year_train, n_days)).stack(space=('j_g', 'i')).T.values
    else:
        S_FOM_train = ds[var].isel(k=k, time=slice(0, 360*n_year_train, n_days)).stack(space=('j', 'i')).T.values

    center_da = xr.open_dataset(root_dir + 'preproc/' + f'center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc', engine='netcdf4')
    alphas = np.load(root_dir + f'/preproc/alpha_{center_opt}_{scale}_{n_year_train}yrs.npy', allow_pickle=True).item()
    alpha = alphas[var]
    if not anom: # add center back in 
        if center_opt == 'global_mean':
            center = center_da.center
            S_ROM = (S_FOM_train - center.values) @ Tr @ Q_ROM_ + center.values
        else:
            center = center_da.center[k*nx*ny:nx*ny*(k+1)]
            S_ROM = (S_FOM_train - center.values[:,np.newaxis]) @ Tr @ Q_ROM_ + center.values[:,np.newaxis]
    else: # don't add in center
        if center_opt == 'global_mean':
            center = center_da.center
            S_ROM = (S_FOM_train - center.values) @ Tr @ Q_ROM_ 
        else:
            center = center_da.center[k*nx*ny:nx*ny*(k+1)]
            S_ROM = (S_FOM_train - center.values[:,np.newaxis]) @ Tr @ Q_ROM_ #+ center.values[:,np.newaxis]
    
    S_ROM[abs(S_ROM) > 100] = np.nan
    S_ROM_3d = S_ROM.T.reshape(S_ROM.shape[1], ny, nx) 

    return S_ROM, S_ROM_3d






def transform_and_project_lon(var, n_year_train, n_days, center_opt, scale, Tr, Q_ROM_, root_dir, i=124, nx=248, ny=248, nz=31, anom=False):
    '''
    undo center/scale and project
    S_rom = [S_fom,train - center] Tr Q_ROM_ + center
    '''

    snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'
    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="netcdf4",  
                    decode_timedelta=False, decode_times=False, chunks={})

    if var =='V':
        S_FOM_train = ds[var].isel(i_g=i, time=slice(0, 360*n_year_train, n_days)).stack(space=('k', 'j_g')).T.values
    else:
        S_FOM_train = ds[var].isel(i=i, time=slice(0, 360*n_year_train, n_days)).stack(space=('k', 'j')).T.values

    center_da = xr.open_dataset(root_dir + 'preproc/' + f'center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc', engine='netcdf4')
    center = center_da.center.values.reshape(nz, ny, nx)
    center = center[:, :, i].ravel()

    if not anom: # add center back in 
        if center_opt == 'global_mean':
            S_ROM = (S_FOM_train - center) @ Tr @ Q_ROM_ + center
        else:
            S_ROM = (S_FOM_train - center[:,np.newaxis]) @ Tr @ Q_ROM_ + center[:,np.newaxis]
    else: # don't add in center
        if center_opt == 'global_mean':
            center = center_da.center
            S_ROM = (S_FOM_train - center) @ Tr @ Q_ROM_ 
        else:
            S_ROM = (S_FOM_train - center[:,np.newaxis]) @ Tr @ Q_ROM_ #+ center.values[:,np.newaxis]

    S_ROM[abs(S_ROM) > 100] = np.nan
    S_ROM_3d = S_ROM.T.reshape(S_ROM.shape[1], nz, ny) 
        
    return S_ROM, S_ROM_3d






### load var

def load_var_fom_k(var, n_year_train, n_year_predict, n_days, root_dir, center_opt, scale, nx=248, ny=248, k=0, anom=False):
    
    snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'
    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="netcdf4",  
                    decode_timedelta=False, decode_times=False, chunks={})

    if var == 'Eta':
        var_FOM = ds[var].isel(time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('j', 'i')).T
    elif var == 'U':
        var_FOM = ds[var].isel(k=k,time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('j', 'i_g')).T
    elif var == 'V':
        var_FOM = ds[var].isel(k=k,time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('j_g', 'i')).T
    else:
        var_FOM = ds[var].isel(k=k, time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('j', 'i')).T
    
    if anom:
        center_da = xr.open_dataset(root_dir + 'preproc/' + f'center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc', engine='netcdf4')
        alphas = np.load(root_dir + f'/preproc/alpha_{center_opt}_{scale}_{n_year_train}yrs.npy', allow_pickle=True).item()
        alpha = alphas[var] 
        if center_opt == 'global_mean':
            center = center_da.center
            var_FOM = (var_FOM - center.values) #/ alpha
        else:
            center = center_da.center[k*nx*ny:nx*ny*(k+1)]
            var_FOM = (var_FOM - center.values[:,np.newaxis]) #/ alpha

   # zero_zone_end = 55
    zero_zone_end = 0
    
    var_FOM_3d = var_FOM.unstack('space')
   # var_FOM_3d = var_FOM_3d.where(var_FOM_3d.i >= zero_zone_end, 0)

    return var_FOM, var_FOM_3d


def load_var_fom_lon(var, n_year_train, n_year_predict, n_days, root_dir, center_opt, scale, nx=248, ny=248, nz=31, i=124,anom=False):
    
    snapshot_dir = '/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/'
    ds = xr.open_dataset(snapshot_dir + 'states_20yrs.nc', engine="netcdf4",  
                    decode_timedelta=False, decode_times=False, chunks={})

    if var == 'U':
        var_FOM = ds[var].isel(i_g=i,time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('k', 'j')).T
    else:
        var_FOM = ds[var].isel(i=i,time=slice(0, 360*(n_year_train + n_year_predict), n_days)).stack(space=('k', 'j')).T
  
    center_da = xr.open_dataset(root_dir + 'preproc/' + f'center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc', engine='netcdf4')
    center = center_da.center.values.reshape(nz, ny, nx)
    center = center[:, :, i].ravel()

    if anom:
        center_da = xr.open_dataset(root_dir + 'preproc/' + f'center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc', engine='netcdf4')
        if center_opt == 'global_mean':
            center = center_da.center
            var_FOM = (var_FOM - center.values) #/ alpha
        else:
            var_FOM = (var_FOM - center[:,np.newaxis]) #/ alpha

   # zero_zone_end = 55
    zero_zone_end = 0
    
    var_FOM_3d = var_FOM.unstack('space')
   # var_FOM_3d = var_FOM_3d.where(var_FOM_3d.i >= zero_zone_end, 0)

    return var_FOM, var_FOM_3d


def read_grid(comm, grid_path='/scratch/shoshi/soma4/grid/'):
    """
    Reads MITgcm grid files on Rank 0 and broadcasts the 
    essential arrays to all other ranks to avoid I/O overhead.
    """

    rank = comm.Get_rank()
    grid_vars = {}
    
   # keys = ['hFacW', 'DRF', 'DYG', 'DXG', 'RA', 'YC', 'XC', 'XG', 'YG', 'Z']
    keys = ['hFacW', 'DRF', 'DYG', 'DXG', 'YC', 'XC', 'XG', 'YG']

    for k in keys:
        data = None
        if rank == 0:
            try:
                data = rdmds(grid_path + k)
                # If rdmds adds a singleton dimension (e.g., 1, 31, 248, 248), squeeze it
                data = np.squeeze(data)
            except Exception as e:
                print(f"Rank 0: Could not read {k} using rdmds: {e}")

        # Broadcast the NumPy array from Rank 0 to all other ranks
        # This is a collective operation; all ranks must call it
        grid_vars[k] = comm.bcast(data, root=0)

        # Clear local variable and force garbage collection to keep RAM low
        del data
        gc.collect()

    return grid_vars









def compute_barotropic_streamfunction(
    comm, grid, U_rank, centerU, svd, Q_ROM_val,
    nx, ny, nz, root_dir, center_opt, scale_type, n_year_train
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast Q_ROM to all ranks 
    Q_ROM_val = comm.bcast(Q_ROM_val, root=0)

    # For memory savings
    U_rank = np.ascontiguousarray(U_rank)
    svd.Tr_global = np.ascontiguousarray(svd.Tr_global)
    Q_ROM_val = np.ascontiguousarray(Q_ROM_val)

    U_rank = U_rank.astype(np.float32, copy=False)
    svd.Tr_global = svd.Tr_global.astype(np.float32, copy=False)
    Q_ROM_val = Q_ROM_val.astype(np.float32, copy=False)

    # Local reconstruction of U_ROM
    Uc = U_rank - centerU[:, None]
    V_u = Uc @ svd.Tr_global
    U_ROM_rank = V_u @ Q_ROM_val
    U_ROM_rank += centerU[:, None]

    # Local domain split of nx
    i_start, i_end = slice_space(nx, rank, size)
    nx_local = i_end - i_start
    nt = U_ROM_rank.shape[1]

    # Reshape to 4d
    U_ROM_rank_3d = U_ROM_rank.reshape(nz, ny, nx_local, nt)

    # --- 5. Depth integration ---
    hFacW = grid['hFacW'][:, :, i_start:i_end]
    drF = grid['DRF'][:, None, None]

    weights = hFacW * drF
    u_bt_rank = np.sum(U_ROM_rank_3d * weights[..., None], axis=0)

    del U_ROM_rank, U_ROM_rank_3d
    gc.collect()

    # --- 6. Gather ---
    all_strips = comm.gather(u_bt_rank, root=0)

    if rank == 0:
        # Concatenate along X (local x blocks)
        u_bt = np.concatenate(all_strips, axis=1)

        u_bt_da = xr.DataArray(
            u_bt.transpose(2, 0, 1),
            dims=('time', 'YC', 'XG'),
            coords={
                'time': np.arange(nt),
                'YC': grid['YC'][:, 0],
                'XG': grid['XG'][0, :]
            },
            name='u_bt'
        )

        path = f"{root_dir}/u_bt_rom_{center_opt}_{scale_type}_r{svd.r}_{n_year_train}trainingyrs.nc"
        u_bt_da.to_netcdf(path)

        # --- 7. Streamfunction ---
        dyG = grid['DYG'][:, :, None]
        psi = np.cumsum(-u_bt * dyG, axis=0) / 1e6

        psi_da = xr.DataArray(
            psi.transpose(2, 0, 1),
            dims=('time', 'YC', 'XG'),
            coords=u_bt_da.coords,
            name='psi'
        )

        psi_path = f"{root_dir}/psi_rom_{center_opt}_{scale_type}_r{svd.r}_{n_year_train}trainingyrs.nc"
        psi_da.to_netcdf(psi_path)

        print("saved barotropic fields", flush=True)

    # --- 8. HARD synchronization point ---
    comm.Barrier()
