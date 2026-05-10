
from mpi4py import MPI
import numpy as np
import xarray as xr
from utils import gather_distributed_array, slice_space
from scipy.signal import convolve2d
import dask.array as da
import os


def ensure_time_chunks(Q_rank, target_time_chunk=360):
    """Rechunk snapshot dask arrays to keep time chunks manageable."""
    if not hasattr(Q_rank, "rechunk"):
        return Q_rank

    try:
        n_time = int(Q_rank.shape[1]) if Q_rank.shape[1] is not None else target_time_chunk
    except Exception:
        n_time = target_time_chunk

    try:
        return Q_rank.rechunk({1: min(target_time_chunk, n_time)})
    except Exception:
        return Q_rank


def scalefactor(Q_rank, scale_type='maxabs', comm=None):
    """
    Computes scaling factors based on Q_shifted (space, time).
    Returns a scalar for global types, or a 1D array for 'var'.
    """
    if Q_rank.size == 0:
        return 1.0 # Avoid division by zero

    if scale_type == 'var':
        # Local variance per spatial point (No MPI needed)
        # Result: (n_spatial_points_on_rank,)
        alpha = da.std(Q_rank, axis=1).compute()
        alpha = np.where(alpha == 0, 1.0, alpha)
        return alpha

    elif scale_type == 'maxabs':
        local_val = da.max(da.abs(Q_rank)).compute()
        if comm:
            return comm.allreduce(float(local_val), op=MPI.MAX)
        return float(local_val)
    
    elif scale_type == 'maxnorm':
        # Frobenius norm / energy per snapshot
        local_val = da.sum(Q_rank**2, axis=0).compute()
        if comm:
            global_sq_sum = np.zeros_like(local_val)
            comm.Allreduce(local_val, global_sq_sum, op=MPI.SUM)
            return float(np.sqrt(global_sq_sum).max())
        return float(np.sqrt(local_val).max())
        
    elif scale_type == 'sigma':
        # Global standard deviation
        local_val = da.std(Q_rank).compute()
        if comm:
            # Note: This is an approximation of global sigma. 
            # For exact global sigma, one would need the Law of Total Variance.
            return comm.allreduce(float(local_val), op=MPI.MAX)
        return float(local_val)
        
    else:
        raise ValueError(f"Unknown scale_type {scale_type}")



def compute_climatology(Q_rank, cycle_len, comm):
    """
    Faster climatology: Reshapes to (space, years, 360) and means over years.
    """
    n_space, n_time = Q_rank.shape
    n_years = n_time // cycle_len
    
    # Trim Q_rank to be exactly divisible by 360 if it isn't already
  #  Q_trimmed = Q_rank[:, :n_years * cycle_len]
    
    # Reshape to (space, years, 360)
  #  Q_reshaped = Q_trimmed.reshape(n_space, n_years, cycle_len)
    Q_reshaped = Q_rank.reshape(n_space, n_years, cycle_len)
    
    # Compute mean along the 'years' 
    seasonal_mean_local = Q_reshaped.mean(axis=1).compute() # Result: (n_space, 360)
    
    if comm:
        comm.Allreduce(MPI.IN_PLACE, seasonal_mean_local, op=MPI.SUM)
    
    return seasonal_mean_local


def shiftscale(Q_rank, n_year_train=None, comm=None, center_type='',
               scale_type='maxabs', save_file=None, nx=248, ny=248, nz=31):
    """
    Q_rank: dask array (space, time)
    """
    if comm is None and hasattr(n_year_train, "Get_rank"):
        comm = n_year_train
        n_year_train = None

    n_space, n_time = Q_rank.shape
    rank = comm.Get_rank() if comm else 0
    
    center_ref = None

    # Keep time chunks reasonable before any reductions.
    if hasattr(Q_rank, 'chunks'):
        Q_rank = ensure_time_chunks(Q_rank, target_time_chunk=min(360, n_time))

    # check if center file already exists:
    file_exists = False
    if rank == 0:
        if save_file and os.path.exists(save_file):
            file_exists = True

    if comm:
        file_exists = comm.bcast(file_exists, root=0)

    # if it exists, read it in!
    if file_exists:
        if rank == 0:
            print(f"Loading center_ref from {save_file}", flush=True)
        try:
            # Only rank 0 reads from disk, then broadcasts
            nz_saved = None
            
            if rank == 0:
                with xr.open_dataset(save_file, engine="h5netcdf") as ds:
                    nz_saved = int(ds.attrs.get('nz', 1))
                    if center_type == 'global_mean':
                        global_center = np.array([ds['center'].values])
                    else:
                        # Load the global center (all spatial points)
                        global_center = ds['center'].values
                    
                    if global_center.ndim > 1 and global_center.shape[1] == 1:
                        global_center = global_center[:, 0]
                
                print(f"Successfully loaded center_ref from {save_file}")
            
            # Broadcast global center to all ranks
            if comm:
                global_center = comm.bcast(global_center if rank == 0 else None, root=0)
                nz_saved = comm.bcast(nz_saved, root=0)
            
            # Now each rank slices its local portion
            if center_type != 'global_mean' and global_center is not None:
                i_start, i_end = slice_space(nx, rank, comm.Get_size() if comm else 1)
                if global_center.ndim == 1:
                    center_ref = global_center.reshape((nz_saved, ny, nx))[:, :, i_start:i_end].ravel()
                else:
                    n_cols = global_center.shape[1]
                    center_ref = (
                        global_center.reshape((nz_saved, ny, nx, n_cols))[:, :, i_start:i_end, :]
                        .reshape((-1, n_cols))
                    )
            else:
                center_ref = global_center
            
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load {save_file}: {e}. Computing from scratch.")
            center_ref = None

    # ---------- COMPUTE CENTER ----------
    if center_ref is None:
        if center_type == 'mean':
            center_ref = da.mean(Q_rank, axis=1).compute()  # (space,)
        
        elif center_type == 'global_mean': #single mean across space and time
            # Identify ocean points only
            is_wet = da.any(Q_rank != 0, axis=1)
            wet_data = Q_rank[is_wet, :]
            
            val = wet_data.mean().compute()
            if comm:
                val = comm.allreduce(val, op=MPI.SUM) / comm.Get_size()
            center_ref = np.array([val])

        elif center_type == 'IC':
            center_ref = Q_rank[:, 0].compute()  # (space,)

        elif center_type == 'seasonal':
            center_ref = compute_climatology(Q_rank, 360, comm)  # (space,360)

        elif center_type == 'monthly':
            center_ref = compute_climatology(Q_rank, 12, comm)  # (space,12)

        else:
            print("No centering specified, not centering")
            center_ref = 0

        # Save reference (SAFE)
        if save_file:
            save_center_ref(center_ref, save_file, comm, center_type, nx=nx, ny=ny, nz=nz)

    # ---------- SHIFT ----------
    if center_type == 'global_mean':
        Q_shifted = Q_rank - center_ref[0]
    elif center_type in ('mean', 'IC'):
        Q_shifted = Q_rank - center_ref[:, None]
    elif center_type == '':
        Q_shifted = Q_rank
    else:
        cycle_len = center_ref.shape[1]
        t_idx = da.arange(n_time, chunks=Q_rank.chunks[1]) % cycle_len
        Q_shifted = Q_rank - center_ref[:, t_idx]

    # ---------- SCALE BASED ON SHIFTED DATA ----------
    if scale_type is not None:
        alpha = scalefactor(Q_shifted, scale_type, comm)
        if np.ndim(alpha) == 1:
            alpha = np.where(alpha == 0, 1.0, alpha)
            alpha = alpha[:, np.newaxis]
    else:
        alpha = 1

    # ---------- FINAL TRANSFORM ----------
    Q_rank = Q_shifted / alpha
    alpha2 = scalefactor(Q_rank, 'maxabs', comm)
    Q_rank = Q_rank / alpha2

    alpha_total = alpha * alpha2

    return Q_rank, center_ref, alpha_total



def save_center_ref(local_ref, save_path, comm, center_type, nx=248, ny=248, nz=31):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. SCALAR CASE: Global Mean (Scalar)
    if center_type == 'global_mean':
        # Since every rank has the same scalar, only rank 0 needs to act
        if rank == 0:
            # Ensure it's a simple float/array
            val = float(local_ref[0] if hasattr(local_ref, "__len__") else local_ref)
            ds_save = xr.Dataset(
                {"center": ([], val)}, # No dimensions (scalar)
                attrs={"center_type": center_type, "nz": nz}
            )
            ds_save.to_netcdf(save_path, engine="h5netcdf")
            print(f"Global mean scalar saved to {save_path}", flush=True)
        return # All ranks exit here

    # 2. SPATIAL CASE: 
    local_data = np.ascontiguousarray(np.array(local_ref), dtype=np.float64)
    all_pieces = comm.gather(local_data, root=0)

    if rank == 0:
        i_start_0, i_end_0 = slice_space(nx, 0, size)
        ni_0 = i_end_0 - i_start_0
        
        # Calculate depth (nz=1 for Eta, nz=31 for T/S/U/V)
        # Note: If local_ref is 1D, shape[0] is total spatial points in that slice
        actual_nz = int(all_pieces[0].shape[0] / (ny * ni_0))
        
        total_space = actual_nz * ny * nx
        if local_ref.ndim == 1:
            global_ref = np.zeros(total_space)
        else:
            n_cols = local_ref.shape[1]
            global_ref = np.zeros((total_space, n_cols))

        for r in range(size):
            i_start, i_end = slice_space(nx, r, size)
            ni_local = i_end - i_start
            
            if local_ref.ndim == 1:
                piece = all_pieces[r].reshape((actual_nz, ny, ni_local))
                target = global_ref.reshape((actual_nz, ny, nx))
                target[:, :, i_start:i_end] = piece
            else:
                n_cols = local_ref.shape[1]
                piece = all_pieces[r].reshape((actual_nz, ny, ni_local, n_cols))
                target = global_ref.reshape((actual_nz, ny, nx, n_cols))
                target[:, :, i_start:i_end, :] = piece

        dims = ['space']
        if global_ref.ndim > 1:
            dims.append('ref_index')

        ds_save = xr.Dataset(
            {"center": (dims, global_ref)}, 
            attrs={"center_type": center_type, "nz": actual_nz}
        )
        ds_save.to_netcdf(save_path, engine="h5netcdf")
        print(f"Global {center_type} spatial map saved to {save_path} (nz={actual_nz})", flush=True)
