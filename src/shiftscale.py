
from mpi4py import MPI
import numpy as np
import xarray as xr
from utils import gather_distributed_array, slice_space
from scipy.signal import convolve2d
import dask.array as da
import os
import time


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
        # Local variance per spatial point 
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


def alpha2_from_row_extrema(q_min, q_max, center_ref, alpha, comm=None):
    """
    Compute max(abs((Q - center) / alpha)) from per-row extrema.
    Valid when center_ref and alpha are constant over time for each row.
    """
    alpha_vec = np.asarray(alpha).reshape(-1)
    if np.ndim(center_ref) == 0:
        center_vec = float(center_ref)
    else:
        center_arr = np.asarray(center_ref).reshape(-1)
        center_vec = center_arr[0] if center_arr.size == 1 else center_arr

    local_vals = np.maximum(
        np.abs(np.asarray(q_min) - center_vec),
        np.abs(np.asarray(q_max) - center_vec),
    ) / alpha_vec
    local_val = float(np.max(local_vals)) if local_vals.size else 0.0
    if comm:
        return comm.allreduce(local_val, op=MPI.MAX)
    return local_val



def _compute(value):
    return value.compute() if hasattr(value, "compute") else value


def _snapshots_per_year(n_days):
    if n_days is None:
        n_days = 1
    n_days = int(n_days)
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    if 360 % n_days != 0:
        raise ValueError(f"n_days={n_days} does not evenly divide the 360-day model year")
    return 360 // n_days


def _samples_per_month(n_days):
    n_days = int(n_days)
    if 30 % n_days != 0:
        raise ValueError(f"n_days={n_days} does not evenly divide the 30-day model month")
    return 30 // n_days


def climatology_time_indices(n_time, center_type, n_days=1):
    """Return the climatology-column index for each snapshot time."""
    t = np.arange(n_time)
    if center_type == 'seasonal':
        return t % _snapshots_per_year(n_days)
    if center_type == 'monthly':
        return ((t * int(n_days)) % 360) // 30
    raise ValueError(f"Unknown climatology center_type {center_type}")


def center_for_time(center_ref, n_time, center_type, n_days=1, chunks=None):
    """Broadcast a climatology center to the full snapshot time axis."""
    center_arr = np.asarray(center_ref)
    t_idx = climatology_time_indices(n_time, center_type, n_days)
    if chunks is None:
        return center_arr[:, t_idx]

    row_chunks = chunks[0] if isinstance(chunks, tuple) else "auto"
    center_dask = da.from_array(center_arr, chunks=(row_chunks, center_arr.shape[1]))
    return center_dask[:, t_idx]


def compute_climatology(Q_rank, center_type, n_days=1):
    """
    Compute a sampled 360-day climatology for spatial rows local to this rank.

    seasonal -> one mean for each sampled day of the 360-day year
    monthly  -> one mean for each 30-day month
    """
    n_space, n_time = Q_rank.shape
    snapshots_per_year = _snapshots_per_year(n_days)
    if n_time % snapshots_per_year != 0:
        raise ValueError(
            f"n_time={n_time} is not an integer number of sampled 360-day years "
            f"for n_days={n_days}"
        )

    n_cycles = n_time // snapshots_per_year
    Q_reshaped = Q_rank.reshape(n_space, n_cycles, snapshots_per_year)

    if center_type == 'seasonal':
        return _compute(Q_reshaped.mean(axis=1))

    if center_type == 'monthly':
        samples_per_month = _samples_per_month(n_days)
        Q_monthly = Q_reshaped.reshape(n_space, n_cycles, 12, samples_per_month)
        return _compute(Q_monthly.mean(axis=(1, 3)))

    raise ValueError(f"Unknown climatology center_type {center_type}")


def shiftscale(Q_rank, n_year_train=None, comm=None, center_type='', n_days=1,
               scale_type='maxabs', save_file=None, nx=248, ny=248, nz=31):
    """
    Q_rank: dask array (space, time)
    n_days: snapshot spacing in model days. The calendar is 360 days/year,
        with 12 months of 30 days.
    """
    if comm is None and hasattr(n_year_train, "Get_rank"):
        comm = n_year_train
        n_year_train = None

    n_space, n_time = Q_rank.shape
    rank = comm.Get_rank() if comm else 0
    center_is_time_constant = center_type in ('mean', 'IC', 'global_mean', '')

    def log_elapsed(message, start_time):
        if rank == 0:
            elapsed = time.perf_counter() - start_time
            print(f"[shiftscale] {message} | elapsed {elapsed / 60:.2f} min ({elapsed:.1f} s)", flush=True)
    
    center_ref = None
    alpha = None
    q_min = None
    q_max = None

    # Keep time chunks reasonable before any reductions.
    if hasattr(Q_rank, 'chunks'):
        Q_rank = ensure_time_chunks(Q_rank, target_time_chunk=min(360, n_time))

    save_center = save_file and center_type != 'monthly'
    if save_file and center_type == 'monthly' and rank == 0:
        print("[shiftscale] Skipping monthly center cache load/save", flush=True)

    # check if center file already exists:
    file_exists = False
    if rank == 0:
        if save_center and os.path.exists(save_file):
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
                    if center_type in ('seasonal', 'monthly'):
                        saved_n_days = ds.attrs.get('n_days')
                        if saved_n_days is None or int(saved_n_days) != int(n_days):
                            raise ValueError(
                                f"cached {center_type} center was not created for n_days={n_days}"
                            )
                    nz_saved = int(ds.attrs.get('nz', 1))
                    if center_type == 'global_mean':
                        global_center = np.array([ds['center'].values])
                    else:
                        # Load the global center (all spatial points)
                        global_center = ds['center'].values
                    
                    if center_type in ('mean', 'IC') and global_center.ndim > 1 and global_center.shape[1] == 1:
                        global_center = global_center[:, 0]
                
                print(f"Successfully loaded center_ref from {save_file}", flush=True)
            
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
        center_start = time.perf_counter()
        if center_type == 'mean':
            if scale_type == 'var':
                # For mean-centering, std(Q - mean(Q)) == std(Q). Computing both
                # together lets Dask share the same source reads/chunk tasks.
                center_ref, alpha, q_min, q_max = da.compute(
                    da.mean(Q_rank, axis=1),
                    da.std(Q_rank, axis=1),
                    da.min(Q_rank, axis=1),
                    da.max(Q_rank, axis=1),
                )
                alpha = np.where(alpha == 0, 1.0, alpha)
            else:
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
            if scale_type == 'var':
                center_ref, alpha, q_min, q_max = da.compute(
                    Q_rank[:, 0],
                    da.std(Q_rank, axis=1),
                    da.min(Q_rank, axis=1),
                    da.max(Q_rank, axis=1),
                )
                alpha = np.where(alpha == 0, 1.0, alpha)
            else:
                center_ref = Q_rank[:, 0].compute()  # (space,)

        elif center_type == 'seasonal':
            center_ref = compute_climatology(Q_rank, 'seasonal', n_days=n_days)

        elif center_type == 'monthly':
            center_ref = compute_climatology(Q_rank, 'monthly', n_days=n_days)

        else:
            print("No centering specified, not centering")
            center_ref = 0
            if center_type == '' and scale_type == 'var':
                alpha, q_min, q_max = da.compute(
                    da.std(Q_rank, axis=1),
                    da.min(Q_rank, axis=1),
                    da.max(Q_rank, axis=1),
                )
                alpha = np.where(alpha == 0, 1.0, alpha)
        if comm:
            comm.Barrier()
        log_elapsed(f"Compute center complete ({center_type})", center_start)

        # Save reference (SAFE)
        if save_center:
            save_center_start = time.perf_counter()
            if rank == 0:
                print(f"[shiftscale] Save center start ({center_type})", flush=True)
            save_center_ref(center_ref, save_file, comm, center_type, nx=nx, ny=ny, nz=nz, n_days=n_days)
            log_elapsed(f"Save center complete ({center_type})", save_center_start)

    # Center files are stored as float64, but the snapshot data are usually
    # float32. Cast after cache loading so shifting does not promote the whole
    # SVD input to float64 and double Gram-assembly memory.
    q_dtype = np.dtype(getattr(Q_rank, "dtype", np.float64))
    if np.issubdtype(q_dtype, np.floating) and center_type != '':
        center_ref = np.asarray(center_ref, dtype=q_dtype)

    # ---------- SHIFT ----------
    if center_type == 'global_mean':
        Q_shifted = Q_rank - center_ref[0]
    elif center_type in ('mean', 'IC'):
        Q_shifted = Q_rank - center_ref[:, None]
    elif center_type == '':
        Q_shifted = Q_rank
    else:
        chunks = getattr(Q_rank, "chunks", None)
        Q_shifted = Q_rank - center_for_time(center_ref, n_time, center_type, n_days=n_days, chunks=chunks)

    # ---------- SCALE BASED ON SHIFTED DATA ----------
    scale_start = time.perf_counter()
    if scale_type is not None:
        if alpha is None:
            if scale_type == 'var' and center_is_time_constant:
                alpha, q_min, q_max = da.compute(
                    da.std(Q_rank, axis=1),
                    da.min(Q_rank, axis=1),
                    da.max(Q_rank, axis=1),
                )
                alpha = np.where(alpha == 0, 1.0, alpha)
            else:
                alpha = scalefactor(Q_shifted, scale_type, comm)
        if np.ndim(alpha) == 1:
            alpha = np.where(alpha == 0, 1.0, alpha)
            alpha = alpha[:, np.newaxis]
    else:
        alpha = 1
    log_elapsed(f"Scale factor complete ({scale_type})", scale_start)

    # ---------- FINAL TRANSFORM ----------
    Q_rank = Q_shifted / alpha
    alpha2_start = time.perf_counter()
    if scale_type == 'var' and center_is_time_constant and q_min is not None and q_max is not None:
        alpha2 = alpha2_from_row_extrema(q_min, q_max, center_ref, alpha, comm)
    else:
        alpha2 = scalefactor(Q_rank, 'maxabs', comm)
    log_elapsed("Alpha2 maxabs complete", alpha2_start)
    Q_rank = Q_rank / alpha2

    alpha_total = alpha * alpha2

    return Q_rank, center_ref, alpha_total



def save_center_ref(local_ref, save_path, comm, center_type, nx=248, ny=248, nz=31, n_days=1):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. SCALAR CASE: Global Mean (Scalar)
    if center_type == 'global_mean':
        # Since every rank has the same scalar, only rank 0 needs to act
        if rank == 0:
            # Ensure it's a simple float/array
            val = float(local_ref[0] if hasattr(local_ref, "__len__") else local_ref)
            attrs = {
                "center_type": center_type,
                "nz": nz,
                "n_days": int(n_days),
                "calendar_days_per_year": 360,
                "calendar_days_per_month": 30,
            }
            ds_save = xr.Dataset(
                {"center": ([], val)}, # No dimensions (scalar)
                attrs=attrs,
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

        attrs = {
            "center_type": center_type,
            "nz": actual_nz,
            "n_days": int(n_days),
            "calendar_days_per_year": 360,
            "calendar_days_per_month": 30,
        }
        ds_save = xr.Dataset(
            {"center": (dims, global_ref)}, 
            attrs=attrs,
        )
        ds_save.to_netcdf(save_path, engine="h5netcdf")
        print(f"Global {center_type} spatial map saved to {save_path} (nz={actual_nz})", flush=True)
