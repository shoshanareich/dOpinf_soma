from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np


class SVDDecomposition:
    """
    Parallel SVD computation using the global Gram matrix
    """

    def __init__(self, Q_rank, comm=MPI.COMM_WORLD, target_ret_energy=None, r=None):
        """
        Parameters
        ----------
        Q_rank : np.ndarray
            Local block of the snapshot matrix (n_local x nt)
        comm : MPI communicator
        target_ret_energy : float
            Fraction of cumulative eigenvalue energy to retain
        """
        self.Q_rank = Q_rank
        self.comm = comm
        self.target_ret_energy = target_ret_energy
        self.r = r

        # Outputs
        self.D_global = None
        self.eigs = None
        self.eigv = None
        #self.r = None
        self.Tr_global = None
        self.Qhat_global = None
        self.ret_energy = None

    # -------------------------------------------------
    def compute(self):
        """Compute global SVD quantities and store results as attributes."""
        comm = self.comm
        rank = comm.Get_rank()
        if rank == 0:
            print("[SVD] Starting local Gram matrix assembly", flush=True)

        def _block_gram(block):
            """
            Compute the local Gram contribution for one row block.
            If dask, break into 4,096 × n_t slices for memory efficiency
            """
            if hasattr(block, "chunks"):
                try:
                    block = block.rechunk({0: min(4096, int(block.shape[0]))})
                except Exception:
                    pass

                D_block = None
                for i in range(block.numblocks[0]):
                    row_chunk = block.blocks[i, :].compute()
                    gram = row_chunk.T @ row_chunk
                    if D_block is None:
                        D_block = gram
                    else:
                        D_block += gram
                return D_block

            return block.T @ block

        if isinstance(self.Q_rank, (list, tuple)):
            D_rank = None
            for block in self.Q_rank:
                D_block = _block_gram(block)
                if D_rank is None:
                    D_rank = D_block
                else:
                    D_rank += D_block
        else:
            D_rank = _block_gram(self.Q_rank)

        # Global reduction
        if rank == 0:
            print("[SVD] Local Gram complete. Starting MPI allreduce", flush=True)
        if rank == 0:
            self.D_global = np.empty_like(D_rank)
        else:
            self.D_global = None

        # Use low-level Allreduce (buffer-based)
        self.D_global = comm.allreduce(D_rank, op=MPI.SUM)
        del D_rank

        if rank ==0:
            print("[SVD] Allreduce complete. Starting eigendecomposition", flush=True)

            # Eigendecomposition
            eigs, eigv = np.linalg.eigh(self.D_global)

            # Sort descending
            idx = np.argsort(eigs)[::-1]
            self.eigs = eigs[idx]
            self.eigv = eigv[:, idx]

            # Cumulative energy
            self.ret_energy = np.cumsum(self.eigs) / np.sum(self.eigs)

            # Rank r that satisfies target energy or desired r
            if self.target_ret_energy is not None:
                self.r = np.argmax(self.ret_energy >= self.target_ret_energy)
            elif self.r is not None:
                self.r = self.r
            else:
                self.r = len(self.eigs)
                print('Warning: Neither r nor target_ret_energy specified. Using full rank.')

            # Tr matrix and reduced coordinates
            self.Tr_global = self.eigv[:, :self.r] @ np.diag(self.eigs[:self.r]**(-0.5))
            self.Qhat_global = np.diag(np.sqrt(self.eigs[:self.r])) @ self.eigv[:, :self.r].T
            print(f"[SVD] Eigendecomposition complete. Selected r={self.r}", flush=True)
        else:
            self.eigs = None
            self.eigv = None
            self.ret_energy = None
            self.Tr_global = None
            self.Qhat_global = None

        # Broadcast results to all ranks
        self.r = comm.bcast(self.r, root=0)
        self.Tr_global = comm.bcast(self.Tr_global, root=0)
        self.Qhat_global = comm.bcast(self.Qhat_global, root=0)
        if rank == 0:
            print("[SVD] Broadcast complete", flush=True)

        return self

    # -------------------------------------------------
    def plot_decay(self, save_path=None):
        """Plot SVD eigenvalue decay and retained energy."""
        if self.eigs is None:
            raise RuntimeError("compute() must be called before plot_decay().")

        eigs = self.eigs
        r = self.r
        target_ret_energy = self.target_ret_energy
        ret_energy = np.cumsum(eigs) / np.sum(eigs)
        no_svals_global = range(1, len(eigs) + 1)

        # styling 
        charcoal = [0.1, 0.1, 0.1]
        color1 = '#D55E00'
        color2 = '#0072B2'

        plt.rcParams['lines.linewidth'] = 0
        plt.rc("figure", dpi=400)
        plt.rc("font", family="serif")
        plt.rc("legend", edgecolor='none')
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.rcParams.update({'font.size': 10})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: singular value decay
        ax = axes[0]
        ax.semilogy(no_svals_global, np.sqrt(eigs)[:len(eigs)]/np.sqrt(eigs[0]), linestyle='-', lw=0.75, color=color1)
        ax.axvline(x=r,  linestyle='--', lw=0.5, color=charcoal)
        ax.text(r+5, 8e-4, f"r = {r}", color=charcoal, va='center')
        ax.set_xlabel('index')
        ax.set_ylabel('normalized singular values')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Right: retained energy
        ax = axes[1]
        ax.plot(range(len(eigs)), self.ret_energy[:len(eigs)], linestyle='-', lw=0.75, color=color1)
        ax.set_xlabel('reduced dimension')
        ax.set_ylabel('retained energy')	
        ax.plot([r, r], [0, ret_energy[r]], linestyle='--', lw=0.5, color=charcoal)
        ax.plot([0, r], [ret_energy[r], ret_energy[r]], linestyle='--', lw=0.5, color=charcoal)
        ax.set_xlim(-1, 200)
        ax.set_ylim(0.1, 1.05)
        ax.text(r+1, 0.15*ret_energy[r], f"r = {r}", color=charcoal, va='center')
        ax.text(0.4*r, ret_energy[r]+0.02, f"{self.ret_energy[r]:.0%}", color=charcoal, ha='center')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        if save_path:
            fig.savefig(save_path, dpi=300)

        return fig, axes

    # -------------------------------------------------
    def save(self, outdir, prefix="svd"):
        """Save Tr_global, Qhat_global, eigs, etc. to disk."""
        if self.Tr_global is None:
            raise RuntimeError("compute() must be called before save().")
        
        np.save(f"{outdir}/Tr.npy", self.Tr_global)
        np.save(f"{outdir}/Qhat.npy", self.Qhat_global)
        np.save(f"{outdir}/eigs.npy", self.eigs)

        # eigs_prefix = prefix.rsplit('_', 1)[0]
        # np.save(f"{outdir}/save_roms/Tr.npy", self.Tr_global)
        # np.save(f"{outdir}/save_roms/Qhat.npy", self.Qhat_global)
        # np.save(f"{outdir}/svals/eigs.npy", self.eigs)

      #  np.save(f"{outdir}/{prefix}_eigv.npy", self.eigv)
      #  np.save(f"{outdir}/{prefix}_r.npy", np.array([self.r]))

