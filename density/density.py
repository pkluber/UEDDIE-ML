import numpy as np

# Adapted from MLCF: https://github.com/semodi/mlcf/blob/master/mlc_func/elf/density.py
class Density():
    """ Class defining the density on a real space grid

    Parameters
    ----------

        rho: np.ndarray
            3-dim real space density.
        unitcell: np.ndarray (3,3)
         unitcell in Angstrom.
        grid: np.ndarray (3,)
         grid points.

    """
    def __init__(self, rho, unitcell, grid, origin):
        if rho.ndim != 3:
            raise Exception('rho.ndim = {}, expected: 3'.format(rho.ndim))
        if unitcell.shape != (3,3):
            raise Exception('unitcell.shape = {}, expected: (3,3)'.format(unitcell.shape))
        if grid.shape != (3,):
            raise Exception('grid.shape = {}, expected: (3,)'.format(grid.shape))
        self.rho = rho
        self.unitcell = unitcell
        self.grid = grid
        self.origin = origin

    def mesh_3d(self, rmax: np.ndarray | None = None, scaled: bool = False):
        """
        Returns a 3d mesh 

        Parameters
        ----------

        rmax: list
            upper cutoff in every euclidean direction.
        scaled: boolean
            scale the meshes with unitcell size?

        Returns
        -------

        X, Y, Z: tuple of np.ndarray
            defines mesh in real space
        """ 
        U = np.array(self.unitcell)  # Matrix to go from real space to mesh coordinates
        for i in range(3):
            U[i,:] = U[i,:] / self.grid[i]

        if rmax is not None:
            # Convert rmax to grid coordinates
            rmax = np.rint(np.dot(U, rmax)).astype(int)
        else: 
            # If no rmax, take the entire grid 
            rmax = np.floor(self.grid / 2).astype(int)  

        x, y, z = [list(range(-rmax[i], rmax[i]+1)) for i in range(3)]
        
        Xm, Ym, Zm = np.meshgrid(x, y, z, indexing='ij')
        
        if not scaled:
            return Xm, Ym, Zm

        Rm = np.concatenate([Xm.reshape(*Xm.shape,1),
                             Ym.reshape(*Xm.shape,1),
                             Zm.reshape(*Xm.shape,1)], axis = 3)
        
        R = np.einsum('ij,klmj -> iklm', U.T , Rm)
        X = R[0,:,:,:] + self.origin
        Y = R[1,:,:,:] + self.origin
        Z = R[2,:,:,:] + self.origin
        
        return X, Y, Z 
