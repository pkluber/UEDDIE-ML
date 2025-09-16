import numpy as np
from typing import Tuple

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
        
        self.U = np.array(self.unitcell)  # Matrix to go from mesh coordinates to real space
        for i in range(3):
            self.U[i,:] = self.U[i,:] / self.grid[i]
    
    def mesh_3d(self, scaled: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a 3d mesh 

        Returns
        -------
        
        if scaled:
        X, Y, Z: tuple of np.ndarray
            defines mesh in real space

        otherwise:
        Xm, Ym, Zm: tuple of np.ndarray
            defines the mesh indices
        """ 
        xm, ym, zm = [list(range(-self.grid[i], self.grid[i]+1)) for i in range(3)]
        
        Xm, Ym, Zm = np.meshgrid(xm, ym, zm, indexing='ij')
        if not scaled:
            return Xm, Ym, Zm
        
        Rm = np.concatenate([Xm.reshape(*Xm.shape,1),
                             Ym.reshape(*Xm.shape,1),
                             Zm.reshape(*Xm.shape,1)], axis = 3)
        
        R = np.einsum('ij,klmj -> iklm', self.U , Rm)
        X = R[0,:,:,:] + self.origin[0]
        Y = R[1,:,:,:] + self.origin[1]
        Z = R[2,:,:,:] + self.origin[2]
        
        return X, Y, Z 

    def evaluate_at(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, scaled: bool = True) -> Tuple[np.ndarray, float]:
        quadrature = np.linalg.det(self.U)
        
        if not scaled:
            Xm, Ym, Zm = X, Y, Z
            return self.rho[Xm, Ym, Zm], quadrature
            
        R = np.concatenate([X.reshape(*X.shape, 1), Y.reshape(*Y.shape, 1), Z.reshape(*Z.shape, 1)], axis=-1)
        R -= self.origin

        U_inv = np.linalg.inv(self.U)

        if R.ndim == 4:
            Rm = np.einsum('ij,klmj -> iklm', U_inv, R)
        elif R.ndim == 2:
            Rm = np.einsum('ij,kj -> ik', U_inv, R)
            Rm = Rm.T

        Rm = np.rint(Rm).astype(int) % self.grid  #TODO implement bicubic interpolation
        if R.ndim == 2:
            Rm = Rm.T

        Xm, Ym, Zm = Rm[0,...], Rm[1,...], Rm[2,...]

        return self.rho[Xm, Ym, Zm], quadrature
