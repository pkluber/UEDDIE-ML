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
        
        self.U = np.array(self.unitcell)  # Matrix to go from real space to mesh coordinates
        for i in range(3):
            self.U[i,:] = U[i,:] / self.grid[i]
    
    def to_grid(self, X: np.ndarray) -> np.ndarray:
        return np.rint(np.dot(self.U, X)).astype(int) % self.grid

    def from_grid(self, Xm: np.ndarray) -> np.ndarray:
        return np.dot(self.U.T, Xm)

    def mesh_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a 3d mesh 

        Returns
        -------

        X, Y, Z: tuple of np.ndarray
            defines mesh in real space
        """ 
        xm, ym, zm = [list(range(-self.grid[i], self.grid[i]+1)) for i in range(3)]
        
        Xm, Ym, Zm = np.meshgrid(xm, ym, zm, indexing='ij')
        
        Rm = np.concatenate([Xm.reshape(*Xm.shape,1),
                             Ym.reshape(*Xm.shape,1),
                             Zm.reshape(*Xm.shape,1)], axis = 3)
        
        R = np.einsum('ij,klmj -> iklm', self.U.T , Rm)
        X = R[0,:,:,:] + self.origin
        Y = R[1,:,:,:] + self.origin
        Z = R[2,:,:,:] + self.origin
        
        return X, Y, Z 

    def evaluate_at(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, float]:
        R = np.concatenate([X.reshape(*X.shape, 1), Y.reshape(*Y.shape, 1), Z.reshape(*Z.shape, 1)], axis=3)
        R -= self.origin

        Rm = np.einsum('ij,klmj -> iklm', self.U, R)
        Rm = np.rint(Rm).astype(int) % self.grid  #TODO implement bicubic interpolation
        Xm = Rm[0,:,:,:]
        Ym = Rm[1,:,:,:]
        Zm = Rm[2,:,:,:]

        quadrature = np.linalg.det(U)

        return self.rho[Xm, Ym, Zm], quadrature
