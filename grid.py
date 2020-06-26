
import numpy as np
from arraylib import *

def wrapArray(func):
    def wrapper(self, *args, **kw):
        return Variable(  self.grid,
                          func( self.grid,
                                self.view(np.ndarray),
                                *args, **kw )
                       )
    return wrapper

class Grid:
    '''
    Class for n-dimensional rectangular descrete grids, which can be
    used to create variable fields based on it and do finite
    differentiations with ndarray of the same shape of the grid.


    Parameters
    ----------

    xs : a series of 1-dimensional np.ndarray


    Attributes
    ----------
    
    ndim : int
        Spatial dimension of the grid.
    shape : tuple of int
        Numbers of points of each dimension.
    xs : a list of 1-dimensional np.ndarray
        Coordinate value of the points in each axis.
    minxs : a list of array.dtype
        Minimum coordinates of each axis.
    maxxs : a list of array.dtype
        Maximun coordinates of each axis
    Xs : a list of np.ndarray of the same shape as the grid's
        Mesh grid version of `xs`.
    deltas : a list of pseudo-1-dimensional np.ndarray object
             that can be broadcast into the grid's shape
        Coordinate differentiations of each axis, used to calculate
        finite differentiations of a variable.
        Elements in `deltas` will be simplified into a single number
        if the coordinates are in equidistances.
    '''

    def __init__(self, *xs):
        self.xs     = [*xs]
        self.maxxs  = [np.max(x) for x in xs]
        self.minxs  = [np.min(x) for x in xs]
        self.ndim   = len(xs)
        self.Xs     = np.meshgrid(*xs, indexing='ij')
        self.shape  = self.Xs[0].shape
        diffxs      = [diff1(x) for x in xs]
        newshape    = lambda i: tuple([
                            len(self.xs[i]) if j == i
                            else 1 for j in range(self.ndim)
                      ])
        deltas      = [ diffxs[i].reshape(newshape(i))
                            for i in range(self.ndim)
                      ]
        self.deltas = [ delta.flat[0] if np.allclose(
                            0,
                            (delta-delta.flat[0])/delta.flat[0] )
                        else delta for delta in deltas
                      ]

    def field(self, f):
        return Variable( grid  = self,
                         array = f(*self.Xs) )

    def emptyVariable(self, rank=0):
        return Variable(
                grid  = self,
                array = np.empty([self.ndim]*rank + list(self.shape))
            )

    def partial(self, arr, axis):
        # arr.ndim - self.ndim <=> self.rankOf(arr)
        return diff1(arr, arr.ndim-self.ndim+axis) / self.deltas[axis]

    def partial2s(self, arr, axis):
        # arr.ndim - self.ndim <=> self.rankOf(arr)
        return diff2s(arr, arr.ndim-self.ndim+axis) / self.deltas[axis]

    def partial2t(self, arr, ax1, ax2):
        # arr.ndim - self.ndim <=> self.rankOf(arr)
        return diff2t( arr,
                       arr.ndim-self.ndim+ax1,
                       arr.ndim-self.ndim+ax2
                     ) / (self.deltas[ax1] * self.deltas[ax2])

    def laplacian(self, arr):
        return sum([ self.partial2s(arr,axis)
                         for axis in range(self.ndim)])

    def grad(self, arr):
        return np.array([
            self.partial(arr,axis)
                for axis in range(self.ndim)
        ])

    def div(self, arr):
        if arr.ndim <= self.ndim:
            raise Exception("Insufficient tensor rank.")
        return sum([
            self.partial(arr[axis], axis)
                for axis in range(self.ndim)
        ])


class MetricGrid(Grid):
    def __init__(self, *xs, gb=None, gt=None, g=None, Γ=None):
        super().__init__(*xs)
        if gb == gt == None:
            raise Exception("Metric not given")
        n   = self.ndim
        inv = onAxes(2,3)(np.linalg.inv)
        det = lambda arr: np.linalg.det(arr.transpose((2,3,0,1)))
        pd  = lambda arr, axis: self.partial(arr, axis)
        if gb == None: gb = inv(gt)
        if gt == None: gt = inv(gb)
        if g  == None: g  = det(gb)
        if Γ  == None:
            Γ = self.emptyTensor(rank=3)
            for i in range(n):
             for j in range(n):
              for k in range(n):
                Γ[i,j,k] = 0.5 * sum([
                    gt[i,m] * (
                        pd(gb[m,j],k) + pd(gb[m,k],j) - pd(gb[j,k],m)
                    ) for m in range(n)
                ])
        self.gb = gb
        self.gt = gt
        self.g  = g
        self.Γ  = Γ

    # to be completed
    def partial(self, arr, axis):
        def curvTerm(arr, a):
            @onAxis(a)
            def inner(arr):
                return np.einsum('ijk,k...->ij...', Γ, arr)
            return inner(arr)
        return sum([ gt[axis,m] * (
                        super().partial(arr,m)
                        + sum([
                                curvTerm(arr,a)[:,m]
                                    for a in range(self.rankOf(arr))
                              ])
                        )  for m in range(self.ndim)
                   ])

    # to be completed
    def grad(self, arr):
        def curvTerm(arr, a):
            @onAxis(a)
            def inner(arr):
                return np.einsum('ijk,k...->ij...', Γ, arr)
            return inner(arr)
        return np.einsum('im,m...->i...', gt)


class Variable(np.ndarray):
    '''
    Class for tensor field on a grid based on `numpy.ndarray`.
    A rank-r tensor over a n-dimensional grid is represented by
    a (m+n) dimensional ndarray.
    Suppose the grid is of shape (L1, L2, ..., Ln),
    then the ndarray is of shape (n, n, ..., n, L1, L2, ..., Ln)
    where spatial dimensions are put at the end for the sake of
    continuity in memory use in order to improve efficiency.

    Parameters
    ----------

    grid : Grid object
        Underlying grid.
    array : numpy.ndarray object
        The variable data itself.

    Attributes
    ----------

    grid : Grid object
        Underlying grid.
    rank : int
        Rank of the tensor variable, equals to `array.ndim-grid.ndim`

    '''

    def __new__(cls, grid, array):
        #if grid.shape != array.shape:
        #    raise Exception("Grid size dismatch.")
        obj = array.view(cls)
        obj.grid = grid
        obj.rank = len(obj.shape) - len(grid.shape)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        #if any([x.grid != inputs[0].grid for x in inputs]):
        #    raise Exception("Grid dismatch.")
        args = [x.view(np.ndarray) for x in inputs]
        ans = super(Variable,self).__array_ufunc__(ufunc, method,
                                                   *args, **kw)
        ans = Variable(inputs[0].grid, ans)
        return ans

    @wrapArray
    def partial  (grid, arr, axis): return grid.partial  (arr, axis)
    @wrapArray
    def partial2s(grid, arr, axis): return grid.partial2s(arr, axis)
    @wrapArray
    def partial2t(grid, arr,  *ax): return grid.partial2t(arr,  *ax)
    @wrapArray
    def grad     (grid, arr):       return grid.grad     (arr)
    @wrapArray
    def div      (grid, arr):       return grid.div      (arr)
    @wrapArray
    def laplacian(grid, arr):       return grid.laplacian(arr)


def main():
    global z, g
    n = 4
    g = Grid(np.linspace(0,1,n), np.linspace(0,1,n))
    x = Variable(g, np.random.randn(n,n))
    y = Variable(g, np.random.randn(n,n))
    z = x + y
    print(z)
    print(z.laplacian())

main()