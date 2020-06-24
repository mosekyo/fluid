
import numpy as np
from arrayLib import *

class Grid:
    def __init__(self, xs):
        self.xs     = xs
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

    def partial(self, var, axis):
        return diff1(var, var.ndim-grid.ndim+axis) / self.deltas[axis]

    def partial2(self, var, axis):
        return diff2(var, var.ndim-grid.ndim+axis) / self.deltas[axis]

    def laplacian(self):
        pass

    def grad(self, var):
        pass

    def field(self, f):
        return f(*self.Xs)

    def emptyVariable(self):
        return np.empty(self.shape)


# to be abandoned
class Variable(np.ndarray):

    def partial(self, axis):
        return self.grid.partial(self, axis)

    def partial2(self, axis):
        return self.grid.partial2(self, axis)

    def laplacian(self):
        return self.grid.laplacian(self)

    def grad(self):
        return Vector(
            self.grid,
            [self.partial(i) for i in self.grid.ndim]
        )

    def __new__(cls, grid, array):
        if grid.shape != array.shape:
            raise Exception("Grid size dismatch.")
        obj = array.view(cls)
        obj.grid = grid
        return obj


class Vector:

    def __init__(self, grid, vs):
        if len(vs) != grid.ndim:
            raise Exception("Grid dimension dismatch.")
        self.grid = grid
        self.vs = vs

def main():
    g = Grid([np.linspace(0, 5, 5), np.array([1,2,4,8])])
    emp = np.empty(g.shape)
    x = Variable(g, emp)
    g.emptyVariable()

if __name__ == '__main__':
    main()