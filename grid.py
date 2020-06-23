
import numpy as np

class Grid:
    def __init__(self, shape):
        self.shape = shape

    def emptyVariable(self):
        return Variable(self, np.empty(self.shape))

class Variable(np.ndarray):

    def __onAxis(axis):
        def decorator(func):
            def wrapper(arr, *args, **kw):
                alter = tuple(
                    [axis] + [(0 if x == axis else x) for x in range(1, arr.ndim)]
                )
                ans = arr.transpose(alter)
                ans = func(ans, *args, **kw)
                return ans.transpose(alter)
            return wrapper
        return decorator

    def partial(self, axis):
        delta = self.grid.deltas[axis]
        @Variable.__onAxis(axis)
        def inner(arr, delta):
            ans = self.grid.emptyVariable()
            ans[1:-1] = (self[2:] - self[:-2]) / (2 * delta)
            ans[0]    = (self[1]  - self[0]  ) / delta
            ans[-1]   = (self[-1] - self[-2] ) / delta
            return ans
        return inner(self, delta)
    
    def partial2(self, axis):
        delta = self.grid.deltas[axis]
        @Variable.__onAxis(axis)
        def inner(arr, delta):
            ans = self.grid.emptyVariable()
            ans[1:-1] = (self[2:] - 2*self[1:-1] + self[:-2]) / (delta ** 2)
            ans[0]    = ans[1]  / 2
            ans[-1]   = ans[-2] / 2
            return ans
        return inner(self, delta)

    def laplacian(self):
        pass

    def __new__(cls, grid, array):
        obj = array.view(cls)
        if grid.shape != obj.shape:
            raise Exception("Grid dismatch")
        obj.grid = grid
        return obj

def main():
    shp = (100, 100)
    g = Grid(shp)
    emp = np.empty(shp)
    x = Variable(g, emp)
    g.emptyVariable()

if __name__ == '__main__':
    main()