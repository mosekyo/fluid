
import numpy as np

def onAxis(axis):
    def decorator(func):
        def wrapper(arr, *args, **kw):
            if axis == 0:
                return func(arr, *args, **kw)
            else:
                perm = [axis] + [
                           (0 if x == axis else x)
                               for x in range(1, arr.ndim)
                       ]
                ans = arr.transpose(perm)
                ans = func(ans, *args, **kw)
                return ans.transpose(perm)
        return wrapper
    return decorator

def onAxes(*axes):
    def decorator(func):
        def wrapper(arr, *args, **kw):
            perm = [*axes] + [i for i in arr.ndims if i not in axes]
            inv = np.argsort(perm)
            ans = arr.transpose(perm)
            ans = func(ans, *args, **kw)
            return ans.transpose(inv)
        return wrapper
    return decorator

def diff1(arr, axis=0):
    @onAxis(axis)
    def inner(arr):
        ans = np.empty(arr.shape)
        ans[1:-1] = (arr[2:] - arr[:-2]) / 2
        ans[0]    = (arr[1]  - arr[0]  )
        ans[-1]   = (arr[-1] - arr[-2] )
        return ans
    return inner(arr)

def diff2(arr, axis=0):
    @onAxis(axis)
    def inner(arr):
        ans = np.empty(arr.shape)
        ans[1:-1] = (arr[2:] - 2*arr[1:-1] + arr[:-2])
        ans[0]    = ans[1]  / 2
        ans[-1]   = ans[-2] / 2
        return ans
    return inner(arr)

def diff2t(arr, ax1, ax2):
    @onAxes(ax1, ax2)
    def inner(arr):
        ans = np.empty(arr.shape)
        ans[1:-1, 1:-1] = ( arr[ 2:, 2:] - arr[ 2:,:-2]
                          - arr[:-2, 2:] + arr[:-2,:-2] ) / 4
        ans[   0, 1:-1] = ( arr[  1, 2:] - arr[  1,:-2]
                          - arr[  0, 2:] + arr[  0,:-2] ) / 2
        ans[  -1, 1:-1] = ( arr[ -1, 2:] - arr[ -1,:-2]
                          - arr[ -2, 2:] + arr[ -2,:-2] ) / 2
        ans[1:-1,    0] = ( arr[ 2:,  1] - arr[ 2:,  0]
                          - arr[:-2,  1] + arr[:-2,  0] ) / 2
        ans[1:-1,   -1] = ( arr[ 2:, -1] - arr[ 2:, -2]
                          - arr[:-2, -1] + arr[:-2, -2] ) / 2
        ans[   0,    0] = ( arr[  1,  1] - arr[  1,  0]
                          - arr[  0,  1] + arr[  0,  0] )
        ans[  -1,    0] = ( arr[ -1,  1] - arr[ -1,  0]
                          - arr[ -2,  1] + arr[ -2,  0] )
        ans[   0,   -1] = ( arr[  1, -1] - arr[  1, -2]
                          - arr[  0, -1] + arr[  0, -2] )
        ans[  -1,   -1] = ( arr[ -1, -1] - arr[ -1, -2]
                          - arr[ -2, -1] + arr[ -2, -2] )
        return ans
    return inner(arr)

def nihilBorder(arr, axis, depth=1):
    @onAxis(axis)
    def inner(arr):
        arr[:depth]  = 0
        arr[-depth:] = 0
        return arr
    return inner(arr)

def nihilBorderAll(arr, depth=1):
    for i in range(arr.ndim):
        nihilBorder(arr, axis=i, depth=depth)
    return arr
