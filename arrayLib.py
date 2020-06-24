
import numpy as np

def onAxis(axis):
    def decorator(func):
        def wrapper(arr, *args, **kw):
            alter = tuple(
                [axis] + [
                    (0 if x == axis else x)
                    for x in range(1, arr.ndim)
                ]
            )
            ans = arr.transpose(alter)
            ans = func(ans, *args, **kw)
            return ans.transpose(alter)
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