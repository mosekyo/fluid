
import numpy as np
import pickle

# constants
dx = 100
dy = 100
g  = 9.801
nu = 1.5e-5
cp = 1004
R  = 287
k  = 30

def onAxis(axis):
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

def partial(arr, axis, delta):
    @onAxis(axis)
    def inner(arr, delta):
        ans = np.empty(arr.shape)
        ans[1:-1] = (arr[2:] - arr[:-2]) / (2 * delta)
        ans[0]    = (arr[1]  - arr[0]  ) / delta
        ans[-1]   = (arr[-1] - arr[-2] ) / delta
        return ans
    return inner(arr, delta)

def partial2(arr, axis, delta):
    @onAxis(axis)
    def inner(arr, delta):
        ans = np.empty(arr.shape)
        ans[1:-1] = (arr[2:] - 2*arr[1:-1] + arr[:-2]) / (delta ** 2)
        ans[0]    = 0
        ans[-1]   = 0
        return ans
    return inner(arr, delta)

def laplacian(arr, deltas):
    ans = np.zeros(arr.shape)
    for (axis, delta) in zip([i for i in range(arr.ndim)], deltas):
        ans += partial2(arr, axis, delta)
    return ans

def nihilBorder(arr, axis):
    @onAxis(axis)
    def inner(arr):
        arr[0] = 0
        arr[-1] = 0
        return arr
    return inner(arr)

def navierStokes(status, exterior):
    u, v, rho, p, T = status
    extQ, Fx, Fy = exterior
    # operators
    px  = lambda arr: partial(arr, 0, dx)
    py  = lambda arr: partial(arr, 1, dy)
    la  = lambda arr: laplacian(arr, (dx,dy))
    adv = lambda arr: u*px(arr) + v*py(arr)
    # navier-stokes eqs
    ut   = - adv(u) - px(p)/rho + Fx + nu*la(u)
    vt   = - adv(v) - py(p)/rho + Fy + nu*la(v)
    rhot = - px(rho*u) - py(rho*v)
    Q    = k*la(T) + extQ
    A    = rho*Q + adv(p) - cp*rho*adv(T)
    B    = R*T*rhot
    Tt   = (A + B) / (rho * (cp - R))
    pt   = B + rho*R*Tt
    # border condition
    nihilBorder(ut, 0)
    nihilBorder(vt, 1)
    # return tuple
    return (ut, vt, rhot, pt, Tt)

def step(status, tendency, dt):
    return tuple([
        x + dervx*dt for (x, dervx) in zip(status, tendency)
    ])

def eulerMethod(x0, ext, f, dt):
    return step(x0, f(x0, ext), dt)

def main():
    global dx, dy
    shp = (xg, yg) = (200, 200)
    H = 200
    L = 300
    dx = L / xg
    dy = H / yg
    dt = 0.00001
    (y, x) = np.meshgrid(
        np.linspace(0, H, yg),
        np.linspace(0, L, xg)
    )
    p00 = 1.013e5
    T0 = 291
    u = np.zeros(shp)
    v = np.zeros(shp)
    T = T0 * np.ones(shp)
    def gauss(x, mu, sigma):
        return 1./(np.sqrt(2*np.pi) * sigma) * np.exp(
            - (x - mu) ** 2 / (2 * sigma ** 2)
        )
    p = p00 * (1 + gauss(x, L/2, L/40) * gauss(y, H/2, L/40) )
    rho = p / (R * T)
    status = (u, v, rho, p, T)
    exterior = (0, 0, 0)
    for i in range(200000 + 1):
        status = (u, v, rho, p, T) = eulerMethod(status, exterior, navierStokes, dt)
        # plot graph codes ...


if __name__ == '__main__':
    main()
