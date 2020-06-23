
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

def gauss(x, mu, sigma):
    return 1./(np.sqrt(2*np.pi) * sigma) * np.exp(
        - (x - mu) ** 2 / (2 * sigma ** 2)
    )

def main():
    global dx, dy, nu
    p00 = 1.013e5
    xg = 400
    yg = 400
    H = 200
    L = 300
    nu = 1.5e-3
    dx = L / xg
    dy = H / yg
    dt = 0.00001
    time = 100
    epochs = int(time / dt)
    T0 = 291
    shp = (xg, yg)
    v = np.zeros(shp)
    u = np.zeros(shp)
    T = T0 * np.ones(shp)
    z0 = np.linspace(0, H, yg)
    x0 = np.linspace(0, L, xg)
    (z, x) = np.meshgrid(z0, x0)
    pz = p00 * np.exp(-g*(z + H/20 * np.sin(2*np.pi*(x/L+0.3*z/H)))/(R * T0))
    pz = p00 * (1 + 0.001 * np.sin(2*np.pi*(x/L+0.4*z/H)))
    pz = p00 * (1 + gauss(x, L/2, L/40) * gauss(z, H/2, L/40) )
    p = pz
    rho = p / (R * T)
    extQ = np.exp(-10*z/H) * np.ones(shp) * (gauss(x, L/3, L/10) - gauss(x, 2*L/3, L/10))
    status = (u, v, rho, p, T)
    exterior = (0, 0, 0)
    def set(ax, name):
        ax.set_title(name)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
    for i in range(epochs+1):
        status = eulerMethod(status, exterior, navierStokes, dt)
        if i % 1000 == 0:
            file = open("dmp/%d.pydmp" % i, 'wb')
            pickle.dump(status, file)
            file.close()


if __name__ == '__main__':
    main()