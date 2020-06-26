
import numpy as np
from grid import *

grad = lambda    x: x.grad()
la   = lambda    x: x.laplacian()
div  = lambda    x: x.div()
adv  = lambda V, x: np.einsum('i...,i...->...', V, grad(x))

def Vt_navierStokes(V, ρ, p, F=0, ν=0):
    ans = - adv(V,V) - grad(p)/ρ + F
    if ν != 0:
        ans += ν/3*grad(div(V)) + ν*la(V)
    return ans

def ρt_continuous(V, ρ, G=0):
    return - div(ρ*V) + G
