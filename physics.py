
import numpy as np

# incomplete!

grad = lambda x: x.grad()
la   = lambda x: x.laplacian()
div  = lambda x: x.div()

def adv(V, x):
    return V.dot(grad(x))

def vt_ns(V, ρ, p, ν, F):
    return - adv(V,V) - grad(p)/ρ + F + ν/3*grad(div(V)) + ν*la(V)