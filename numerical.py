
import numpy as np
from model import *

def rk1(f, t, x, h):
	k1 = h * f(t, x)
	return x + k1


def rk2(f, t, x, h):
	k1 = h * f(t     , x    )
	k2 = h * f(t+h/2, x+k1/2)
	return x + k2

def rk3(f, t, x, h):
	k1 = h * f(t    , x          )
	k2 = h * f(t+h/2, x+k1/2     )
	k3 = h * f(t+h  , x-k1  +k2/2)
	return x + (k1 + 4*k2 + k3) / 6

def rk4(f, t, x, h):
	k1 = h * f(t    , x     )
	k2 = h * f(t+h/2, x+k1/2)
	k3 = h * f(t+h/2, x+k2/2)
	k4 = h * f(t+h  , x+k3  )
	return x + (k1 + 2*k2 + 2*k3 + k4) / 6


eulerForward = rk1
midPoint     = rk2