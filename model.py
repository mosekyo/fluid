
import numpy as np
from grid import *

class VariableSet(dict):
	def __mul__(self, scalar):
		ans = VariableSet()
		for key in self:
			ans[key] = scalar * self[key]
		return ans

	__rmul__ = __mul__

	def __add__(self, other):
		ans = VariableSet()
		for key in self:
			ans[key] = self[key] + other[key]
		return ans

class Model:

	def __init__(self, s0, trend, t0=0):
		self.t     = t0
		self.epoch = 0
		self.state = s0
		self.trend = trend

	def update(self, estimate, dt):
		self.state  = estimate(self.trend, self.t, self.state, dt)
		self.epoch += 1
		self.t     += dt