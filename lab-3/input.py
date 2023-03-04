from typing import Callable
import math

A = 1
B = 5
N = 20

X_F: Callable[[float], float] = lambda t: math.sqrt(0.1 * t) + 1

_diff = (B - A) / (N - 1)
TS_LEARN = [A + (_diff * i) for i in range(N)]
XS_LEARN = [X_F(t) for t in TS_LEARN]

TF: Callable[[float], float] = lambda x: x

TS_PREDICT = [B + (_diff * i) for i in range(1, N + 1)]
XS_PREDICT = [X_F(t) for t in TS_PREDICT]
