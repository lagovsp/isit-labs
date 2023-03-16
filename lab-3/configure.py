from typing import Callable
import math

N = 20

# Random var 1
# VAR, A, B = 1, -5, 5
# F_STR = 'x(t) = 0.5 * math.cos(0.5 * t) - 0.5'
# X_F: Callable[[float], float] = lambda t: 0.5 * math.cos(0.5 * t) - 0.5

# Random var 3
# VAR, A, B = 3, -2, 2
# F_STR = 'x(t) = exp(t - 1)'
# X_F: Callable[[float], float] = lambda t: math.exp(t - 1)

# Sergey Ex
# VAR, A, B = 6, 1, 5
# F_STR = 'x(t) = sqrt(0.1 * t) + 1'
# X_F: Callable[[float], float] = lambda t: math.sqrt(0.1 * t) + 1

# Andrew —> Sergey
VAR, A, B = 10, -2, 2
F_STR = 'x(t) = sin(t - 1)'
X_F: Callable[[float], float] = lambda t: math.sin(t - 1)

# Random var 11
# VAR, A, B = 11, 2, 3
# F_STR = 'x(t) = tan(t)'
# X_F: Callable[[float], float] = lambda t: math.tan(t)

# Random var 14
# VAR, A, B = 14, 4.5, 5
# F_STR = 'x(t) = sin(2 * sqrt(exp(t)))'
# X_F: Callable[[float], float] = lambda t: math.sin(2 * math.sqrt(math.exp(t)))

# Random var 22
# VAR, A, B = 22, 1.2, 1.5
# F_STR = 'x(t) = 0.2 * sin(4 * t)'
# X_F: Callable[[float], float] = lambda t: 0.2 * math.sin(4 * t)

_diff = (B - A) / (N - 1)
TS_LEARN = [A + (_diff * i) for i in range(N)]
XS_LEARN = [X_F(t) for t in TS_LEARN]

TF: Callable[[float], float] = lambda x: x

TS_PREDICT = [B + (_diff * i) for i in range(1, N + 1)]
XS_PREDICT = [X_F(t) for t in TS_PREDICT]
