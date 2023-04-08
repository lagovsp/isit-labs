import math


def f(net: float) -> float:
    e = math.exp(-net)
    return (1 - e) / (1 + e)


def f_der(net: float) -> float:
    return (1 - f(net) ** 2) / 2


def init_weights(n: int, j: int, m: int) -> (list[list[float]], list[list[float]]):
    w1 = [[0.1 for _ in range(n + 1)] for _ in range(j)]
    w2 = [[0.1 for _ in range(j + 1)] for _ in range(m)]

    return w1, w2


def train(x: list[float],
          y: list[float],
          norm: float,
          eps: float,
          n: int,
          j: int,
          m: int) -> (list[list[float]], list[list[float]]):
    w1, w2 = init_weights(n, j, m)
    epoch = 0

    while True:
        hidden_nets = [sum([x[i] * w1[r][i] for i in range(n + 1)]) for r in range(j)]
        hidden_fs = [f(net) for net in hidden_nets]

        hidden_fs = [float(1)] + hidden_fs

        out_nets = [sum([hidden_fs[l] * w2[k][l] for l in range(j + 1)]) for k in range(m)]
        out_fs = [f(net) for net in out_nets]

        e = math.sqrt(sum([(y[i] - out_fs[i]) ** 2 for i in range(m)]))
        print(f'k = {epoch}\ty = {out_fs}\t\t' + "{0:0.5f}".format(e))

        if e < eps:
            break

        delta_out = [(y[i] - out_fs[i]) * f_der(out_nets[i]) for i in range(m)]
        delta_hidden = [f_der(hidden_nets[i]) * sum([delta_out[k] * w2[k][i] for k in range(m)]) for i in range(j)]

        for k in range(m):
            for i in range(j + 1):
                w2[k][i] += norm * delta_out[k] * hidden_fs[i]
        for k in range(j):
            for i in range(n + 1):
                w1[k][i] += norm * delta_hidden[k] * x[i]
        epoch += 1
    return w1, w2


def test(x: list[float],
         w1: list[list[float]],
         w2: list[list[float]],
         N: int,
         J: int,
         M: int) -> list[float]:
    hidden_nets = [sum([x[i] * w1[j][i] for i in range(N + 1)]) for j in range(J)]
    hidden_fs = [f(net) for net in hidden_nets]
    hidden_fs = [float(1)] + hidden_fs

    out_nets = [sum([hidden_fs[j] * w2[k][j] for j in range(J + 1)]) for k in range(M)]
    out_fs = [f(net) for net in out_nets]

    return out_fs


def main():
    N = 2
    J = 1
    M = 2
    x = [1, 1, -1]
    y = [0.2, -0.1]
    norm = 1
    eps = 0.001

    # N = 3
    # J = 3
    # M = 4
    # x = [1, 0.3, -0.1, 0.9]
    # y = [0.1, -0.6, 0.2, 0.7]
    # n = 1

    w1, w2 = train(x, y, norm, eps, N, J, M)
    out_fs = test(x, w1, w2, N, J, M)

    print(F'\nLAYER-1 J:')
    for j in range(J):
        print(F'j = {j + 1}, {"".join(map("{: 3.2f}".format, w1[j]))}')

    print(F'\nLAYER-2 M:')
    for m in range(M):
        print(F'm = {m + 1}, {"".join(map("{: 3.2f}".format, w2[m]))}')

    print(f'\ny = {out_fs}')


if __name__ == '__main__':
    main()
