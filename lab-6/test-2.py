import math


def f(net: float) -> float:
    e = math.exp(-net)
    return 1 / (1 + e)


def f_der(net: float) -> float:
    # return math.exp(net) / ((math.exp(net) + 1) ** 2)
    return f(net) * (1 - f(net))


def init_weights(n: int, j: int, m: int,
                 w1s: list[float],
                 w2s: list[float]) -> (list[list[float]], list[list[float]]):
    w1 = [[w1s[i] for i in range(n + 1)] for _ in range(j)]
    # w1 = [[0.1 for _ in range(n + 1)] for _ in range(j)]
    # w2 = [[0.1 for _ in range(j + 1)] for _ in range(m)]
    w2 = [[w2s[i] for i in range(j + 1)] for _ in range(m)]

    # print(w1)
    # print(w2)
    return w1, w2


def train(x: list[float],
          y: list[float],
          norm: float,
          eps: float,
          n: int,
          j: int,
          m: int,
          w1s: list[float],
          w2s: list[float]) -> (list[list[float]], list[list[float]]):
    w1, w2 = init_weights(n, j, m, w1s, w2s)
    epoch = 0

    while True:
        epoch += 1
        print(F'EP {epoch}')

        hidden_nets = [sum([x[i] * w1[r][i] for i in range(n + 1)]) for r in range(j)]
        for jit in range(j):
            print(F'net_{jit + 1}(1)(e{epoch}) = {hidden_nets[jit]}', end='; ')
        print()

        hidden_fs = [f(net) for net in hidden_nets]
        for jit in range(j):
            print(F'out_{jit + 1}(1)(e{epoch}) = {hidden_fs[jit]}', end='; ')
        print()
        hidden_fs = [float(1)] + hidden_fs

        out_nets = [sum([hidden_fs[l] * w2[k][l] for l in range(j + 1)]) for k in range(m)]
        for mit in range(m):
            print(F'net_{mit + 1}(2)(e{epoch}) = {out_nets[mit]}', end='; ')
        print()

        out_fs = [f(net) for net in out_nets]
        for mit in range(m):
            print(F'out_{mit + 1}(2)(e{epoch}) = {out_fs[mit]}', end='; ')
        print()

        e = math.sqrt(sum([(y[i] - out_fs[i]) ** 2 for i in range(m)]))

        ys_formatted = ', '.join(list(map(lambda x: '{:2.3f}'.format(x), out_fs)))
        print(f'k = {epoch}\ty = ({ys_formatted})\t' + "{: 8.4f}".format(e))

        if e < eps:
            break

        if epoch == 2:
            break

        print(F'BACK PROPAGATION')
        print(F'delM = f_der(net_m(2)(e{epoch}))*(tm - ym(e{epoch})')

        delta_out = [(y[i] - out_fs[i]) * f_der(out_nets[i]) for i in range(m)]
        for mit in range(m):
            der = '{:2.3f}'.format(f_der(out_nets[mit]))
            of = '{:2.3f}'.format(out_fs[mit])
            do = '{:2.3f}'.format(delta_out[mit])
            print(F'delM{mit + 1} = {der}*({y[mit]} - {of}) = {do}')
        print()

        delta_hidden = [f_der(hidden_nets[i]) * sum([delta_out[k] * w2[k][i] for k in range(m)]) for i in range(j)]
        print(F'delJ = f_der(net_j(1)(e{epoch}))*Σ(w_jm(2)(e{epoch})*delM(e{epoch})')
        for jit in range(j):
            der = '{:2.3f}'.format(f_der(hidden_nets[jit]))
            print(F'delJ{jit + 1} = {der}*{sum([delta_out[k] * w2[k][jit] for k in range(m)])} = {delta_hidden[jit]}')
        print()

        for k in range(m):
            for i in range(j + 1):
                w2[k][i] += norm * delta_out[k] * hidden_fs[i]
        for k in range(j):
            for i in range(n + 1):
                w1[k][i] += norm * delta_hidden[k] * x[i]
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

    w1s = [0.5, -0.7, -0.2]
    w2s = [0.5, 0.3, -0.4]

    w1, w2 = train(x, y, norm, eps, N, J, M, w1s, w2s)
    out_fs = test(x, w1, w2, N, J, M)

    print(F'\nLAYER-1 J')
    for j in range(J):
        print(F'j = {j + 1}: {"".join(map("{: 7.2f}".format, w1[j]))}')

    print(F'\nLAYER-2 M')
    for m in range(M):
        print(F'm = {m + 1}: {"".join(map("{: 7.2f}".format, w2[m]))}')

    print(f'\ny = {out_fs}')


if __name__ == '__main__':
    main()
