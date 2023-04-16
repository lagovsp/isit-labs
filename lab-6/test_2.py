import math


def f(net: float) -> float:
    e = math.exp(-net)
    return 1 / (1 + e)


def f_der(net: float) -> float:
    return f(net) * (1 - f(net))


def train(x: list[float],
          y: list[float],
          norm: float,
          eps: float,
          n: int,
          j: int,
          m: int,
          w1: list[list[float]],
          w2: list[list[float]]) -> (list[list[float]], list[list[float]]):
    print(F'START WEIGHTS: J{w1}, M{w2}\n')

    epoch = 0
    while True:
        epoch += 1
        print(F'EP {epoch}')

        hidden_nets = [sum([x[i] * w1[r][i] for i in range(n + 1)]) for r in range(j)]
        hidden_fs = [f(net) for net in hidden_nets]

        for jit in range(j):
            print(F'net_l1_{jit + 1} = {hidden_nets[jit]}', end='; ')
            print(F'out_l1_{jit + 1} = {hidden_fs[jit]}', end=';\n')
        print()

        hidden_fs = [float(1)] + hidden_fs
        out_nets = [sum([hidden_fs[l] * w2[k][l] for l in range(j + 1)]) for k in range(m)]
        out_fs = [f(net) for net in out_nets]

        for mit in range(m):
            print(F'net_l2_{mit + 1} = {out_nets[mit]}', end='; ')
            print(F'out_l2_{mit + 1} = {out_fs[mit]}', end=';\n')
        print()

        e = math.sqrt(sum([(y[i] - out_fs[i]) ** 2 for i in range(m)]))

        ys_formatted = ', '.join(list(map(lambda x: '{:2.3f}'.format(x), out_fs)))
        print(f'y = ({ys_formatted})\t' + "E = {: 6.4f}".format(e), end='\n\n')

        if e < eps:
            break

        delta_out = [(y[i] - out_fs[i]) * f_der(out_nets[i]) for i in range(m)]
        for mit in range(m):
            print(F'del_l2_{mit + 1} = {f_der(out_nets[mit]) * (y[mit] - out_fs[mit])}')
        print()

        delta_hidden = list[float]()

        for jit in range(j):
            delta = f_der(hidden_nets[jit]) * (sum([delta_out[k] * w2[k][j] for k in range(m)]))
            delta_hidden.append(delta)
            print(F'del_l1_{jit + 1} = {delta}')
        print()

        for mit in range(m):
            for jit in range(j + 1):
                w2[mit][jit] += norm * delta_out[mit] * hidden_fs[jit]
        for jit in range(j):
            for nit in range(n + 1):
                w1[jit][nit] += norm * delta_hidden[jit] * x[nit]

        print(F'J-LAYER: {w1}')
        print(F'M-LAYER: {w2}')
        print()

        if epoch == 2:
            break

    return w1, w2


def test(x: list[float],
         w1: list[list[float]],
         w2: list[list[float]],
         n: int,
         j: int,
         m: int) -> list[float]:
    hidden_nets = [sum([x[nit] * w1[jit][nit] for nit in range(n + 1)]) for jit in range(j)]
    hidden_fs = [float(1)] + [f(net) for net in hidden_nets]

    out_nets = [sum([hidden_fs[jit] * w2[mit][jit] for jit in range(j + 1)]) for mit in range(m)]
    out_fs = [f(net) for net in out_nets]

    return out_fs


def main():
    n, j, m = 2, 1, 2

    x = [1, 1, -1]
    y = [2, -1]

    norm = 1
    eps = 0.001

    w1s = [[0.5, -0.7, -0.2]]
    w2s = [[0.5, 0.3], [0.5, -0.4]]

    w1, w2 = train(x, y, norm, eps, n, j, m, w1s, w2s)
    out_fs = test(x, w1, w2, n, j, m)

    # print(F'\nLAYER-1 J')
    # for j in range(j):
    #     print(F'j = {j + 1}: {"".join(map("{: 7.2f}".format, w1[j]))}')
    #
    # print(F'\nLAYER-2 M')
    # for m in range(m):
    #     print(F'm = {m + 1}: {"".join(map("{: 7.2f}".format, w2[m]))}')
    #
    # print(f'\ny = {out_fs}')


if __name__ == '__main__':
    main()
