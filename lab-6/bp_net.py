import math


class BPNet:
    class Neuron:
        def __init__(self, ws: list[float]) -> None:
            self.ws = ws
            self.last_net = None

        @staticmethod
        def f(net: float) -> float:
            e = math.exp(-net)
            return (1 - e) / (1 + e)

        @staticmethod
        def f_der(net: float) -> float:
            return (1 - BPNet.Neuron.f(net) ** 2) / 2

        def net(self, xs: list[float]) -> float:
            print(xs)
            self.last_net = sum([w * xs[i] for i, w in enumerate(self.ws)])
            return self.last_net

        def out(self, xs: list[float]) -> float:
            print(F'OUT XS: {xs}')
            return BPNet.Neuron.f(self.net(xs))

    def __init__(self,
                 n: int,
                 j: int,
                 m: int,
                 norm: float,
                 t: list[float],
                 def_weight: float = 0.5) -> None:
        self.n = n
        self.j = j
        self.m = m
        self.norm = norm
        self.t = t
        self.hidden_layer = [BPNet.Neuron([def_weight] * (n + 1))] * j
        self.out_layer = [BPNet.Neuron([def_weight] * (j + 1))] * m

    def e(self, ys: list[float]) -> float:
        return math.sqrt(sum([(t - ys[i]) ** 2 for i, t in enumerate(self.t)]))

    def predict(self, xs: list[float]) -> (list[float], list[list[float]]):
        print('PREDICTING . . .')
        xis = [float(1)] + xs
        xjs = [float(1)] + [n.out(xis) for n in self.hidden_layer]
        yms = [n.out(xjs) for n in self.out_layer]
        for m, y in enumerate(yms):
            print(f'y({m}) = {y}')

        return yms, [xis, xjs]

    def _correct_weights(self,
                         ys: list[float],
                         xis: list[float],
                         xjs: list[float]) -> None:
        print('CORRECTING')
        delta_ms = [BPNet.Neuron.f_der(self.out_layer[m].last_net) * (self.t[m] - y)
                    for m, y in enumerate(ys)]
        delta_js = [BPNet.Neuron.f_der(self.hidden_layer[j].last_net) *
                    sum([self.out_layer[m].ws[j + 1] * delta for m, delta in enumerate(delta_ms)])
                    for j in range(len(self.hidden_layer))]

        print(F'delta_ms {delta_ms}')
        print(F'delta_js {delta_js}')

        # correcting layer J
        for j in range(self.j):
            for i in range(len(self.hidden_layer[j].ws)):
                delta = self.norm * xis[i] * delta_js[j]
                self.hidden_layer[j].ws[i] += delta
                print(F'i = {i}, j = {j}, delW = {delta}, w+d = {self.hidden_layer[j].ws[i]}')

        # correcting layer M
        for m in range(self.m):
            for j in range(len(self.out_layer[m].ws)):
                delta = self.norm * xjs[j] * delta_ms[m]
                self.out_layer[m].ws[j] += delta
                print(F'i = {j}, j = {m}, delW = {delta}, w+d = {self.out_layer[m].ws[j]}')

    def learn(self,
              xs: list[float],
              e_th: float,
              ep_limit: int = 10_000) -> (bool, list[list]):
        k = 0
        history = list[list]()

        while True:
            print(k)
            ys, xijs = self.predict(xs)
            e = self.e(ys)
            history.append([k, ys.copy(), e])

            if e <= e_th:
                return True, history
            if k == ep_limit:
                return False, history

            self._correct_weights(ys, xijs[0], xijs[1])
            k += 1
