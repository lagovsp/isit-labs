from configure import *


class FPNet:
    def __init__(self, p: int, norm: float):
        self.p = p
        self.norm = norm
        self.weights: list[float] = [0] * (self.p + 1)

    @staticmethod
    def e(xs_ref: list[float], xs_pred: list[float]) -> float:
        assert len(xs_ref) == len(xs_pred)
        return math.sqrt(sum([(x - xs_pred[i]) ** 2 for i, x in enumerate(xs_ref)]))

    def predict_next_x(self, xs: list[float]) -> float:
        assert len(xs) == self.p
        return self.weights[0] + sum([self.weights[k] * xs[k - 1] for k in range(1, self.p + 1)])

    def predict_next_xs(self, init_xs: list[float], num: int) -> list[float]:
        for i in range(num):
            init_xs.append(self.predict_next_x(init_xs[-self.p:]))
        return init_xs

    def _predict_learn_range(self, xs_l: list[float]) -> list[float]:
        return [self.predict_next_x(xs_l[i:i + self.p]) for i in range(len(xs_l) - self.p)]

    def _learn_epoch(self, xs_l: list[float]) -> list[float]:
        predicted = list[float]()

        for i in range(len(xs_l) - self.p):
            predicted.append(self.predict_next_x(xs_l[i:i + self.p]))

            for j in range(1, self.p + 1):
                self.weights[j] += self.norm * (xs_l[i + self.p] - predicted[i]) * xs_l[i + j]

        return predicted

    def learn(self,
              xs_l: list[float],
              m_limit: int = 10_000,
              e_th=10 ** (-4)) -> (int, float):
        assert self.p < len(xs_l)

        m = 0
        while True:
            e = FPNet.e(xs_l[self.p:], self._learn_epoch(xs_l))
            m += 1
            if m == m_limit:
                return m, e
            if e < e_th:
                return m, e
