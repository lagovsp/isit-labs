import random

import matplotlib.pyplot as plt
from input import *


class SWNet:
    @staticmethod
    def e(xs_ref: list[float],
          xs_pred: list[float]) -> float:
        return math.sqrt(sum([(x - xs_pred[i]) ** 2 for i, x in enumerate(xs_ref)]))

    def __init__(self, p: int, norm: float):
        self.p = p
        self.norm = norm
        self.weights = [float(0)] + [random.uniform(0, 1) for _ in range(self.p)]

    def predict_next_x(self, xs: list[float]) -> float:
        return sum([x * self.weights[i + 1] for i, x in enumerate(xs)]) + self.weights[0]

    def _correct_weights(self, refs: list[float], net_ans: float) -> None:
        delta = refs[-1] - net_ans
        for k in range(len(self.weights)):
            print('WEIGHTS', self.weights)
            print(k, refs, refs[k])
            self.weights[k] += self.norm * delta * refs[k]

    def _learn_epoch(self, xs_l: list[float]):
        for i in range(len(xs_l) - self.p):
            print(F'WIN SIZE = {self.p}, {xs_l[i:i + self.p]}')
            net_ans = self.predict_next_x(xs_l[i:i + self.p])
            print(f'xs_l{i + self.p} real {xs_l[i + self.p]}, predicted {net_ans}')
            self._correct_weights(xs_l[i:i + self.p + 1], net_ans)

    def learn(self, xs_l: list[float], m_num: int):
        assert self.p < len(xs_l)

        for m in range(m_num):
            self._learn_epoch(xs_l)


def display_graph(xs: list[float], ys: list[float]) -> None:
    plt.title(F'GRAPH')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(xs, ys, marker='^')
    plt.gcf().savefig(F'graph.png', dpi=500)


def main():
    # assert len(TS_LEARN + TS_PREDICT) == len(XS_LEARN + XS_PREDICT)
    # plt.title(F'GRAPH')
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.plot(TS_LEARN, XS_LEARN, marker='^', color='C1')
    # plt.plot(TS_PREDICT, XS_PREDICT, marker='o', color='C2')
    # plt.gcf().savefig(F'graph.png', dpi=500)
    # print(TS_LEARN)
    print(XS_LEARN)
    net = SWNet(4, 1)
    net.learn(XS_LEARN, 2)


if __name__ == '__main__':
    main()
