import math
from typing import Callable


class TF:
    def __init__(self,
                 f: Callable[[float], float],
                 f_der: Callable[[float], float],
                 threshold_val: float):
        self.f = f
        self.f_der = f_der
        self.y_threshold_val = threshold_val

    def y(self, net: float) -> (int, float):
        out = self.f(net)
        y = 1 if out >= self.y_threshold_val else 0
        return y, out


class RBFNet:
    class RBFNeuron:
        def __init__(self, cc: list[int], is_one: bool, num: int):
            self.is_one = is_one
            self.center_coords = cc
            self.num = num

        def __repr__(self) -> str:
            return f'RBF{self.num}{self.center_coords}'

        def compute_phi(self, s: list[int]) -> float:
            phi = 1 if self.is_one else math.exp(
                -sum([(x_val - self.center_coords[i]) ** 2 for i, x_val in enumerate(s)]))
            return phi

    def __init__(self,
                 f_tf: TF,
                 norm: float,
                 name: str = None):
        self.norm = norm
        self.tf = f_tf
        self.name = name
        self.rbf_neurons = list[RBFNet.RBFNeuron]()
        self.vs = list[float]()

    def predict(self, args: list[int]) -> (int, float, float):  # y, out, net
        net = sum([self.vs[j] * rbf_n.compute_phi(args)
                   for j, rbf_n in enumerate(self.rbf_neurons)]) * self.norm
        return *self.tf.y(net), net

    def set_rbf_neurons(self, rbfs: list[RBFNeuron]) -> None:
        self.rbf_neurons = rbfs
        self.vs = [0] * (len(rbfs))

    def _correct_weights(self,
                         net: float,
                         delta: int,
                         xs: list[int]) -> None:
        for i, _ in enumerate(self.vs):
            self.vs[i] += delta * self.tf.f_der(net) * self.rbf_neurons[i].compute_phi(xs[:-1])

    def learn_epoch(self, xs_sets: list[list[int]]) -> None:
        for i, xs in enumerate(xs_sets):
            y, out, net = self.predict(xs[:-1])
            self._correct_weights(net, xs[-1] - y, xs)


def test_sets(net: RBFNet, xs_sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list[int](), 0
    for xs in xs_sets:
        net_ans, *_ = net.predict(xs[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == xs[-1] else 1
    return answers, mistakes


def learn(net: RBFNet,
          sets: list[list[int]],
          learn_indexes: set[int],
          epoch_limit=None) -> (bool, list[list]):
    epochs = list()
    learn_sets = [sets[i] for i in learn_indexes]

    # learning which are more:  0s or 1s ?
    ones = sum(map(lambda x: x[-1], sets), start=0)
    centers = set[int]()
    for i, s in enumerate(sets):
        if not s[-1] == 1:
            continue
        centers.add(i)

    # setting RBF-neurons
    if not ones == len(sets) - ones:
        centers = {i for i in (
            centers if (ones < len(sets) - ones) else (set(range(len(sets))) - centers))}

    net.set_rbf_neurons([RBFNet.RBFNeuron(list(), True, 0)] + [RBFNet.RBFNeuron(sets[ci][:-1], False, i + 1)
                                                               for i, ci in enumerate(sorted(centers))])
    j = 0
    while True:
        if epoch_limit is not None and j == epoch_limit:
            return False, epochs
        net.learn_epoch(learn_sets)
        j += 1
        answers, mistakes = test_sets(net, sets)
        epochs.append([j, list(map(lambda x: x * net.norm, net.vs.copy())), answers.copy(), mistakes])
        if mistakes == 0:
            return True, epochs
