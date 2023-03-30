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
        y = 1 if out - self.y_threshold_val >= 0 else 0
        return y, out


class Net:
    class RBFNeuron:
        def __init__(self, coords: list[int], fictitious: bool):
            self.is_fictitious = fictitious
            self.center = coords

        def __repr__(self) -> str:
            return f'RBF{self.center}'

        def compute_set(self, s: list[int]) -> float:
            print(f'got s {s}, c {self.center}')
            print(1 if self.is_fictitious else math.exp(
                -sum([(x_val - self.center[i]) ** 2 for i, x_val in enumerate(s)])))
            return 1 if self.is_fictitious else math.exp(
                -sum([(x_val - self.center[i]) ** 2 for i, x_val in enumerate(s)]))

    def __init__(self,
                 tf: TF,
                 norm: float,
                 name: str = None):
        self.norm = norm
        self.tf = tf
        self.name = name
        self.rbf_neurons = list[Net.RBFNeuron]()
        self.vs = list[float]()

    def predict(self, args: list[int]) -> (int, float, float):  # y, out, net
        # print(len(self.vs))
        # print(len(self.rbf_neurons))
        net = sum([self.vs[j] * rbf_n.compute_set(args[1:])
                   for j, rbf_n in enumerate(self.rbf_neurons)])
        net *= self.norm
        return *self.tf.y(net), net

    def set_rbf_neurons(self, rbfs: list[RBFNeuron]) -> None:
        self.rbf_neurons = rbfs
        self.vs = [0] * (len(rbfs))

    def _correct_weights(self,
                         net: float,
                         delta: int,
                         xs: list[int]) -> None:
        for i, _ in enumerate(self.vs):
            self.vs[i] += delta * self.tf.f_der(net) * self.rbf_neurons[i].compute_set(xs[1:-1])

    def learn_epoch(self, xs_sets: list[list[int]]) -> None:
        for i, xs in enumerate(xs_sets):
            print(f'offer {xs[:-1]}')
            y, out, net = self.predict(xs[:-1])
            self._correct_weights(net, xs[-1] - y, xs)


def test_sets(net: Net, xs_sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list(), 0
    for xs in xs_sets:
        net_ans, *_ = net.predict(xs[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == xs[-1] else 1
    return answers, mistakes


def learn(net: Net,
          sets: list[list[int]],
          learn_indexes: set[int],
          epoch_limit=None) -> (bool, list[list]):
    epochs = list()
    learn_sets = [sets[i] for i in learn_indexes]

    # learning which are more:  0s or 1s ?
    ones = sum(map(lambda x: x[-1], sets), start=0)
    centers_indexes_ones = set[int]()
    for i, s in enumerate(sets):
        if not s[-1] == 1:
            continue
        centers_indexes_ones.add(i)

    # setting RBF-neurons
    if not ones == len(sets) - ones:
        centers = {i for i in (
            centers_indexes_ones if (ones < len(sets) - ones) else set(range(len(sets))) - centers_indexes_ones)}
        net.set_rbf_neurons([Net.RBFNeuron(list(), True)] + [Net.RBFNeuron(sets[i][1:-1], False)
                                                             for i in sorted(centers)])

    print(net.rbf_neurons)

    j = 0
    while True:
        if epoch_limit is not None and j == epoch_limit:
            return False, None
        net.learn_epoch(learn_sets)
        j += 1
        answers, mistakes = test_sets(net, sets)
        epochs.append([j, list(map(lambda x: x * net.norm, net.vs.copy())), answers.copy(), mistakes])
        print(epochs[-1])
        if mistakes == 0:
            return True, epochs
