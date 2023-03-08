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
    def __init__(self,
                 tf: TF,
                 weights_num: int,
                 norm: float,
                 name: str = None):
        self.weights: list[float] = [0] * weights_num
        self.norm = norm
        self.tf = tf
        self.name = name

    def predict(self, args: list[int]) -> (int, float, float):
        net = sum([norm_counter * args[i] for i, norm_counter in enumerate(self.weights)]) * self.norm
        return *self.tf.y(net), net

    def _correct_weights(self,
                         net: float,
                         delta: int,
                         xs: list[int]):
        for i, _ in enumerate(self.weights):
            self.weights[i] += delta * self.tf.f_der(net) * xs[i]

    def learn_epoch(self, xs_sets: list[list[int]]):
        for i, xs in enumerate(xs_sets):
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

    j = 1
    while True:
        net.learn_epoch(learn_sets)
        answers, mistakes = test_sets(net, sets)
        epochs.append([j, list(map(lambda x: x * 0.3, net.weights.copy())), answers.copy(), mistakes])
        if mistakes == 0:
            return True, epochs
        if epoch_limit is not None and j > epoch_limit:
            return False, None
        j += 1
