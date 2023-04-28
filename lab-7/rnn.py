import numpy as np

SAVED_IMAGES = [
    [1] * 6 + [-1, 1] * 2 + [1, -1] + [1] * 3,
    ([1] + [-1] * 4) * 2 + [1] * 5,
    [1] * 3 + [-1] * 2 + [-1, -1, 1, -1, -1] + [1] * 5,
]


class RNN:
    def __init__(self, images) -> None:
        self.images = images
        self.l = len(images)
        self.k = len(images[0])
        self.weights = self._set_weights(images)
        self.prev_y = ([0] * self.k) * 4
        self.net = [0] * self.k
        self.eps = 0
        self._pretty_print_weights_matrix()

    def _clear(self) -> None:
        self.prev_y = [[0] * self.k, [0] * self.k, [0] * self.k, [0] * self.k]
        self.net = [0] * self.k
        self.eps = 0

    def _set_weights(self, images) -> list:
        weights = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                for k in range(self.l):
                    weights[i][j] += images[k][i] * images[k][j]

        return list(weights)

    def _pretty_print_weights_matrix(self) -> None:
        print('W:')
        print('\n'.join([''.join([str(int(cell)).rjust(4, ' ') for cell in row]) for row in self.weights]))

    def tf(self, i) -> None:
        if self.net[i] > 0:
            self.prev_y[3][i] = 1
        elif self.net[i] < 0:
            self.prev_y[3][i] = -1

    def move_y(self) -> None:
        self.prev_y[0], self.prev_y[1], self.prev_y[2] = \
            self.prev_y[1].copy(), self.prev_y[2].copy(), self.prev_y[3].copy()

    def check_cycling(self, image) -> (int, str):
        if self.prev_y[0] == self.prev_y[2] and self.prev_y[1] == self.prev_y[3]:
            return 1, '2 IMAGES CYCLING'
        elif image == self.prev_y[3]:
            return 2, 'INPUT = OUTPUT'
        elif self.prev_y[3] == self.prev_y[2]:
            return 3, "INPUT = OUTPUT V2"
        return 0, ''

    def sync_mode(self) -> None:
        for i in range(self.k):
            self.net[i] = 0
            for j in range(self.k):
                if j == i: continue
                self.net[i] += self.weights[j][i] * self.prev_y[3][j]
        for i in range(self.k):
            self.tf(i)

    def async_mode(self) -> None:
        for i in range(self.k):
            self.net[i] = 0
            for j in range(self.k):
                if j == i: continue
                self.net[i] += self.weights[j][i] * self.prev_y[3][j]
            self.tf(i)

    def _print_epoch(self) -> None:
        print(F'EP{self.eps} Y = (', end='')
        for i in range(len(self.prev_y[3])):
            print(self.prev_y[3][i], end=", " * (i != len(self.prev_y[3]) - 1))
        print(')')

    def recover_image(self, image, mode) -> bool:
        self._clear()
        self.prev_y[3] = image.copy()
        while True:
            self.eps += 1
            if mode.upper() == 'SYNC':
                self.sync_mode()
            elif mode.upper() == 'ASYNC':
                self.async_mode()

            self._print_epoch()
            for i in range(len(self.images)):
                if self.images[i] == self.prev_y[3]:
                    print('Y\' = (', end='')
                    for k in range(len(self.prev_y[3])):
                        print(image[k], end=", " * (k != len(image) - 1))
                    print(')')
                    return True

            check_result, check_mes = self.check_cycling(image)
            if check_result:
                print('Y\' = (', end='')
                for k in range(len(self.prev_y[3])):
                    print(image[k], end=', ' * (k != len(image) - 1))
                print(F')\nUNABLE TO IDENTIFY IMAGE\n{check_mes}')
                return False

            self.move_y()
