import sys

from bp_net import BPNet
from texttable import Texttable


def display_net(logs: list[list],
                to_file: bool = False,
                file_name: str = f'new-table') -> None:
    def modify_lists(logs_list: list[list]) -> list[list]:
        for i, log in enumerate(logs_list):
            log[1] = ''.join(map('{: 6.2f}'.format, log[1]))
        return logs_list

    logs = modify_lists(logs)
    logs.insert(0, ['k', 'y', 'E'])

    t = Texttable()
    t.set_chars(['—', '|', '+', '—'])
    t.set_cols_dtype(['i', 't', 't'])
    t.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)
    t.add_rows(logs)

    if not to_file:
        print(t.draw())
        return
    with open(f'{file_name}.log', 'w') as logger:
        logger.write(t.draw())


# Lagov N—J—M (2-1-2) x = (1, 1, -1) t = (2, -1)

def main():
    n, j, m = 2, 1, 2
    xs_input = [1, 1, -1]
    t = [0.2, -0.1]
    norm = 1
    e_th = 10 ** (-3)

    net = BPNet(n, j, m, norm, t)
    status, logs = net.learn(xs_input, e_th, 10000)

    display_net(logs, to_file=True, file_name='epochs-table')


if __name__ == '__main__':
    main()
