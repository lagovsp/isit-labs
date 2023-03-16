import sys
import matplotlib.pyplot as plt

from function_predictive_net import *


def main():
    p = 4 if len(sys.argv) < 2 else int(sys.argv[1])
    m_limit = 500 if len(sys.argv) < 3 else int(sys.argv[2])
    es = list[float]()
    ns = list[float]()

    n_min, n_max, n_mult = 1, 20, 0.05
    for i in range(n_min, n_max + 1):
        norm = i * n_mult
        net = FPNet(p, norm)
        _, e = net.learn(XS_LEARN, m_limit=m_limit)
        es.append(e)
        ns.append(norm)

    plt.title(F'v — {VAR}, M — {m_limit}, p — {p}')
    plt.xlabel('n')
    plt.ylabel('e')

    plt.plot(ns, es, marker=',', color='C1', label='e(n)')
    plt.legend()

    n_range_str = f'{str(n_min * n_mult).replace(".", "-")}-{str(n_max * n_mult).replace(".", "-")}'
    plt.gcf().savefig(
        F'an-v{VAR}-({F_STR.split("=")[-1].replace(" ", "")})-n{n_range_str}-p{p}.png',
        dpi=500)


if __name__ == '__main__':
    main()
