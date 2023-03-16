import sys
import matplotlib.pyplot as plt

from function_predictive_net import *


def main():
    norm = 0.1 if len(sys.argv) < 2 else float(sys.argv[1])
    p = 4 if len(sys.argv) < 3 else int(sys.argv[2])
    es = list[float]()
    ms = list[int]()

    m_min, m_max, m_mult = 1, 50, 10
    for i in range(m_min, m_max + 1):
        m = i * m_mult
        net = FPNet(p, norm)
        _, e = net.learn(XS_LEARN, m)
        es.append(e)
        ms.append(m)

    plt.title(F'v — {VAR}, n — {norm}, p — {p}')
    plt.xlabel('M')
    plt.ylabel('e')

    plt.plot(ms, es, marker=',', color='C1', label='e(M)')
    plt.legend()

    m_range_str = f'{m_min * m_mult}-{m_max * m_mult}'
    plt.gcf().savefig(
        F'am-v{VAR}-({F_STR.split("=")[-1].replace(" ", "")})-m{m_range_str}-n{str(norm).replace(".", "-")}-p{p}.png',
        dpi=500)


if __name__ == '__main__':
    main()
