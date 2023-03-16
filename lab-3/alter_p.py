import sys
import matplotlib.pyplot as plt

from function_predictive_net import *


def main():
    norm = 0.1 if len(sys.argv) < 2 else float(sys.argv[1])
    m_limit = 100 if len(sys.argv) < 3 else int(sys.argv[2])
    es = list[float]()
    ps = list[int]()

    p_min, p_max = 1, 19
    for p in range(p_min, p_max + 1):
        net = FPNet(p, norm)
        _, e = net.learn(XS_LEARN, m_limit=m_limit)
        es.append(e)
        ps.append(p)

    plt.title(F'v — {VAR}, n — {norm}, M — {m_limit}')
    plt.xlabel('p')
    plt.ylabel('e')

    plt.plot(ps, es, marker=',', color='C1', label='e(p)')
    plt.legend()

    plt.gcf().savefig(
        F'ap-v{VAR}-({F_STR.split("=")[-1].replace(" ", "")})-n{str(norm).replace(".", "-")}-p{p_min}-{p_max}.png',
        dpi=500)


if __name__ == '__main__':
    main()
