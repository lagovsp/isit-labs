import sys
import matplotlib.pyplot as plt
from function_predictive_net import *


def main():
    p = int(sys.argv[1])  # recommended 4
    norm = float(sys.argv[2])  # 0 < norm <= 1
    m_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000  # recommended to leave empty

    net = FPNet(p, norm)
    m, e = net.learn(XS_LEARN, m_limit)
    predicted = net.predict_next_xs(XS_LEARN[-net.p:], len(TS_PREDICT))

    plt.title(F'v — {VAR}, n — {norm}, M — {m}/{m_limit}, p — {p}, e — {round(e, 4)}')
    plt.xlabel('t')
    plt.ylabel('x')

    plt.plot(TS_LEARN, XS_LEARN, marker=',', color='C1', label='LEARN')
    plt.plot(TS_PREDICT, XS_PREDICT, marker='s', color='C2', label=F'{F_STR}')
    plt.plot(TS_PREDICT, predicted[net.p:], marker='3', color='C3', label='PREDICTION')

    plt.legend()
    plt.gcf().savefig(
        F'v{VAR}-({F_STR.split("=")[-1].replace(" ", "")})-m{m}-n{str(norm).replace(".", "-")}-p{p}.png',
        dpi=500)


if __name__ == '__main__':
    main()
