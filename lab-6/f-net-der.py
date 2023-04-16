import sys

from test_2 import f, f_der


def main():
    net = float(sys.argv[1])
    print(F'F(NET) = {f(net)}')
    print(F'F\'(NET) = {f_der(net)}')


if __name__ == '__main__':
    main()
