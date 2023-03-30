from configure import *


def main():
    with open(f'truthtable-v{VAR}.dat', 'w') as logger:
        logger.write(truthtable(bf, ARG_NUM))


if __name__ == '__main__':
    main()
