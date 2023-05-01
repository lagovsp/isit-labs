from dataclasses import dataclass


@dataclass
class Cluster:
    id: int
    name: str


@dataclass
class Object:
    id: int
    name: str
    x: float
    y: float
    original_district: str
    analyzed_district: str


def main():
    pass


if __name__ == '__main__':
    main()
