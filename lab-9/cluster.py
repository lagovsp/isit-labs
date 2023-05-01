import math
import sys
from dataclasses import dataclass
from prettytable import PrettyTable


@dataclass
class Cluster:
    name: str
    x: float
    y: float


@dataclass
class Object:
    name: str
    x: float
    y: float
    original_district: str
    analyzed_district: str
    distance: str

    @staticmethod
    def dist(o: 'Object', c: Cluster) -> float:
        return math.sqrt((c.x - o.x) ** 2 + (c.y - o.y) ** 2)

    def __lt__(self, other: 'Object') -> bool:
        return self.name < other.name


def main():
    # adm-areas file path, objects file path
    assert len(sys.argv) == 3

    clusters = list[Cluster]()
    with open(sys.argv[1], 'r') as adms:
        for line in adms.readlines():
            name, x, y = line.split(',')
            clusters.append(Cluster(name, float(x), float(y)))

    objects = list[Object]()
    with open(sys.argv[2], 'r') as obs:
        for line in obs.readlines():
            name, x, y, district = line.rstrip('\n').split('&')
            objects.append(Object(name, float(x), float(y), district, None, None))

    for i, _ in enumerate(objects):
        closest_c = min(clusters,
                        key=lambda c: Object.dist(objects[i], c))
        objects[i].analyzed_district = closest_c.name
        objects[i].distance = Object.dist(objects[i], closest_c)

    pt = PrettyTable()
    pt.field_names = ['OBJECT', 'ANALYZED DISTRICT', 'ORIGINAL DISTRICT', 'STATUS']

    mistakes = 0
    for ob in sorted(objects):
        belongs = True if ob.original_district == ob.analyzed_district else False
        mistakes += 0 if belongs else 1
        pt.add_row([ob.name,
                    ob.analyzed_district,
                    ob.original_district,
                    '' if belongs else 'X'])

    mis_per = mistakes / len(objects) * 100
    info = F'OVERALL OBJECTS: {len(objects)}, MISTAKES: {mistakes} ({mis_per:6.2f}%)'

    with open('table.ans', 'w') as writer:
        pt.align = 'c'
        writer.write(pt.get_string())
        writer.write(info)

    print(info)


if __name__ == '__main__':
    main()
