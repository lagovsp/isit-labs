import json
import sys


def main():
    assert len(sys.argv) == 2

    with open(sys.argv[1], 'r', encoding='cp1251') as reader:
        array = json.load(reader)

        areas = set[str]()
        with open('objects.dat', 'w') as ob_writer:
            for item in array:
                coords = item['geoData']["coordinates"]
                ob_writer.write(F'{item["Name"]}&{coords[1]}&{coords[0]}&{item["AdmArea"]}\n')
                areas.add(item["AdmArea"])

        print('DISTRICTS')
        print('\n'.join(areas))


if __name__ == '__main__':
    main()
