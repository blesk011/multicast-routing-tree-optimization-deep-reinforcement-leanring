import sys
import random


def write_file(topos, filename):
    with open(filename, 'a+') as f:
        for topo in topos:
            f.write(topo)


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def main():
    network_size = 20
    density = 3
    # network_size = int(sys.argv[1])
    # density = int(sys.argv[2])

    filename = "topology_set.txt"
    save_filename = "pre_topology" + str(network_size) + "_" + str(density) + ".txt"

    link_size = int(int(network_size * int(network_size-1) / 2) * float(density / 10.0))
    lines = read_file(filename)

    topos = []
    for line in lines:
        pre_line = list(map(int, line.strip()[1:-1].split(",")))
        if len(pre_line) == link_size:
            topos.append(line)
    topo = random.choice(topos)
    write_file(topo, save_filename)


if __name__ == "__main__":
    main()
