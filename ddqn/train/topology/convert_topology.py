import sys
import numpy as np


def read_topology_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    return lines


def write_topology_file(filename, lines, node):
    with open(filename, 'w+') as f:
        f.write(node + "\n")
        for line in lines:
            converted_line = line.split(",")
            f.write(' '.join(converted_line))


def save_topology(filename, matrix, node):
    with open(filename, 'w+') as f:
        for row in matrix:
            f.write(','.join(map(str, row)) + "\n")


def mapping_topology_to_matrix(topology, node):
    """
    Mapping topology to matrix
    ex)
        max link:6(node 4개)
        (0, 1, 3, 4) --> 2 dimension matrix
    """
    topology = tuple(map(int, topology.strip()[1:-1].split(",")))

    size = int(node * (node - 1) / 2)
    link_list = [0 for _ in range(0, size)]
    matrix = [[0 for _ in range(node)] for _ in range(node)]
    for link in topology:
        link_list[link] = 1
    # link info = [1, 1, 0, 1, 1, 0, 0]
    # print(link_list)

    # Here convert info to matrix
    # 비 정방형 배열로 생성
    s = 0
    temp = []
    for i in range(node - 1, 0, -1):
        r = []
        for j in range(i):
            r.append(link_list[s])
            s += 1
        temp.append(r)
    # print(temp)

    # 정방형 배열로 바꾸기 위해서 앞자리에 0으로 다 채움
    for i, t in enumerate(temp):
        done = True
        while done:
            if len(t) < node:
                t.insert(0, 0)
            else:
                done = False
    temp.append([0 for _ in range(node)])

    # print(temp)
    # 대칭이 다르면 서로 1이 들어가야 하기 때문에 다르면 1, 같으면 0
    for row in range(len(temp)):
        for col in range(len(temp[0])):
            if temp[row][col] != temp[col][row]:
                matrix[row][col] = 1
                matrix[col][row] = 1
            else:
                matrix[row][col] = 0
                matrix[col][row] = 0

    # print(matrix)
    matrix = np.array(matrix)

    return matrix


def main():
    # node = int(sys.argv[1])
    # density = int(sys.argv[2])
    node = 20
    density = 3
    filename = "pre_topology" + str(node) + "_" + str(density) + ".txt"

    lines = read_topology_file(filename)
    for i, topology in enumerate(lines):
        matrix = mapping_topology_to_matrix(topology, node)
        converted_filename = "topology" + str(node) + "_" + str(density) + ".txt"
        save_topology(converted_filename, matrix, node)


if __name__ == "__main__":
    main()


