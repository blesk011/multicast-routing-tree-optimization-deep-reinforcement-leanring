import random
import numpy as np
import scipy.stats as ss
import networkx as nx
from tqdm import tqdm

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
전체 과정
# 0~45시그마nCi를 하게 되면 너무 많은 토폴로지가 생성되기 때문에 원하는 개수만큼만 분포확률로 치환해서 샘플링한다.
1. 샘플링하고 싶은 개수를 정한다.(100,000개 == 샘플개수)
2. 모집단은 정규분포를 따르기 때문에 샘플 또한 정규분포를 따르도록 하기 위해서 스케일을 포함하여 정규분포를 따르는 확률값을 구한다.
3. 정해진 확률값에 따라서 샘플개수에서 샘플링한다. --> 3번을 거치고 나면 링크개수 각각에 대한 토폴로지 개수가 정해진다. <-- 여기까진 이전에 해본적 있음.
4. 링크 개수에 따라서 랜덤으로 링크를 선택 후 중복되지 않는다면 토폴로지 집합에 추가한다. <-- 여기가 제일 중요함. 각 인덱스를 링크로 어떻게 매핑하는지가 중요함. 
5. 토폴로지 집합을 loop 돌면서 real 토폴로지 데이터를 생성한다.
"""


def read_topology(filename):
    adj_matrix = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            links = [int(link) for link in line.split(',')]
            adj_matrix.append(links)

    matrix = np.array(adj_matrix)
    return matrix


def save_topology(topology_set, node):
    with open("topology_set.txt", 'a+') as f1:
        for i, topology in enumerate(topology_set):
            f1.write("(" + ",".join(map(str, topology)) + ")\n")


def mapping_topology_to_matrix(topology, node):
    """
    Mapping topology to matrix
    ex)
        max link:6(node 4개)
        (0, 1, 3, 4) --> 2 dimension matrix
    """
    size = int(node*(node-1) / 2)
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


def convert_topology_to_G(topology, node):
    """
    Convert topology to graph G
    ex)
        (0, 1, 3, 4) --> matrix -> graph G
    """
    matrix = mapping_topology_to_matrix(topology, node)
    # matrix = np.array(matrix)
    G = nx.from_numpy_matrix(matrix)

    return G


def convert_numpy_to_graph(array):
    g = nx.from_numpy_matrix(array)

    return g


def validation_subgraph(G, node):
    """
    Check whether graph has disconnected
    method: bfs search
    """
    graph = [[] for i in range(1, int(node)+1)]
    node_list = list(range(0, node))
    count = 0

    edges = list(G.edges)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    # print("graph: {}".format(graph))

    visit = list()
    queue = list()

    queue.append(0)
    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.extend(graph[node])

    for node in node_list:
        if node not in visit:
            count += 1

    if count > 0:
        return False
    else:
        return True


def validation_isolation_node(G, node):
    """
    Check whether isolation of node is exist
    """
    graph = [[] for i in range(1, int(node)+1)]
    count = 0

    edges = list(G.edges)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for i in range(0, int(node)):
        if (len(graph[i]) == 0):
            # print("i: {}".format(i))
            count += 1

    if count > 0:
        return False
    else:
        return True


def create_topology(network_size, link_size):
    possible_link_size = int(network_size*(network_size-1)/2)
    link_list = list(range(0, possible_link_size))

    random_selected_links = random.sample(link_list, link_size)
    # print("선택된 링크들: {}".format(random_selected_links))

    return random_selected_links


def get_prob_by_size_of_link(node, sample_size):
    possible = int((node*(node-1))/2)
    link_range = list(range(0, possible))

    prob = normal_distribution_probability(possible)
    prob = [p for p in prob]
    
    link_sizes = [int(idx) for idx in np.random.choice(link_range, size=sample_size, p=prob)]
    result = [link_sizes.count(link) for link in range(1, possible+1)]

    dict = {}
    for i, link_size in enumerate(result):
        dict[i+1] = link_size
    dict[possible] = 1

    return dict


def normal_distribution_probability(link_range):
    link_range = int(link_range/2)
    scale = 50
    x = np.arange((-1)*link_range, link_range)
    xU, xL = x + 1, x - 1
    prob = ss.norm.cdf(xU, scale=scale) - ss.norm.cdf(xL, scale=scale)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1

    return np.array(prob)


def main():
    sample_size = 15000
    network_size = 20
    total_topology_set = list()
    link_sizes_dict = get_prob_by_size_of_link(network_size, sample_size)

    for link_size, count in tqdm(link_sizes_dict.items()):
        if link_size < network_size:
            continue
        done = True
        topology_set = set()
        while done:
            '''Create Topology'''
            topology = create_topology(network_size, link_size)
            topology.sort()
            topology = tuple(topology)
            graph = convert_topology_to_G(topology, network_size)

            ''' validation topology'''
            if validation_isolation_node(graph, network_size) and validation_subgraph(graph, network_size):
                topology_set.add(topology)
            if len(topology_set) >= count:
                done = False

        total_topology_set.extend(topology_set)
    save_topology(total_topology_set, 100)


if __name__ == "__main__":
    main()
    """
    matrix = mapping_topology_to_matrix((0, 1), 4)
    print(matrix)
    """

    """
    result = convert_topology_to_G((0, 1, 3, 4), 4)
    print(result.edges)

    check = validation_subgraph(result, 4)
    print(check)
    """
    """
    matrix = read_topology("topology_4.txt")
    g = convert_numpy_to_graph(matrix)

    # result = convert_topology_to_G((0, 5), 4)
    # print(result.edges)

    check = validation_subgraph(g, 20)
    print(check)
    """
