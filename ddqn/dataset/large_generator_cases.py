import sys
import random

import numpy as np
import scipy.stats as ss
from tqdm import tqdm


def open_file(file_name):
    return open(file_name, 'a')


def normal_distribution_probability(scale_size, node):
    scale = scale_size
    link_range = int(node/2) - 1
    x = np.arange((-1) * link_range, link_range)
    xU, xL = x + 1, x - 1
    prob = ss.norm.cdf(xU, scale=scale) - ss.norm.cdf(xL, scale=scale)
    # normalize the probabilities so their sum is 1
    prob = prob / prob.sum()

    return np.array(prob)


def get_number_of_cases(length, cases):
    cnt = 0
    for case in cases:
        if len(case) == int(length):
            cnt += 1

    return cnt


def create_case(destination_length, source, node):
    candidate_nodes = list(range(0, node))
    candidate_nodes.remove(source)
    random_selected_case = random.sample(candidate_nodes, destination_length)
    
    return random_selected_case


def generator_case_of_node(source, node, scale_size, data_size):
    #  return 각 source 별 A
    destination = list(range(0, node))
    destination.remove(source)
    length_list = list(range(2, node))

    prob = normal_distribution_probability(scale_size, node)
    prob = [p for p in prob]

    case_size = int(data_size / node)
    length_of_destinations = [int(idx) for idx in np.random.choice(length_list, size=case_size, p=prob)]

    number = [length_of_destinations.count(length) for length in range(2, node)]
    # plt.plot(np.array(np.arange(2, node, 1)), number)
    # plt.show()

    count_dict = {}
    for i, count in enumerate(number, start=2):
        count_dict[i] = count
    if count_dict[node-1] != 1:
        count_dict[node-1] = 1

    total_case_set = list()
    for destination_length, count in count_dict.items():
        if count == 0:
            continue
        print("destination_length: {} | count: {}".format(destination_length, count))
        done = False
        case_set = set()

        while not done:
            case = create_case(destination_length, source, node)
            case.sort()
            case = tuple(case)
            case_set.add(case)
            if len(case_set) == count:
                done = True

        total_case_set.extend(case_set)
    random.shuffle(total_case_set)
    
    return total_case_set


def save_file(node):
    for source in range(0, node):
        print("source: {}\n".format(source))
        generator_case_of_node(source)


def main():
    node = int(sys.argv[1])
    scale_size = int(sys.argv[2])
    data_size = int(sys.argv[3])

    train_file_name = "train_set.txt"
    test_file_name = "test_set.txt"
    train_set_ratio = 0.8
    all_train_set = []
    all_test_set = []

    for source in tqdm(range(0, node)):
        print("source: {}\n".format(source))
        destination_set = generator_case_of_node(source, node, scale_size, data_size)
        random.shuffle(destination_set)

        length = len(destination_set)
        print("length :{}\n".format(length))
        length_train_set = int(length * train_set_ratio)

        source_destination_set = []
        for d_set in destination_set:
            temp = [str(i) for i in d_set]
            temp = ', '.join(temp)
            source_destination_set.append("(" + temp + ")*"+str(source))

        train_set = source_destination_set[:length_train_set]
        # print("len_train_set:", len(train_set))
        test_set = source_destination_set[length_train_set:]
        # print("len_test_set:", len(test_set))
        all_train_set += train_set
        all_test_set += test_set

    random.shuffle(all_train_set)
    random.shuffle(all_test_set)

    train_f = open_file(train_file_name)
    for i, case in enumerate(all_train_set):
        temp = str(i+1) + "*" + case
        train_f.write(temp+"\n")

    train_f.close()

    test_f = open_file(test_file_name)
    for i, case in enumerate(all_test_set):
        temp = str(i+1) + "*" + case
        test_f.write(temp+"\n")

    test_f.close()


if __name__ == "__main__":
    main()
