import sys
import random
from itertools import combinations


def open_file(file_name):
    return open(file_name, 'w')


def generator_case_of_node(source, node, case_size):
    #  return 각 source 별 A
    destination = list(range(0, node))
    destination.remove(source)
    all_destination = []

    for length_of_destination in range(2, node):
        temp_destination = list(combinations(destination, length_of_destination))
        random.shuffle(temp_destination)
        all_destination += temp_destination
    random.shuffle(all_destination)
    destination_set = random.sample(all_destination, case_size)

    return destination_set


def main():
    # 전체 case 를  모두 구하고 그중 임의의 10만개 추 출
    # source, length_of_destination
    # A = 19C2 + 19C3 + 19C4 + 19C5 + ... + 19C19
    # Total case = A * 20
    # A에 서 5000개 씩 extract ==> 5000 * 20 = 100000 (10만개)
    # 5000 --> 80:20 == 4000:1000 == train:test
    """
    node = int(sys.argv[1])
    data_size = int(sys.argv[2])
    case_size = int(data_size / node)
    """
    node = 20
    data_size = 100000
    case_size = int(data_size / node)

    train_file_name = "train_set.txt"
    test_file_name = "test_set.txt"
    train_set_ratio = 0.8
    all_train_set = []
    all_test_set = []

    for source in range(0, node):
        # destination_set = generator_case_of_node_20(source)
        destination_set = generator_case_of_node(source, node, case_size)
        random.shuffle(destination_set)

        length = len(destination_set)
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
