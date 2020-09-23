import numpy as np
import random
import copy


class ENV:

    def __init__(self, network_size, topology_name):
        self.network_size = network_size
        self.selected_node = np.zeros([self.network_size], dtype=bool)
        self.selected_link = np.zeros([self.network_size**2], dtype=bool)
        # self.selected_link = [False for _ in range(self.network_size * self.network_size)]
        # self.selected_node = [False for _ in range(self.network_size)]
        self.source = -1
        self.destination = []
        self.topology_name = topology_name
        self.adj_mat = np.loadtxt("./topology/" + self.topology_name, dtype=int, delimiter=",")
        self.reward_check = []
        self.current_state = np.zeros([network_size, network_size], dtype=int)
        self.possible_link = np.zeros([network_size, network_size], dtype=int)
        # self.current_state = np.loadtxt("init_state20.txt", dtype=int, delimiter=",")
        # self.possible_link = np.loadtxt("init_state20.txt", dtype=int, delimiter=",")

    def step(self, action):
        source = int(action / self.network_size)
        destination = int(action % self.network_size)
        
        self.current_state[source, destination] = 1
        self.current_state[destination, source] = 1
        self.current_state[source, source] = 0
        self.current_state[destination, destination] = 0

        # update selected action
        self.selected_link[source * self.network_size + destination] = True
        self.selected_link[destination * self.network_size + source] = True
        self.selected_node[source] = True
        self.selected_node[destination] = True

        select_node = set(np.where(self.current_state == 1)[0])
        # select_node = list(select_node)

        # test topo
        adjacency_mat = self.adj_mat

        # 선택된 노드(i)의 가능한 목적지(j) 찾음
        for i in select_node:
            possible_dst = np.where(adjacency_mat[i] == 1)
            for j in possible_dst[0]:
                index = i * self.network_size + j
                inv_index = j * self.network_size + i
                if self.selected_node[j]:
                    self.possible_link[i][j] = 0
                    self.possible_link[j][i] = 0
                    continue
                if self.selected_link[index] is True or self.selected_link[inv_index] is True:
                    self.selected_link[index] = True
                    self.selected_link[inv_index] = True
                    self.possible_link[i][j] = 0
                    self.possible_link[j][i] = 0
                    continue
                self.possible_link[i][j] = 1
                self.possible_link[j][i] = 1

        next_state = copy.deepcopy(self.current_state)

        return next_state, self.done(next_state), self.reward(action)

    def extract_action(self):
        possible_action = []

        possible_link_mat = self.possible_link
        possible_links = np.where(possible_link_mat == 1)
        for i in range(len(possible_links[0])):
            source = possible_links[0][i]
            destination = possible_links[1][i]
            index = source * self.network_size + destination
            possible_action.append(index)
        action = random.choice(possible_action)

        self.selected_link[action] = True
        action2 = int(int(action % self.network_size)*self.network_size) + int(action/self.network_size)
        self.selected_link[action2] = True

        return action

    def reset(self, source, destination):
        self.source = source
        self.destination = destination

        self.reward_check = destination
        self.reward_check = list(self.reward_check)

        index = int(source*self.network_size) + source

        self.selected_link[index] = True
        self.selected_node[source] = True

        self.current_state[source, source] = 1
        for i in self.destination:
            self.current_state[i, i] = -1
        adjacency_mat = self.adj_mat

        possible_dst = np.where(adjacency_mat[source] == 1)

        for i in possible_dst[0]:
            self.possible_link[source][i] = 1
            self.possible_link[i][source] = 1
        
        now_state = copy.deepcopy(self.current_state)

        return now_state

    def reward(self, action):
        reward = 0
        source = int(action / self.network_size)
        destination = int(action % self.network_size)

        if destination in self.reward_check:
            reward = 1
            self.reward_check.remove(destination)
        if source in self.reward_check:
            reward = 1
            self.reward_check.remove(source)

        return reward

    def done(self, state):
        check = np.zeros([len(self.destination)], dtype=bool)

        for i, dst in enumerate(self.destination):
            if state[dst][dst] == 0:
                check[i] = True
            else:
                check[i] = False

        if all(check):
            done = True
        else:
            done = False

        return done


