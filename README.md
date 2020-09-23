# Multicast routing tree optimization with deep reinforcement learning for Software-Defined Networking

---

## How to set
```
python3 -m virtualenv venv_name (ex, python3 -m virtualenv venv)
source ./venv/bin/activate
pip3 install -r requirments.txt
```

## How to train
1. Reads configuration from ```./config/config.json```
2. Run agent

```
cd ddqn/train
vi ./config/config.json
python3 agent.py
```

### EXAMPLE JSON CONFIG
```
{
    "TOPOLOGY_NAME" : "topology20_3.txt",                       # network topology file name
    "NETWORK_SIZE" : "20",                                      # number of nodes in the network
    "GAMMA" : "0.9",                                            # discounted factor
    "MAX_EPISODES" : "200001",                                  # max episosde for agent
    "MAX_REPLAY_BUFFER" : "10000",                              # max size of memory
    "TRAIN_FREQUENCY" : "10",                                   # frequency of train upate (target network update)
    "MINI_BATCH_SIZE" : "20",                                   # size of learning batch
    "MODEL_SAVE_FREQUENCY": "1000"                              # frequency of model save
}
```

## How to visualize result
Run tensorboard 
```
cd ddqn/train
tensorboard --logdir='./folder' --port=port (ex, tensorboard --logdir='./logs' --port=99999)
```

## To do
* convert env.py into gym-based environment(openai)
* need to convert tensorflow version 1 into pytorch or tensorflow version 2
---
* [Tensorflow](https://www.tensorflow.org/)
* [DQN](https://www.nature.com/articles/nature14236)
* [Double DQN](https://arxiv.org/abs/1509.06461)
