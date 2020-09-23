# -*- coding: utf-8 -*-
from collections import deque

# import tensorflow as tf
import numpy as np

import policy
import env
from utils import *

config = load_config()
network_size = int(config["NETWORK_SIZE"])
input_size = network_size ** 2
output_size = network_size ** 2

topology_name = config["TOPOLOGY_NAME"]
gamma = float(config["GAMMA"])
max_episodes = int(config["MAX_EPISODES"])
max_replay_buffer = int(config["MAX_REPLAY_BUFFER"])
train_frequency = int(config["TRAIN_FREQUENCY"])
mini_batch_size = int(config["MINI_BATCH_SIZE"])
model_save_frequency = int(config["MODEL_SAVE_FREQUENCY"])


def ddqn_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        target_action1 = action
        target_action2 = int(action / network_size) + int(int(action % network_size) * network_size)

        if done:
            Q[0, target_action1] = reward
            Q[0, target_action2] = reward
        else:
            Q[0, target_action1] = reward + gamma * targetDQN.predict(next_state)[
                0, np.argmax(mainDQN.preprocess_predict(next_state))]
            Q[0, target_action2] = Q[0, target_action1]

        # x_stack 쌓기 위해 [1, 100] 으로 변환
        trans_state = np.reshape(state, [1, input_size])

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, trans_state])

    return mainDQN.update(x_stack, y_stack)


def main():
    replay_buffer = deque()
    train_cases = load_data("train")

    with tf.Session() as sess:

        mainDQN = policy.DQN(sess, network_size, input_size, output_size, topology_name, name="main")
        targetDQN = policy.DQN(sess, network_size, input_size, output_size, topology_name, name="target")
        
        ratios = tf.placeholder(tf.float32, [None])
        max_qs = tf.placeholder(tf.float32, [None])
        rewards = tf.placeholder(tf.float32, [None])
        losses = tf.placeholder(tf.float32, [None])
        tf.summary.scalar('avg.ratios/ep10', tf.reduce_mean(ratios))
        tf.summary.scalar('avg.max_q/ep10', tf.reduce_mean(max_qs))
        tf.summary.scalar('avg.reward/ep10', tf.reduce_mean(rewards))
        tf.summary.scalar('avg.losses/ep10', tf.reduce_mean(losses))

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter('./logs', sess.graph)
        summary_merged = tf.summary.merge_all()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        episode_10_reward_list = []
        episode_10_loss_list = []
        episode_10_ratio_list = []
        episode_10_max_qs_list = []

        for episode in range(0, max_episodes):
            environment = env.ENV(network_size, topology_name)

            e = 2. / ((episode / 10) + 1)
            done = False
            case = random.choice(train_cases)
            source, destination = extract_info(case)
            state = environment.reset(source, destination)
            hop = 0
            max_q = 0
            episode_reward = []

            while not done:
                # exploration
                if np.random.rand(1) < e:
                    action = environment.extract_action()
                # greedy
                else:
                    pred_q = mainDQN.preprocess_predict(state)
                    action = np.argmax(pred_q)
                    max_q += np.amax(pred_q)

                next_state, done, reward = environment.step(action)
                hop = hop + 1

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > max_replay_buffer:
                    replay_buffer.popleft()

                episode_reward.append(reward)
                state = next_state

                if done:
                    episode_avg_max_q = max_q / float(hop)
                    episode_10_max_qs_list.append(episode_avg_max_q)
 
            member_cnt = len(destination)
            episode_ratio = float(hop / member_cnt)
            episode_10_ratio_list.append(episode_ratio)
            episode_10_reward_list.append(sum(episode_reward))
            
            # print("destination: {}, episode_sum_reward: {}, episode_avg_reward: {}".
            #       format(len(destination), sum(temp_reward), int(sum(temp_reward) / len(destination))))

            if episode % train_frequency == 0 and episode is not 0:
                for _ in range(50):
                    mini_batch = random.sample(replay_buffer, mini_batch_size)
                    loss, _ = ddqn_train(mainDQN, targetDQN, mini_batch)
                    episode_10_loss_list.append(loss)
                summary = sess.run(summary_merged, feed_dict={ratios: episode_10_ratio_list, max_qs: episode_10_max_qs_list, rewards: episode_10_reward_list, losses: episode_10_loss_list})
                writer.add_summary(summary, episode)

                print("episode: {} | avg10.loss: {} | avg10.max_q: {}".format(episode, np.mean(np.array(episode_10_loss_list)), np.mean(np.array(episode_10_max_qs_list))))
                episode_10_ratio_list = []
                episode_10_max_qs_list = []
                episode_10_reward_list = []
                episode_10_loss_list = []

                # Update target network
                sess.run(copy_ops)

            if episode % model_save_frequency == 0:
                mainDQN.check_save(episode)


if __name__ == "__main__":
    main()

