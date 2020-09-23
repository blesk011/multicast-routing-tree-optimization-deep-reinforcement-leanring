import sys
import json
import random
from ast import literal_eval

import tensorflow as tf


def extract_info(case):
    nodes = case.split("*")
    source = int(nodes[2])
    destination = literal_eval(nodes[1])

    return source, destination


def load_data(data_type):
    if data_type == "train":
        with open("../dataset/train_set.txt", 'r') as f:
            data = f.readlines()
    elif data_type == "test":
        with open("../dataset/test_set.txt", 'r') as f:
            data = f.readlines()

    return data


def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):

    op_holder =[]

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def load_config():
    with open("./config/config.json") as f:
        config = json.load(f)

    return config

