#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:17:39 2022

@author: Jorgen Svane

"""

import numpy as np
from gym.spaces import Discrete, Box

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.ppo import PPO

from ray.rllib.examples.env.random_env import RandomEnv
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()


""" Building blocks for Impala deep Residual Model """

def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1] == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    # print(f"The shape of x is {x.shape}")
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x

""" ------------------------------------------------------------------- """

""" Impala Forward """

def build_model(obs_space,num_outputs):

    depths = [16, 32, 32]

    inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
    scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

    x = scaled_inputs
    for i, depth in enumerate(depths):
        x = conv_sequence(x, depth, prefix=f"seq{i}")

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.ReLU()(x)
    res_cnn_out = tf.keras.layers.Dense(units=num_outputs, activation="relu", name="hidden")(x)
    # logits = tf.keras.layers.Dense(units=num_outputs, activation="softmax", name="pi")(x)
    # value = tf.keras.layers.Dense(units=1, name="vf")(x)
     
    return tf.keras.Model(inputs=inputs, outputs=res_cnn_out, name="custom_tf_model")

""" ------------------------------------------------------------------- """

# The custom model that will be wrapped by an LSTM.
class MyCustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        # print(f"The shape of obs_space is {obs_space.shape}")
        lstm_shape = list(obs_space.shape)
        lstm_shape.insert(0,-1)
        self.lstm_shape = lstm_shape # for reshaping obs_flat below 
        # print(f"The lstm_shape is {self.lstm_shape}")
        self.num_outputs = 256 # number of outputs from impala
        self._last_batch_size = None
        self.impala = build_model(obs_space,self.num_outputs)

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        # print(f"The shape of obs is {obs.shape}")
        obs_visual = tf.reshape(obs,self.lstm_shape)
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        # Return impala out (and empty states).
        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        return self.impala(obs_visual), []

    def value_function(self):
        return tf.zeros(shape=(self._last_batch_size,))


if __name__ == "__main__":
    
    cnn_shape = (4, 4, 3)
    
    # Register the above custom model.
    ModelCatalog.register_custom_model("my_model", MyCustomModel)
    
    def env_creator(config):
        env = RandomEnv(config=config)
        return env

    register_env("RandomEnv", env_creator)
    
    config={
        "env": "RandomEnv",
        "framework": "tf2",
        "eager_tracing": True,
        # "eager": True,
        "model": {
            "custom_model": "my_model",
            # Auto-wrap the custom(!) model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 20,
            # Size of the LSTM cell.
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": True,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": True
        },
        # "vf_share_layers": True,
        "num_workers": 0,  # no parallelism
        "env_config": {
            "action_space": Discrete(2),
            # Test a simple Tuple observation space.
            "observation_space": Box(
                0.0, 1.0, shape=cnn_shape, dtype=np.float32)
        }
    }
    
    
    algo = PPO(config=config)
        
    for _ in range(1):
        print(algo.train())




