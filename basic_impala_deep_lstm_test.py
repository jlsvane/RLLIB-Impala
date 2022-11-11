#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:08:50 2022

@author: Jorgen Svane

Inspired from here: https://github.com/ray-project/ray/issues/6928
and here: https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook/blob/master/Chapter08/5_scaling_deep_rl_training_using_ray_tune_rllib/custom_model.py

"""

import numpy as np

from gym.spaces import Discrete, Box
# import ray.rllib.examples.env.random_env
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

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

class CustomModel(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        # print(f"The shape of obs_space is {obs_space.shape}") #(4, 4, 3)

        self.cell_size = 256

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        """ Impala Forward """
        visual_size = np.product(obs_space.shape) # multiply all elements of tuple
        inputs = tf.keras.layers.Input(
            shape=(None, visual_size), name="visual_inputs")

        input_visual = inputs
        input_visual = tf.reshape(
            input_visual, 
            [-1, obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]])
        
        depths = [16, 32, 32]
        
        x = tf.cast(input_visual, tf.float32) #/ 255.0
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        vision_out = tf.keras.layers.Dense(units=256, activation="relu", name="to_lstm")(x)
        
        # print(f"The shape of vision_out is {vision_out.shape}") #(?, 256)
        
        """ ------------------------------------------------------------- """
        
        vision_out = tf.reshape(
            vision_out,
            [-1, tf.shape(inputs)[1], vision_out.shape.as_list()[-1]])
        
        # print(f"The shape of vision_out is {vision_out.shape}") #(?, ?, 256)
        
        # LSTM 
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size,
            return_sequences=True,
            return_state=True,
            name="lstm")(
                inputs=vision_out,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values.
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        # self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    
    cnn_shape = (4, 4, 3)
    
    ModelCatalog.register_custom_model("my_model", CustomModel)
    
    def env_creator(config):
        env = RandomEnv(config=config)
        return env

    register_env("RandomEnv", env_creator)
    
    config={
        "env": "RandomEnv",
        "framework": "tf2",
        # "eager": True,
        "model": {
            "custom_model": "my_model",
            "max_seq_len": 20, 
        },
        "vf_share_layers": True,
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