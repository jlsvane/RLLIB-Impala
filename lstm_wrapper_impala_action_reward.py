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
        
        print(f"The shape of obs_space is {obs_space.shape}")
        lstm_shape = list(obs_space.shape)
        lstm_shape.insert(0,-1)
        self.lstm_shape = lstm_shape # for reshaping obs_flat below 
        print(f"The lstm_shape is {self.lstm_shape}")
        self.num_outputs = 256 # number of outputs from impala
        self._last_batch_size = None
        self.impala = build_model(obs_space,self.num_outputs)

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        print(f"The shape of obs is {obs.shape}")
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
            "lstm_use_prev_reward": True,
            "vf_share_layers": True
        },
        "vf_share_layers": True,
        "num_workers": 0,  # no parallelism
        "env_config": {
            "action_space": Discrete(3),
            # Test a simple Tuple observation space.
            "observation_space": Box(
                0.0, 1.0, shape=cnn_shape, dtype=np.float32)
        }
    }
    
    
    algo = PPO(config=config)
        
    for _ in range(1):
        print(algo.train())
    
    algo.get_policy().model.impala.summary()
    
    algo.get_policy().model._rnn_model._name = "impala_deep_residual_lstm_head"
    algo.get_policy().model._rnn_model.summary()
    
"""

MODELS SUMMARY (2 models):


Model: "custom_tf_model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 observations (InputLayer)      [(None, 4, 4, 3)]    0           []                               
                                                                                                  
 tf.cast_1 (TFOpLambda)         (None, 4, 4, 3)      0           ['observations[0][0]']           
                                                                                                  
 tf.math.truediv_1 (TFOpLambda)  (None, 4, 4, 3)     0           ['tf.cast_1[0][0]']              
                                                                                                  
 seq0_conv (Conv2D)             (None, 4, 4, 16)     448         ['tf.math.truediv_1[0][0]']      
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 2, 2, 16)    0           ['seq0_conv[0][0]']              
                                                                                                  
 re_lu_13 (ReLU)                (None, 2, 2, 16)     0           ['max_pooling2d_3[0][0]']        
                                                                                                  
 seq0_block0_conv0 (Conv2D)     (None, 2, 2, 16)     2320        ['re_lu_13[0][0]']               
                                                                                                  
 re_lu_14 (ReLU)                (None, 2, 2, 16)     0           ['seq0_block0_conv0[0][0]']      
                                                                                                  
 seq0_block0_conv1 (Conv2D)     (None, 2, 2, 16)     2320        ['re_lu_14[0][0]']               
                                                                                                  
 tf.__operators__.add_6 (TFOpLa  (None, 2, 2, 16)    0           ['seq0_block0_conv1[0][0]',      
 mbda)                                                            'max_pooling2d_3[0][0]']        
                                                                                                  
 re_lu_15 (ReLU)                (None, 2, 2, 16)     0           ['tf.__operators__.add_6[0][0]'] 
                                                                                                  
 seq0_block1_conv0 (Conv2D)     (None, 2, 2, 16)     2320        ['re_lu_15[0][0]']               
                                                                                                  
 re_lu_16 (ReLU)                (None, 2, 2, 16)     0           ['seq0_block1_conv0[0][0]']      
                                                                                                  
 seq0_block1_conv1 (Conv2D)     (None, 2, 2, 16)     2320        ['re_lu_16[0][0]']               
                                                                                                  
 tf.__operators__.add_7 (TFOpLa  (None, 2, 2, 16)    0           ['seq0_block1_conv1[0][0]',      
 mbda)                                                            'tf.__operators__.add_6[0][0]'] 
                                                                                                  
 seq1_conv (Conv2D)             (None, 2, 2, 32)     4640        ['tf.__operators__.add_7[0][0]'] 
                                                                                                  
 max_pooling2d_4 (MaxPooling2D)  (None, 1, 1, 32)    0           ['seq1_conv[0][0]']              
                                                                                                  
 re_lu_17 (ReLU)                (None, 1, 1, 32)     0           ['max_pooling2d_4[0][0]']        
                                                                                                  
 seq1_block0_conv0 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_17[0][0]']               
                                                                                                  
 re_lu_18 (ReLU)                (None, 1, 1, 32)     0           ['seq1_block0_conv0[0][0]']      
                                                                                                  
 seq1_block0_conv1 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_18[0][0]']               
                                                                                                  
 tf.__operators__.add_8 (TFOpLa  (None, 1, 1, 32)    0           ['seq1_block0_conv1[0][0]',      
 mbda)                                                            'max_pooling2d_4[0][0]']        
                                                                                                  
 re_lu_19 (ReLU)                (None, 1, 1, 32)     0           ['tf.__operators__.add_8[0][0]'] 
                                                                                                  
 seq1_block1_conv0 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_19[0][0]']               
                                                                                                  
 re_lu_20 (ReLU)                (None, 1, 1, 32)     0           ['seq1_block1_conv0[0][0]']      
                                                                                                  
 seq1_block1_conv1 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_20[0][0]']               
                                                                                                  
 tf.__operators__.add_9 (TFOpLa  (None, 1, 1, 32)    0           ['seq1_block1_conv1[0][0]',      
 mbda)                                                            'tf.__operators__.add_8[0][0]'] 
                                                                                                  
 seq2_conv (Conv2D)             (None, 1, 1, 32)     9248        ['tf.__operators__.add_9[0][0]'] 
                                                                                                  
 max_pooling2d_5 (MaxPooling2D)  (None, 1, 1, 32)    0           ['seq2_conv[0][0]']              
                                                                                                  
 re_lu_21 (ReLU)                (None, 1, 1, 32)     0           ['max_pooling2d_5[0][0]']        
                                                                                                  
 seq2_block0_conv0 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_21[0][0]']               
                                                                                                  
 re_lu_22 (ReLU)                (None, 1, 1, 32)     0           ['seq2_block0_conv0[0][0]']      
                                                                                                  
 seq2_block0_conv1 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_22[0][0]']               
                                                                                                  
 tf.__operators__.add_10 (TFOpL  (None, 1, 1, 32)    0           ['seq2_block0_conv1[0][0]',      
 ambda)                                                           'max_pooling2d_5[0][0]']        
                                                                                                  
 re_lu_23 (ReLU)                (None, 1, 1, 32)     0           ['tf.__operators__.add_10[0][0]']
                                                                                                  
 seq2_block1_conv0 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_23[0][0]']               
                                                                                                  
 re_lu_24 (ReLU)                (None, 1, 1, 32)     0           ['seq2_block1_conv0[0][0]']      
                                                                                                  
 seq2_block1_conv1 (Conv2D)     (None, 1, 1, 32)     9248        ['re_lu_24[0][0]']               
                                                                                                  
 tf.__operators__.add_11 (TFOpL  (None, 1, 1, 32)    0           ['seq2_block1_conv1[0][0]',      
 ambda)                                                           'tf.__operators__.add_10[0][0]']
                                                                                                  
 flatten_21 (Flatten)           (None, 32)           0           ['tf.__operators__.add_11[0][0]']
                                                                                                  
 re_lu_25 (ReLU)                (None, 32)           0           ['flatten_21[0][0]']             
                                                                                                  
 hidden (Dense)                 (None, 256)          8448        ['re_lu_25[0][0]']               
                                                                                                  
==================================================================================================
Total params: 106,048
Trainable params: 106,048
Non-trainable params: 0
__________________________________________________________________________________________________


######################################################################################################

Model: "impala_deep_residual_lstm_head"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 seq_in (InputLayer)            [(None,)]            0           []                               
                                                                                                  
 inputs (InputLayer)            [(None, None, 260)]  0           []                               
                                                                                                  
 h (InputLayer)                 [(None, 256)]        0           []                               
                                                                                                  
 c (InputLayer)                 [(None, 256)]        0           []                               
                                                                                                  
 tf.sequence_mask_1 (TFOpLambda  (None, None)        0           ['seq_in[0][0]']                 
 )                                                                                                
                                                                                                  
 lstm (LSTM)                    [(None, None, 256),  529408      ['inputs[0][0]',                 
                                 (None, 256),                     'h[0][0]',                      
                                 (None, 256)]                     'c[0][0]',                      
                                                                  'tf.sequence_mask_1[0][0]']     
                                                                                                  
 logits (Dense)                 (None, None, 3)      771         ['lstm[0][0]']                   
                                                                                                  
 values (Dense)                 (None, None, 1)      257         ['lstm[0][0]']                   
                                                                                                  
==================================================================================================
Total params: 530,436
Trainable params: 530,436
Non-trainable params: 0
__________________________________________________________________________________________________

Note the input dim has added 3 one_hot actions and one reward taking it from 256 to 260

"""




