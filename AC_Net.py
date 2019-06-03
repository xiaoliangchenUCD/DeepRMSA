from __future__ import division
import numpy as np
import tensorflow as tf
from collections import deque
import random
import string
import tensorflow.contrib.slim as slim

np.random.seed(1)
tf.set_random_seed(1)

class AC_Net:

    def __init__(self, scope, trainer, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar):
        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar
        self.trainer = trainer
        
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regu_scalar, scope = None)
        
        self.w_initializer, self.b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        
        self.Input_p = tf.placeholder(tf.float32, [None, self.x_dim_p], name='policy_input')
        self.Input_v = tf.placeholder(tf.float32, [None, self.x_dim_v], name='value_input')
        
        with tf.variable_scope(self.scope):
            
            with tf.variable_scope('policy', regularizer = self.regularizer): # Policy Q-Network
                self.policy = slim.fully_connected(self.dnn(self.Input_p), self.n_actions,
                                                         activation_fn = tf.nn.softmax,
                                                         weights_initializer = self.normalized_columns_initializer(0.01),
                                                         biases_initializer = None)
       
            with tf.variable_scope('value', regularizer = self.regularizer):
                self.value = slim.fully_connected(self.dnn(self.Input_v), 1,
                                                 activation_fn = None,
                                                 weights_initializer = self.normalized_columns_initializer(1.0),
                                                 biases_initializer = None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if self.scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.regu_loss_policy = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = self.scope + '/policy'))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-6)) # 1e-6 is for preventing NaN
                lost_policy_net = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-6)*self.advantages)
                # weight of entropy: 0.01 or 0.1
                #self.loss_policy = lost_policy_net - self.entropy * 0.01 + self.regu_loss_policy
                self.loss_policy = lost_policy_net - self.entropy * 0.01
                
                self.regu_loss_value = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = self.scope + '/value'))
                loss_value_net = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                #self.loss_value = loss_value_net + self.regu_loss_value
                self.loss_value = loss_value_net

                #Get gradients from local network using local losses
                local_vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope + '/policy')
                self.gradients_policy = tf.gradients(self.loss_policy, local_vars_policy)
                self.var_norms_policy = tf.global_norm(local_vars_policy)
                grads_policy, self.grad_norms_policy = tf.clip_by_global_norm(self.gradients_policy, 40.0)
                
                local_vars_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope + '/value')
                self.gradients_value = tf.gradients(self.loss_value, local_vars_value)
                self.var_norms_value = tf.global_norm(local_vars_value)
                grads_value, self.grad_norms_value = tf.clip_by_global_norm(self.gradients_value, 40.0)
                
                #Apply local gradients to global network
                global_vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global' + '/policy')
                self.apply_grads_policy = self.trainer.apply_gradients(zip(grads_policy, global_vars_policy))
                
                global_vars_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global' + '/value')
                self.apply_grads_value = self.trainer.apply_gradients(zip(grads_value, global_vars_value))   

    def dnn(self, InputFeatures):
        # initiate
        with tf.variable_scope('first'):
            x_h = slim.fully_connected(InputFeatures, self.layer_size, activation_fn = tf.nn.elu)
        # multiple hidden
        for ii in range(self.num_layers-1):
            with tf.variable_scope('hidden_%d' % ii):
                x_h = slim.fully_connected(x_h, self.layer_size, activation_fn = tf.nn.elu)
        return x_h
    
    def normalized_columns_initializer(self, std = 1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer