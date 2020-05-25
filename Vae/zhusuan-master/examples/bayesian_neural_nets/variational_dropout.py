#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.meta_bayesian_net(scope="model", reuse_variables=True)
def var_dropout(x, n, net_size, n_particles, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    bn = zs.BayesianNet()
    h = x
    for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
        eps_mean = tf.ones([n, n_in])
        eps = bn.normal(
            'layer' + str(i) + '/eps', eps_mean, std=1.,
            n_samples=n_particles, group_ndims=1)
        h = layers.fully_connected(
            h * eps, n_out, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        if i < len(net_size) - 2:
            h = tf.nn.relu(h)
    y = bn.categorical('y', h)
    bn.deterministic('y_logit', h)
    return bn


@zs.reuse_variables(scope="variational")
def q(n, net_size, n_particles):
    bn = zs.BayesianNet()
    for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
        with tf.variable_scope('layer' + str(i)):
            logit_alpha = tf.get_variable('logit_alpha', [n_in])

        std = tf.sqrt(tf.nn.sigmoid(logit_alpha) + 1e-10)
        std = tf.tile(tf.expand_dims(std, 0), [n, 1])
        eps = bn.normal('layer' + str(i) + '/eps',
                        1., std=std,
                        n_samples=n_particles, group_ndims=1)
    return bn


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path, one_hot=False)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.concatenate([y_train, y_valid]).astype('int32')
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    n_x = x_train.shape[1]

    # Define training/evaluation parameters
    epochs = 500
    batch_size = 1000
    lb_samples = 10
    ll_samples = 100
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 3
    learning_rate = 0.001
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # placeholders
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.int32, shape=(None))
    n = tf.shape(x)[0]

    net_size = [n_x, 100, 100, 100, 10]
    e_names = ['layer' + str(i) + '/eps' for i in range(len(net_size) - 1)]

    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])

    model = var_dropout(x_obs, n, net_size, n_particles, is_training)
    variational = q(n, net_size, n_particles)

    def log_joint(bn):
        log_pe = bn.cond_log_prob(e_names)
        log_py_xe = bn.cond_log_prob('y')
        return tf.add_n(log_pe) + log_py_xe * x_train.shape[0]

    model.log_joint = log_joint

    lower_bound = zs.variational.elbo(model, {'y': y_obs},
                                      variational=variational, axis=0)
    y_logit = lower_bound.bn["y_logit"]
    h_pred = tf.reduce_mean(tf.nn.softmax(y_logit), 0)
    y_pred = tf.argmax(h_pred, 1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))

    cost = tf.reduce_mean(lower_bound.sgvb()) / x_train.shape[0]
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(cost)

    lower_bound = tf.reduce_mean(lower_bound) / x_train.shape[0]

    params = tf.trainable_variables()
    for i in params:
        print('variable name = {}, shape = {}'
              .format(i.name, i.get_shape()))

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            time_epoch = -time.time()
            indices = np.random.permutation(x_train.shape[0])
            x_train = x_train[indices]
            y_train = y_train[indices]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               is_training: True,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_accs = []
                for t in range(10):
                    x_batch = x_test[t * 1000:(t + 1) * 1000]
                    y_batch = y_test[t * 1000:(t + 1) * 1000]
                    lb, acc1 = sess.run(
                        [lower_bound, acc],
                        feed_dict={n_particles: ll_samples,
                                   is_training: False,
                                   x: x_batch, y: y_batch})
                    test_lbs.append(lb)
                    test_accs.append(acc1)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test accuracy = {}'.format(np.mean(test_accs)))
