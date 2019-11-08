from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset, save_image_collections


# decoder
@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(x_dim, z_dim, n, n_particles=1):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


# encoder
@zs.meta_bayesian_net(scope="q_net", reuse_variables=True)
def build_q_net(x, z_dim, n_z_per_x, std_noise=0):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim) + std_noise
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


def main():
    # Load MNIST
    data_path = os.path.join(conf.data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)

    # Define model parameters
    x_dim = x_train.shape[1]
    z_dim = 40

    # Build the computation graph

    # how many samples to draw from the distribution, more samples, more accuracy
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")

    # input data to feed the variational
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input),
                tf.int32)

    # batch size
    n = tf.placeholder(tf.int32, shape=[], name="n")

    # add random noise to the variance of the q_model so to
    # get more various samples when generating new digits
    std_noise = tf.placeholder_with_default(0., shape=[], name="std_noise")

    # build the model (encoder) and the q_model (variational or decoder)
    model = build_gen(x_dim, z_dim, n, n_particles)
    q_model = build_q_net(x, z_dim, n_particles, std_noise)
    variational = q_model.observe()

    # calculate ELBO
    lower_bound = zs.variational.elbo(
        model, {"x": x}, variational=variational, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    # calculate marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(model, {"x": x}, proposal=variational, axis=0))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # define training/evaluation parameters
    epochs = 1000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    test_freq = 100
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/vae_digits"
    checkpoints_path = "checkpoints/vae_digits"

    # used to save checkpoints during training
    saver = tf.train.Saver(max_to_keep=10)
    save_model_freq = 100

    # run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore the model parameters from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(checkpoints_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        # begin training
        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            # test marginal log likelihood
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs, test_lls = [], []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1,
                                                  n: test_batch_size})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1000,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            # save model parameters
            if epoch % save_model_freq == 0:
                print('Saving model...')
                save_path = os.path.join(checkpoints_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

        # random generation of images from latent distribution
        x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])
        images = sess.run(x_gen, feed_dict={n: 100, n_particles: 1})
        name = os.path.join(result_path, "random_samples.png")
        save_image_collections(images, name)

        # the following code generates 100 samples for each number
        test_n = [3, 2, 1, 90, 95, 23, 11, 0, 84, 7]
        # map each digit to a corresponding sample from the test set so we can generate similar digits
        for i in range(len(test_n)):
            # get latent distribution from the variational giving as input a fixed sample from the dataset
            z = q_model.observe(x=np.expand_dims(x_test[test_n[i]], 0))['z']
            # run the computation graph adding noise to computed variance to get different output samples
            latent = sess.run(z, feed_dict={x_input: np.expand_dims(x_test[test_n[i]], 0),
                                            n: 1,
                                            n_particles: 100,
                                            std_noise: 0.7})
            # get the image from the model giving as input the latent distribution z
            x_gen = tf.reshape(model.observe(z=latent)["x_mean"], [-1, 28, 28, 1])
            images = sess.run(x_gen, feed_dict={n: 100, n_particles: 1})
            name = os.path.join(result_path, "{}.png".format(i))
            save_image_collections(images, name)


if __name__ == "__main__":
    main()
