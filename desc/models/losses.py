import numpy as np
import tensorflow as tf
from keras import backend as K

class NB(object):
    def __init__(self, theta=None, scope='nbinom_loss/'):

        # for numerical stability
        self.eps = 1e-10
        self.scope = scope
        self.theta = theta

    def loss(self, y_true, y_pred):
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.lgamma(theta+eps) + tf.lgamma(y_true+1.0) - tf.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.log(theta+eps) - tf.log(y_pred+eps)))

            final = t1 + t2

        return tf.reduce_mean(final)


