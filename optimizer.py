import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import optimizers


class Schedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup):
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup = warmup

    def __call__(self, step):
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup ** -1.5))


def make_optimizer(adam_beta_1, adam_beta_2, adam_epsilon, d_model, warmup):
    return optimizers.Adam(Schedule(d_model, warmup), beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)


def loss_function_tfm(real, pred):
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(real, pred)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


def accuracy_function_tfm(real, pred):
    accuracy = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracy = tf.cast(tf.math.logical_and(accuracy, mask), tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)


def loss_function_clf(real, pred):
    loss_object = losses.BinaryCrossentropy(from_logits=True)
    return loss_object(real, pred)


def accuracy_function_clf(real, pred):
    accuracy = tf.cast(tf.equal(real, tf.cast(tf.greater(pred, 0), tf.int64)), tf.float32)
    return tf.reduce_sum(accuracy) / tf.cast(real.shape, tf.float32)
