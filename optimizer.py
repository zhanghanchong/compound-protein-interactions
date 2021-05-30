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


loss_object = losses.BinaryCrossentropy(from_logits=True)


def loss_function(real, pred):
    return loss_object(real, pred)


def accuracy_function(real, pred):
    accuracy = tf.cast(tf.equal(real, tf.cast(tf.greater(pred, 0), tf.float32)), tf.float32)
    return tf.reduce_sum(accuracy) / tf.cast(real.shape, tf.float32)
