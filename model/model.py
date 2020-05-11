import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, huggingface, dim, num_labels, dropout):
        super(Model, self).__init__()
        self.huggingface = huggingface 
        self.linear = tf.keras.layers.Dense(dim, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classification = tf.keras.layers.Dense(num_labels, activation='tanh')
        
    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        x = self.huggingface(inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
        x = self.linear(x[0])
        x = self.dropout(x)
        x = self.classification(x)
        return x