import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, huggingface, dim, num_labels, dropout):
        super(Model, self).__init__()
        self.huggingface = huggingface 
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(
            num_labels, name="classifier"
        )
        
    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        outputs = self.huggingface(inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits