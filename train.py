import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from dataset.utils import create_tf_dataset
from model.model import Model
from evaluation.utils import run_evaluation, print_score_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of the HugginFace Model", default="bert-base-cased")
parser.add_argument("--model_dim", type=int, help="dim of classification layer", default=64)
parser.add_argument("--dropout", type=float, help="dropout", default=0.2)
parser.add_argument("--train_path", type=str, help="path to the train  file", default="dataset/data/eng.rst.rstdt/eng.rst.rstdt_train.conll")
parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=256)
parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--num_labels", type=int, help="number of labels", default=2)
parser.add_argument("--epochs", type=int, help="number of epochs", default=5)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=3e-5)
parser.add_argument("--filepath", type=str, help="filename for saving", default="model/saved_weights/weights.{val_loss:.2f}.h5")
parser.add_argument("--test_gold_path", type=str, help="path to the test gold file", default="dataset/data/eng.rst.rstdt/eng.rst.rstdt_dev.conll")
parser.add_argument("--candidate_path", type=str, help="path to the candidate file", default="dataset/candidate.conll")
parser.add_argument("--weight_for_0", type=float, help="weight_for_0", default=1.)
parser.add_argument("--weight_for_1", type=float, help="weight_for_1", default=1.)
parser.add_argument("--reduction", type=str, help="sum or mean", default='sum')
args = parser.parse_args()

# parameters
model_name = args.model_name
model_dim = args.model_dim
dropout = args.dropout
num_labels = args.num_labels
train_path = args.train_path
max_length = args.max_length
test_size = args.test_size
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
filepath = args.filepath
test_gold_path = args.test_gold_path
candidate_path = args.candidate_path
weight_for_0 = args.weight_for_0
weight_for_1 = args.weight_for_1
if args.reduction == 'sum':
    reduction = np.sum
elif args.reduction == 'mean':
    reduction = np.mean
else:
    assert True, 'Choose a correct reduction'

# init tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Model(TFAutoModel.from_pretrained(model_name), model_dim, num_labels, dropout)

# create dataset
train_dataset, validation_dataset = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size)

# optimizer, loss and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]
class_weight = {0: weight_for_0, 1: weight_for_1}

# compile
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

# train
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))
    history = model.fit(train_dataset, epochs=1, validation_data=validation_dataset, callbacks=[model_checkpoint],
                        #class_weight=class_weight
                        )
    score_dict = run_evaluation(test_gold_path, candidate_path, model, tokenizer, max_length, reduction=reduction)
    print_score_dict(score_dict)
