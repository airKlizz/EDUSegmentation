import numpy as np
from transformers import BertTokenizer
from utils import run_evaluation, print_score_dict

class FakeModel():
    def __init__(self):
        pass

    def predict(self, inputs):
        return np.array([[0, 1]] * len(inputs[0]))

test_path = 'dataset/data/deu.rst.pcc/deu.rst.pcc_dev.conll'
candidate_path = 'candidate.conll'
model = FakeModel()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
max_length = 64

score_dict = run_evaluation(test_path, candidate_path, model, tokenizer, max_length, reduction=np.sum)
print_score_dict(score_dict)