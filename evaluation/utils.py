import re
import numpy as np
from tqdm import tqdm
from evaluation.seg_eval import get_scores

def get_labels(results, text, tokenizer, reduction=np.sum):
    subwords = list(map(tokenizer.tokenize, text))
    subword_lengths = list(map(len, subwords))
    sublabels = np.argmax(results, axis=-1)[1:] # remove first result because [CLS] token
    labels = []

    idx = 0
    for l in subword_lengths:
        label = []
        for _ in range(l):
            if idx >= len(sublabels):
                label.append(0)
                continue
            label.append(sublabels[idx])
            idx += 1
        labels.append(min(1, int(round(reduction(label)))))
    
    return labels



def create_candidate(test_path, candidate_path, model, tokenizer, max_length, reduction=np.sum):

    doc_pattern = r'# newdoc id = .+?\n'

    with open(test_path, 'r') as f:
        lines = f.readlines()

    with open(candidate_path, 'w') as f:

        pbar = tqdm(total=len(lines), desc='Evaluation in progress...')

        i = 0
        while i < len(lines):

            if re.match(doc_pattern, lines[i]) != None:
                f.write(lines[i])
                pbar.update(1)
                i += 1

            buffer = []
            text = []

            while i < len(lines) and lines[i] != '\n':
                buffer.append(lines[i].split('\t')[:-1] + ['_\n'])
                text.append(lines[i].split('\t')[1])
                pbar.update(1)
                i += 1

            inputs = tokenizer.encode_plus(text=' '.join(text), max_length=max_length, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True)

            results = model.predict([inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']])

            labels = get_labels(results, text, tokenizer, reduction=reduction)
            assert len(labels) == len(text)

            for b, l in zip(buffer, labels):
                if l == 0:
                    f.write('\t'.join(b))
                else:
                    f.write('\t'.join(b[:-1] + ['BeginSeg=Yes\n']))

            f.write('\n')
            pbar.update(1)
            i += 1
    pbar.close()


def run_evaluation(test_path, candidate_path, model, tokenizer, max_length, reduction=np.sum):
    create_candidate(test_path, candidate_path, model, tokenizer, max_length, reduction=np.sum)
    score_dict = get_scores(test_path, candidate_path)
    return score_dict

def print_score_dict(score_dict):
    print("File: " + score_dict["doc_name"])
    print("o Total tokens: " + str(score_dict["tok_count"]))
    print("o Gold " +score_dict["seg_type"]+": " + str(score_dict["gold_seg_count"]))
    print("o Predicted "+score_dict["seg_type"]+": " + str(score_dict["pred_seg_count"]))
    print("o Precision: " + str(score_dict["prec"]))
    print("o Recall: " + str(score_dict["rec"]))
    print("o F-Score: " + str(score_dict["f_score"]))
