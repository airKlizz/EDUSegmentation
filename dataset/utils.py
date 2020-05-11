import re
from sklearn.model_selection import train_test_split
import tensorflow as tf

def pad_to_max_length(labels, max_length):
    return ([0] + labels + [0] * max(0, max_length-len(labels)-1))[:max_length]

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, shuffle=10000, random_state=2020):

    doc_pattern = r'# newdoc id = .+?\n'
    blank_pattern = r'^ *$'

    with open(train_path) as f:
        data = f.read()

    docs = re.split(doc_pattern, data)

    X = []
    y = []

    for doc in docs:

        if re.match(blank_pattern, doc) != None:
            continue

        for sentence in doc.split('\n\n'):

            text = ""
            y_ = []

            for line in sentence.split('\n'):

                if re.match(blank_pattern, line) != None:
                    continue

                elems = line.split('\t')
                assert len(elems) == 10, 'wrong line: {line}'
                string = elems[1]
                label = 0 if elems[-1][:-1] == '_' else 1
                toks = tokenizer.encode(string, add_special_tokens=False)
                text += string
                y_ += [label]*len(toks)
            
            inputs = tokenizer.encode_plus(text=text,
                                            max_length=max_length,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True, 
                                            return_attention_mask=True)

            X.append([inputs['input_ids'],
                    inputs['attention_mask'],
                    inputs['token_type_ids']
            ])
            y_ = pad_to_max_length(y_, max_length)
            y.append(y_)
    
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset