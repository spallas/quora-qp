import os

import numpy as np
import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.optim import Adam

torch.manual_seed(42)

train_data = 'data/quora-question-pairs/xsmall_train.csv'
# test_data = 'data/quora-question-pairs/xsmall_train.csv'
test_data = 'data/quora-question-pairs/small_test.csv'

batch_size = 16
num_epochs = 6
max_grad_norm = 1.0

checkpoint_path = 'logs/checkpoint.pth'  # 'drive/My Drive/dm/logs/checkpoint.pth'
eval_report = 'logs/report.txt'  # 'drive/My Drive/dm/logs/eval_report.txt'

ON_CUDA = torch.cuda.is_available()

tok = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


def load_pairs(data_path):
    df = pd.read_csv(data_path, index_col=0)
    examples = []
    num_duplicates = 0
    for i, row in df.iterrows():
        q1_tokens = tok.tokenize(str(row['question1']))
        q2_tokens = tok.tokenize(str(row['question2']))
        input_tokens = ['[CLS]'] + q1_tokens + ['[SEP]'] + q2_tokens + ['[SEP]']
        sent_types = ([0] * (len(q1_tokens) + 2)) + ([1] * (len(q2_tokens) + 1))
        label = int(row['is_duplicate'])
        if label == 1:
            num_duplicates += 1
        examples.append((tok.convert_tokens_to_ids(input_tokens), sent_types, label))
        print(f'\r{len(examples)}', end='')
    print()
    print(f"Positive examples: {num_duplicates}, {num_duplicates * 100/len(examples):.3f}% of total")
    return examples


def batch_iter(examples):
    batch = {'ids_words': [], 'mask': [], 'seq_tag': [], 'types': []}
    for example in examples:
        ids, types, label = example
        batch['ids_words'].append(ids)
        batch['mask'].append([1] * len(ids))
        batch['seq_tag'].append(label)
        batch['types'].append(types)
        if len(batch['ids_words']) == batch_size:
            b = {k: pad_sequences(v, padding='post') for k, v in batch.items() if k != 'seq_tag'}
            b['seq_tag'] = batch['seq_tag']
            yield tuple(map(torch.LongTensor, (b[k] for k in sorted(b))))
            batch = {'ids_words': [], 'mask': [], 'seq_tag': [], 'types': []}
    if len(batch['ids_words']) != 0:
        b = {k: pad_sequences(v, padding='post') for k, v in batch.items() if k != 'seq_tag'}
        b['seq_tag'] = batch['seq_tag']
        yield tuple(map(torch.LongTensor, (b[k] for k in sorted(b))))


if __name__ == '__main__':

    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    if ON_CUDA:
        model.to('cuda')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, lr=5e-5)

    checkpoint = {}
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path) if ON_CUDA else torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
    else:
        last_epoch = 0
        min_loss = 1e3

    for epoch in range(last_epoch+1, num_epochs):
        print(f'Epoch: {epoch}')

        print('Evaluation')
        test_iter = batch_iter(load_pairs(test_data))
        model.eval()
        with torch.no_grad():
            pred, true = [], []
            for i, eval_batch in enumerate(test_iter):
                if ON_CUDA:
                    eval_batch = tuple(t.to('cuda') for t in eval_batch)
                b_input_ids, b_input_mask, b_labels, b_types = eval_batch
                logits = model(b_input_ids, b_types, b_input_mask)
                logits = logits.detach().cpu().numpy()
                b_labels = b_labels.to('cpu').tolist()
                pred += np.argmax(logits, axis=1).tolist()
                true += b_labels
                print("\rBatch:", i, end='')
            print()
        with open(eval_report, 'w') as fo:
            print(classification_report(true, pred, digits=3), file=fo)
            print(f'Accuracy: {accuracy_score(true, pred)}\n')
            print(f'F1: {f1_score(true, pred, average="micro")}\n')

        print('Training...')
        train_iter = batch_iter(load_pairs(train_data))
        model.train()
        for step, batch in enumerate(train_iter):
            model.zero_grad()
            if ON_CUDA:
                batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_labels, b_types = batch
            loss = model(b_input_ids, b_types, b_input_mask, b_labels)
            loss /= b_input_mask.float().sum()
            loss.backward()
            if step % 200 == 0:
                print(f'{loss.item():.4f}')
                # possibly save progress
                current_loss = loss.item()
                if current_loss < min_loss:
                    min_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'current_loss': current_loss,
                        'min_loss': min_loss,
                    }, checkpoint_path)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

