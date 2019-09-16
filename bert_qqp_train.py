import os

import numpy as np
import pandas as pd
import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.extmath import softmax
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.manual_seed(42)


class QQPLoader:

    def __init__(self, device, data_path, batch_size=32):
        self.device = device
        self.df = pd.read_csv(data_path, index_col=0)
        self.batch_size = batch_size
        self.tok: BertTokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.stop_flag = False

    def __iter__(self):
        self.pandas_iterator = self.df.iterrows()
        self.stop_flag = False
        return self

    def __next__(self):
        if self.stop_flag:
            raise StopIteration
        bx, bt, by = [], [], []
        for i in range(self.batch_size):
            try:
                row = next(self.pandas_iterator)
            except StopIteration:
                self.stop_flag = True
                break
            row = row[1]
            a = self.tok.encode('[CLS]' + str(row['question1']) + '[SEP]')
            b = self.tok.encode(str(row['question2']) + '[SEP]')
            types = [0] * len(a) + [1] * len(b)
            y = int(row['is_duplicate'])
            bx.append(torch.tensor(a + b))
            bt.append(torch.tensor(types))
            by.append(torch.tensor(y))

        bx = nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bt = nn.utils.rnn.pad_sequence(bt, batch_first=True, padding_value=1)
        by = torch.stack(by)
        return bx.to(self.device), bt.to(self.device), by.to(self.device)


class PretrainedLMForQQP:

    def __init__(self,
                 checkpoint_path='logs/checkpoint.pth',
                 eval_report_path='logs/report.txt',
                 is_training=True,
                 train_path='train.csv',
                 test_path='test.csv',
                 batch_size=16):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 5e-5
        self.num_epochs = 6
        self.batch_size = batch_size
        self.log_interval = 1000
        self.is_training = is_training
        self._plot_server = None

        self.checkpoint_path = checkpoint_path
        self.best_model_path = checkpoint_path + '.best'
        self.eval_report = eval_report_path
        self.train_data_path = train_path
        self.test_data_path = test_path
        self.train_loader = QQPLoader(self.device, self.train_data_path, self.batch_size)
        self.test_loader = QQPLoader(self.device, self.test_data_path, self.batch_size)

        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._maybe_load_checkpoint()
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.last_epoch + 1, self.num_epochs):
            print(f'Epoch: {epoch}')
            step = 0
            for step, (b_x, b_t, b_y) in enumerate(self.train_loader, self.last_step):
                self.model.zero_grad()
                b_m = (b_x != 0)
                loss, _ = self.model(b_x, b_t, b_m, b_y)
                loss /= b_m.float().sum()
                loss.backward()
                self._log(step, loss, epoch)
                self.optimizer.step()
            self.last_step += step

    def test(self):
        print('Evaluation')
        self.model.eval()
        with torch.no_grad():
            pred, true = [], []
            for step, (b_x, b_t, b_y) in enumerate(self.test_loader):
                outputs = self.model(b_x, b_t, (b_x != 0))
                logits = outputs[0].to('cpu').numpy()
                b_labels = b_y.to('cpu').tolist()
                pred += np.argmax(logits, axis=-1).tolist()
                true += b_labels
        self.model.train()
        f1 = f1_score(true, pred, average="micro")
        if self.is_training:
            self._save_best(f1)
        with open(self.eval_report, 'w') as fo:
            print(classification_report(true, pred, digits=3), file=fo)
            print(f'Accuracy: {accuracy_score(true, pred)}')
            print(f'F1: {f1}\n')
        return f1

    def retrieve(self, query: str, data_file: str = None):

        def get_batch_scores(bx, bt):
            with torch.no_grad():
                bx = nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0).to(self.device)
                bt = nn.utils.rnn.pad_sequence(bt, batch_first=True, padding_value=1).to(self.device)
                outputs = self.model(bx, bt, (bx != 0))
                logits = outputs[0].to('cpu').numpy()
                b_scores = softmax(logits)
                scores.extend(b_scores[:, 1].tolist())

        tok = self.train_loader.tok
        scores = []
        a = tok.encode('[CLS] ' + query + ' [SEP]')
        b_x, b_t = [], []
        self._load_best()
        self.model.eval()
        with open(data_file) as f:
            for line in f:
                candidate = line.strip()
                b = tok.encode(candidate + ' [SEP]')
                types = [0] * len(a) + [1] * len(b)
                b_x.append(torch.tensor(a + b))
                b_t.append(torch.tensor(types))
                if len(b_x) == self.batch_size:
                    get_batch_scores(b_x, b_t)
                    b_x, b_t = [], []
            if len(b_x) > 0:
                get_batch_scores(b_x, b_t)
        ranking = zip(range(len(scores)), scores)
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranking]

    def _log(self, step, loss, epoch_i):
        if step % self.log_interval == 0:
            print(f'\rLoss: {loss.item():.4f} ', end='')
            self._plot('Train loss', loss.item(), step)
            self._gpu_mem_info()
            f1 = self.test()
            self._maybe_checkpoint(loss, epoch_i)
            self._plot('Dev F1', f1, step)
            self.model.train()  # return to train mode after evaluation

    def _save_best(self, f1):
        if f1 > self.best_f1_micro:
            self.best_f1_micro = f1
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'f1': self.best_f1_micro
            }, self.best_model_path)

    def _load_best(self):
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=str(self.device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Could not find any best model checkpoint: {self.best_model_path}")

    def _maybe_checkpoint(self, loss, epoch_i):
        current_loss = loss.item()
        if current_loss < self.min_loss:
            min_loss = current_loss
            torch.save({
                'epoch': epoch_i,
                'last_step': self.last_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_loss': current_loss,
                'min_loss': min_loss,
                'f1': self.best_f1_micro
            }, self.checkpoint_path)

    def _maybe_load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.last_step = checkpoint['last_step']
            self.min_loss = checkpoint['min_loss']
            self.best_f1_micro = checkpoint['f1']
            print(f"Loaded checkpoint from: {self.checkpoint_path}")
            if self.last_epoch >= self.num_epochs:
                print("Training finished for this checkpoint")
        else:
            self.last_epoch = 0
            self.last_step = 0
            self.min_loss = 1e3
            self.best_f1_micro = 0.0

    def _plot(self, name, value, step):
        if not self._plot_server:
            self._plot_server = SummaryWriter(log_dir='logs')
        self._plot_server.add_scalar(name, value, step)

    @staticmethod
    def _gpu_mem_info():
        if torch.cuda.is_available():  # check if memory is leaking
            print(f'Allocated GPU memory: '
                  f'{torch.cuda.memory_allocated() / 1_000_000} MB')


def evaluate_bert_qqp(test_dataset: str,
                      test_questions: str,
                      model: PretrainedLMForQQP,
                      cache_file: str = None):
    num_questions = 0
    num_correct_3 = 0
    num_correct_5 = 0
    num_correct_10 = 0
    num_correct = 0
    last_step = 0

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            line0 = next(f)
            last_step = eval(line0.strip())
            line1 = next(f)
            num_questions, num_correct_3, \
                num_correct_5, num_correct_10, num_correct = eval(line1)

    step = 0
    with open(test_questions) as f:
        for line in tqdm(f):
            if step < last_step:
                continue
            q, dupl = line.strip().split('\t')
            retrieved = model.retrieve(q, test_dataset)

            if int(dupl) in retrieved[:10]:
                num_correct_10 += 1
            if int(dupl) in retrieved[:5]:
                num_correct_5 += 1
            if int(dupl) in retrieved[:3]:
                num_correct_3 += 1
            if int(dupl) in retrieved[:1]:
                num_correct += 1
            num_questions += 1

            step += 1
            last_step = step
            if last_step % 10 == 0:
                with open(cache_file, 'w') as f:
                    print(last_step, file=f)
                    print([num_questions, num_correct_3, num_correct_5, num_correct_10, num_correct],
                          file=f)

                print(f"\nDetection @1: {100 * num_correct / num_questions} %")
                print(f"Detection @3: {100 * num_correct_3 / num_questions} %")
                print(f"Detection @5: {100 * num_correct_5 / num_questions} %")
                print(f"Detection @10: {100 * num_correct_10 / num_questions} %")


def main():
    t = PretrainedLMForQQP(train_path='data/quora-question-pairs/train.csv',
                           test_path='data/quora-question-pairs/test.csv',
                           batch_size=8)

    test_questions = 'data/test_questions.txt'
    test_dataset = 'data/test_dataset.txt'
    evaluate_bert_qqp(test_dataset, test_questions, t, 'data/cache.txt')


if __name__ == '__main__':
    main()
