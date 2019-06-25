import pandas as pd
import random

TEST_SIZE = 40_000


def sample_uniform(file):
    """
    Each line has the same probability of being sampled.
    File is read only once. Implements reservoir sampling.
    :param file:
    :return:
    """
    for line in file:
        head = line
        break
    sampled = []
    i = 0
    for line in file:
        if i < TEST_SIZE:
            sampled.append(line)
        else:
            j = random.randint(0, i)
            if j < TEST_SIZE:
                sampled[j] = line
        i += 1
    sampled_ids = []
    with open("data/quora-question-pairs/test.csv", "w") as fo:
        print(head, end="", file=fo)
        for line in sampled:
            try:
                sampled_ids.append(int(line.split(',', maxsplit=1)[0][1:-1]))
                print(line, end="", file=fo)
            except ValueError:
                continue
    return set(sampled_ids)


def get_candidate_pairs(untagged_path):
    df = pd.read_csv(untagged_path, index_col=0)
    pass


def train_test_sample():
    with open('data/quora-question-pairs/tagged.csv') as f:
        ids = sample_uniform(f)

    with open('data/quora-question-pairs/tagged.csv') as f, \
            open('data/quora-question-pairs/train.csv', 'w') as fo:
        for line in f:
            print(line, end="", file=fo)
            break
        for line in f:
            try:
                pair_id = int(line.split(',', maxsplit=1)[0][1:-1])
                if pair_id not in ids:
                    print(line, end="", file=fo)
            except ValueError:
                print(line)


if __name__ == '__main__':
    # train_test_sample()
    get_candidate_pairs('data/quora-question-pairs/untagged.csv')
