
import pandas as pd
from tqdm import tqdm

from question_recommend import QuestionRecommendation, TfIdfSearch, MinHashSearch

TEST_QUESTIONS = 'data/test_questions.txt'
TEST_DATASET = 'data/test_dataset.txt'


def save_test_questions():

    df = pd.read_csv('data/quora-question-pairs/train.csv', index_col=0)

    test_questions = []
    test_dataset = []

    num_test_questions = 1000
    num_dataset_questions = 400_000

    for i, row in df.iterrows():
        if row['is_duplicate'] == 1 and num_test_questions != 0:
            test_questions.append((row['question1'],
                                   len(test_dataset)))
            num_test_questions -= 1
        else:
            test_dataset.append(row['question1'])
            num_dataset_questions -= 1

        test_dataset.append(row['question2'])
        num_dataset_questions -= 1

        if num_dataset_questions <= 0:
            break

    with open(TEST_QUESTIONS, 'w') as f:
        for q in test_questions:
            print(f"{q[0]}\t{q[1]}", file=f)

    with open(TEST_DATASET, 'w') as f:
        for q in test_dataset:
            print(f"{q}", file=f)


def evaluate(engine):
    num_questions = 0
    num_correct = 0

    with open(TEST_QUESTIONS) as f:
        for line in tqdm(f):
            q, dupl = line.strip().split('\t')

            retrieved = engine.search(q, k=100)

            if int(dupl) in retrieved:
                num_correct += 1
            num_questions += 1

    print(f"Metric: {100 * num_correct/num_questions} %")


if __name__ == '__main__':

    se = QuestionRecommendation(TEST_DATASET)
    tf = TfIdfSearch(TEST_DATASET)
    lsh = MinHashSearch(TEST_DATASET)
    print("Loaded indices", flush=True)
    for e in [se, tf, lsh]:
        evaluate(e)
