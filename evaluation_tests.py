
import pandas as pd
from tqdm import tqdm

from bert_qqp_train import PretrainedLMForQQP
from question_recommend import QuestionRecommendation, TfIdfSearch, MinHashSearch
from semantic_sim import SimServer
import tf_sentencepiece


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


def evaluate(engine, k):
    num_questions = 0
    num_correct_3 = 0
    num_correct_5 = 0
    num_correct_10 = 0
    num_correct = 0

    with open(TEST_QUESTIONS) as f:
        for line in tqdm(f):
            q, dupl = line.strip().split('\t')

            retrieved = engine.search(q, k=k)

            if int(dupl) in retrieved[:10]:
                num_correct_10 += 1
            if int(dupl) in retrieved[:5]:
                num_correct_5 += 1
            if int(dupl) in retrieved[:3]:
                num_correct_3 += 1
            if int(dupl) in retrieved[:1]:
                num_correct += 1
            num_questions += 1

    print(f"Detection @1: {100 * num_correct/num_questions} %")
    print(f"Detection @3: {100 * num_correct_3/num_questions} %")
    print(f"Detection @5: {100 * num_correct_5/num_questions} %")
    print(f"Detection @10: {100 * num_correct_10/num_questions} %")


if __name__ == '__main__':

    se = QuestionRecommendation(TEST_DATASET, SimServer.UNIV_SENT_ENCODER)
    se1 = QuestionRecommendation(TEST_DATASET, SimServer.USE_QA)
    se2 = QuestionRecommendation(TEST_DATASET, SimServer.USE_MULTILINGUAL)
    se3 = QuestionRecommendation(TEST_DATASET, SimServer.USE_WITH_DAN)
    tf = TfIdfSearch(TEST_DATASET)
    lsh = MinHashSearch(TEST_DATASET)
    print("Loaded indices", flush=True)
    print("Standard USE: ")
    evaluate(se, 20)
    print("USE per Question Answering: ")
    evaluate(se1, 20)
    print("USE advanced multilingual: ")
    evaluate(se2, 20)
    print("Standard USE with DAN network: ")
    evaluate(se3, 20)
    print("TF-IDF based search : ")
    evaluate(tf, 20)
    print("LSH based search : ")
    evaluate(lsh, 1000)
    print("BERT model based search: ")
    # evaluate_bert_qqp(TEST_DATASET)
