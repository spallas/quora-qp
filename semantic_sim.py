import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece


USE_QA_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"
USE_EMBED_DIM = 512


def embed_use(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def embed_use_multilingual(module):
    # Graph set up.
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(module)
        embedded_text = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)
    return lambda x: session.run(embedded_text, feed_dict={text_input: x})


class SimServer:

    # models
    UNIV_SENT_ENCODER = 2
    USE_WITH_DAN = 1
    USE_MULTILINGUAL = 0
    USE_QA = 3

    def __init__(self, model=0, win_size=None):
        self.model = model  # which model to use
        self.win_size = win_size
        self._context = ""
        self.context_emb = None
        if self.model == self.UNIV_SENT_ENCODER:  # UNIVERSAL SENTENCE ENCODER
            self.use_embed_fn = embed_use("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        elif self.model == self.USE_WITH_DAN:
            self.use_embed_fn = embed_use("https://tfhub.dev/google/universal-sentence-encoder/2")
        elif self.model == self.USE_MULTILINGUAL:
            self.use_embed_fn = embed_use_multilingual("https://tfhub.dev/google/"
                                                       "universal-sentence-encoder-multilingual-large/1")
        elif self.model == self.USE_QA:
            self.use_embed_fn, self.embed_answers_batch = self.embed_use_qa()
        print("SimilarityServer set up.")

    @staticmethod
    def embed_use_qa(tfhub_path=USE_QA_URL):

        def question_embed(question_text):
            return session.run(question_embedding, {question: [question_text]})['outputs']

        def answers_embed(answers, contexts):
            return session.run(response_embedding, {
                response: answers,
                response_context: contexts
            })['outputs']

        g = tf.Graph()
        with g.as_default():
            module = hub.Module(tfhub_path)
            question = tf.placeholder(dtype=tf.string, shape=[None])
            response = tf.placeholder(dtype=tf.string, shape=[None])
            response_context = tf.placeholder(dtype=tf.string, shape=[None])
            question_embedding = module(question, signature="question_encoder", as_dict=True)
            response_embedding = module(
                inputs={
                    "input": response,
                    "context": response_context
                },
                signature="response_encoder", as_dict=True)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()
        # Initialize session.
        session = tf.Session(graph=g)
        session.run(init_op)
        return question_embed, answers_embed

    def set_context(self, context_sentence: str):
        self._context = context_sentence
        self.context_emb = self.use_embed_fn([self._context])[0]

    def _dot_prod_sim(self, b):
        # b is a sentence embedding could be a batch
        sim = np.inner(self.context_emb / np.linalg.norm(self.context_emb),
                       b / np.linalg.norm(b))
        return sim

    def _euclidean_sim(self, b):
        # b is a sentence embedding could be a batch
        dist = np.linalg.norm(self.context_emb, b)
        return 1 / (1 + dist)

    def embed(self, batch):
        return self.use_embed_fn(batch)

    def similarity(self, b):
        # b is a string could be a batch...
        score = self._dot_prod_sim(self.use_embed_fn([b])[0])
        return score
