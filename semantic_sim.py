import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# from gensim.models.wrappers import FastText

if os.uname().nodename == 'spallas-macbook.local':  # i.e. on my pc :)
    os.environ["TFHUB_CACHE_DIR"] = '/Users/davidespallaccini/sourcecode/tfhub_caches'
VERBOSE = False

"""
Try with at least three implementations of semantic similarity.
1) Universal Sentence Encoder
2) Universal Sentence Encoder faster and less precise (DAN)
3) Fuzzy sets with word embeddings
4) Facebook's LASER
"""
# TODO: manage similarity in batches of sentences for higher speed.


def embed_use(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


class SimilarityServer:

    # models
    UNIV_SENT_ENCODER = 0
    USE_WITH_DAN = 1
    # FUZZY_SET = 2
    LASER = 3

    # constants
    _LASER_TMP = 'laser_temp.txt'
    _LASER_DIM = 1024

    def __init__(self, model=0, win_size=None, laser_home_dir=None):
        self.model = model  # which model to use
        self.win_size = win_size
        self._context = ""
        self.context_emb = None
        if self.model == self.UNIV_SENT_ENCODER:  # UNIVERSAL SENTENCE ENCODER
            self.use_embed_fn = embed_use("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        elif self.model == self.USE_WITH_DAN:
            self.use_embed_fn = embed_use("https://tfhub.dev/google/universal-sentence-encoder/2")
        # elif self.model == self.FUZZY_SET:
        #     self.word_embed_model = FastText.load_fasttext_format('models/ft_model.bin')
        elif self.model == self.LASER:
            if not laser_home_dir:
                raise ValueError('To use LASER you have to provide path to the source directory.')
            else:
                self.laser_home = laser_home_dir
        else:
            raise ValueError("Please set up the similarity server with a valid model index.")
        print("SimilarityServer set up.")

    def set_context(self, context_sentence):
        self._context = ' '.join(context_sentence)
        if self.model == self.UNIV_SENT_ENCODER:
            self.context_emb = self.use_embed_fn([self._context])[0]
        elif self.model == self.USE_WITH_DAN:
            self.context_emb = self.use_embed_fn([self._context])[0]
        # elif self.model == self.FUZZY_SET:
        #     words = context_sentence.split(' ')
        #     self.context_emb = np.array([self.word_embed_model.wv[w] for w in words if w in self.word_embed_model.wv])
        elif self.model == self.LASER:
            self.context_emb = self.embed([self._context])[0]

    def _dot_prod_sim(self, b):
        # b is a sentence embedding could be a batch
        sim = np.inner(self.context_emb, b)
        return sim

    def _euclidean_sim(self, b):
        # b is a sentence embedding could be a batch
        dist = np.linalg.norm(self.context_emb, b)
        return 1 / (1 + dist)

    def embed(self, batch):
        if self.model == self.UNIV_SENT_ENCODER or self.model == self.USE_WITH_DAN:
            return self.use_embed_fn(batch)
        # elif self.model == self.FUZZY_SET:
        #     raise ValueError('Embedding of batch of sentences not supported in Fuzzy Sets mode')
        elif self.model == self.LASER:
            f = open(self._LASER_TMP, 'w')
            for line in batch:
                f.write(line.strip() + '\n')
                f.flush()
            f.close()
            if os.path.exists(f'{self._LASER_TMP}.raw'):
                os.system(f'rm {self._LASER_TMP}.raw')
            command = f"{self.laser_home}/tasks/embed/embed.sh {self._LASER_TMP} it {self._LASER_TMP}.raw"
            os.system(command)
            embedding = np.fromfile(self._LASER_TMP + '.raw', dtype=np.float32, count=-1)
            embedding.resize(embedding.shape[0] // self._LASER_DIM, self._LASER_DIM)
            return embedding

    # def _fuzzy_bow_sim(self, sentence):
    #
    #     words_list = sentence.split(' ')
    #     X = self.context_emb
    #     Y = np.array([self.word_embed_model.wv[w] for w in words_list if w in self.word_embed_model.wv])
    #
    #     U = np.vstack((X, Y))
    #     z = np.array([0] * U.shape[0])
    #     x = np.vstack((X.dot(U.T), z))
    #     y = np.vstack((Y.dot(U.T), z))
    #
    #     x = np.max(x, axis=0)
    #     y = np.max(y, axis=0)
    #
    #     xy = np.vstack((x, y))
    #     r = np.min(xy, axis=0)
    #     q = np.max(xy, axis=0)
    #
    #     return np.sum(r) / np.sum(q)

    def similarity(self, b, w_i=None):
        if not w_i:
            w_i = len(b) // 2
        if w_i >= len(b) or w_i <= 0:
            raise ValueError('Argument w_i is out of range for input sentence')
        if self.win_size is not None:
            b = b.split(' ')
            w = b[w_i]
            b = ' '.join(b[w_i - self.win_size : w_i + self.win_size])

        # b is a string could be a batch...
        if self.model == self.UNIV_SENT_ENCODER:
            begin = time.time()
            score = self._dot_prod_sim(self.use_embed_fn([b])[0])
            if VERBOSE:
                nl = '\n'
                print(f"\t{score:.4f} time: {time.time() - begin:.3f} s; {w.upper()}: {b.replace(nl, ' ')}")
            return score
        elif self.model == self.USE_WITH_DAN:
            begin = time.time()
            score = self._dot_prod_sim(self.use_embed_fn([b])[0])
            if VERBOSE:
                nl = '\n'
                print(f"\t{score:.4f} time: {time.time() - begin:.3f} s; {w.upper()}: {b.replace(nl, ' ')}")
            return score
        # elif self.model == self.FUZZY_SET:
        #     return self._fuzzy_bow_sim(b)
        elif self.model == self.LASER:
            return self._euclidean_sim(self.embed([b])[0])

    def __hash__(self):
        return self.model

