import os
import pickle
import random
import re
import string

import faiss
import hdbscan
import numpy as np
import umap
import xxhash
from datasketch import MinHash, MinHashLSH

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from spacy.lang.en import English
from spacy.lang.it import Italian

import whoosh.index as index
from tqdm import tqdm
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup

from semantic_sim import SimServer

tokenize_it = Italian().Defaults.create_tokenizer()
tokenize_en = English().Defaults.create_tokenizer()
wnl = WordNetLemmatizer()
punct = string.punctuation.replace('.', '').replace(',', '')


def to_shingles(doc, k=5):
    shingles = set()
    doc_string = doc.lower()
    if len(doc_string) <= k:
        doc_string = doc + 'no_txt_' + str(xxhash.xxh64(str(random.random())).hexdigest())
    for i in range(len(doc_string) - k + 1):
        h = doc_string[i:i+k]
        shingles.add(h.encode('utf8'))
    return list(shingles)


def search_preprocess(language):
    """

    :param language:
    :return:
    """
    def wrapped(text):
        only_print = re.sub(r'\s+', ' ',
                            re.sub(fr'([{string.punctuation}])(\S)', r'\1 \2',
                                   re.sub(r'[^ -~]+', ' ', text)))
        tokens = tokenize_en(only_print) if language == 'english' else tokenize_it(only_print)
        stops = set(stopwords.words(language))
        if language == 'english':
            tokens = map(lambda w: wnl.lemmatize(str(w)), tokens)
        processed = ' '.join([str(t).lower() for t in tokens if str(t) not in stops and str(t) not in punct])
        return processed
    return wrapped


class QuestionRecommendation:
    """

    """

    _CACHED_VECTORS = 'data/questions.faiss'

    def __init__(self, questions_file, model):
        self.questions_file = questions_file
        self.sim_server = SimServer(model)
        self._CACHED_VECTORS = self._CACHED_VECTORS + '.' + str(model)
        p = search_preprocess('english')
        if os.path.exists(self._CACHED_VECTORS):
            self.vector_index = faiss.read_index(self._CACHED_VECTORS)
        else:
            documents_matrix = []
            with open(self.questions_file) as f:
                batch, batch_size = [], 128
                for line in tqdm(f):
                    batch.append(line)
                    if len(batch) == batch_size:
                        documents_matrix.extend(self.sim_server.embed(batch))
                        batch = []
                if len(batch) != 0:
                    documents_matrix.extend(self.sim_server.embed(batch))
            documents_matrix = np.array(documents_matrix)
            faiss.normalize_L2(documents_matrix)  # will use cosine similarity
            self.vector_index = faiss.IndexFlatIP(documents_matrix[0].shape[0])
            # noinspection PyArgumentList
            self.vector_index.add(documents_matrix)
            faiss.write_index(self.vector_index, self._CACHED_VECTORS)

    def recommend(self, seed_questions, k):
        doc_vectors = []
        for s in seed_questions:
            sv = self.sim_server.embed([search_preprocess('english')(s)])
            faiss.normalize_L2(sv)
            doc_vectors.append(sv[0])
        embeddings = umap.UMAP(n_neighbors=4, n_components=128).fit_transform(np.array(doc_vectors))
        clustering = hdbscan.HDBSCAN(min_cluster_size=3)
        clustering.fit(embeddings)
        # print(clustering.labels_)
        clusters = {}
        for v, label in zip(doc_vectors, clustering.labels_):
            clusters.setdefault(label, []).append(v)
        avg_vectors = []
        for l in clusters:
            avg_vectors.append(np.average(clusters[l], axis=0))
        results = []
        for v in avg_vectors:
            cosine_scores, keys = self.vector_index.search(np.array([v]), k)
            results.append(keys[0])
        return results

    def search(self, query, k=10):
        sv = self.sim_server.embed([query])
        faiss.normalize_L2(sv)
        cosine_scores, keys = self.vector_index.search(sv, k)
        return keys[0]


class TfIdfSearch:
    """

    """

    _CACHED_INDEX = 'data/tfidf_index'

    def __init__(self, questions_file):
        """
        """
        self.questions_file = questions_file

        if os.path.exists(self._CACHED_INDEX):
            self.indx = index.open_dir(self._CACHED_INDEX)
        else:
            os.makedirs(self._CACHED_INDEX)

            schema = Schema(key=ID(stored=True, unique=True), text=TEXT(stored=False))
            ix = index.create_in(self._CACHED_INDEX, schema)
            writer = ix.writer()
            with open(self.questions_file) as f:
                for i, row in tqdm(enumerate(f)):
                    writer.add_document(key=str(i), text=row.strip())
                writer.commit()
            self.indx = ix
        self.query_parser = QueryParser('text', self.indx.schema)
        self.searcher = self.indx.searcher()

    def search(self, query, k=10):
        """

        :param query:
        :param k:
        :return:
        """
        q = self.query_parser.parse(query)
        results = self.searcher.search(q, limit=k)
        if not results:
            q = QueryParser('text', schema=self.indx.schema, group=OrGroup).parse(query)
            results = self.searcher.search(q, limit=k)
        return [int(r['key']) for r in results]


class MinHashSearch:
    """

    """

    _CACHED_INDEX = 'data/min_hash_index.pkl'
    _NUM_PERMUTATIONS = 128
    _THRESHOLD = 0.5

    def __init__(self, questions_file):
        self.questions_file = questions_file

        if os.path.exists(self._CACHED_INDEX):
            with open(self._CACHED_INDEX, 'rb') as f:
                self.indx = pickle.load(f)  # load the index
        else:
            # build the index.
            self.indx = MinHashLSH(threshold=self._THRESHOLD,
                                   num_perm=self._NUM_PERMUTATIONS)
            with open(questions_file) as f:
                for i, line in tqdm(enumerate(f)):
                    tokens = to_shingles(line.strip())
                    min_hash = MinHash(num_perm=self._NUM_PERMUTATIONS)
                    for t in tokens:
                        min_hash.update(t)
                    self.indx.insert(i, min_hash)
            with open(self._CACHED_INDEX, 'wb') as f:
                pickle.dump(self.indx, f)

    def search(self, query, k=10):
        tokens = to_shingles(query)
        m = MinHash(num_perm=self._NUM_PERMUTATIONS)
        for t in tokens:
            m.update(t)
        results = self.indx.query(m)
        return results[:k]
