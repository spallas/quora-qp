
import xxhash
import random
from itertools import combinations

PRIME_96_BITS = int(0x8cc0709b2987fb9bcb932247)
PRIME_128_BITS = int(0xa58b842eee27bf6164ed1f46eb169f05)


def to_shingles(docs, k=10):
    docs_shingled = []
    for doc in docs:
        shingles = set()
        doc_string = " ".join(doc)
        for i in range(len(doc_string) - k + 1):
            h = xxhash.xxh32(doc_string[i:i+k])
            shingles.add(h.intdigest())
        docs_shingled.append(list(shingles))
    return docs_shingled


def min_hash(docs_shingles, dim=48, dim_hash=32, p=PRIME_128_BITS):
    def universal_hash(a, b, x):
        return ((a*x + b) % p) % (2**dim_hash)
    rnd_ab = [(random.randint(1, p), random.randint(0, p)) for _ in range(dim)]
    min_hash_signatures = []
    for doc in docs_shingles:
        sig = [min([universal_hash(a, b, x) for x in doc]) for a, b in rnd_ab]
        min_hash_signatures.append(sig)
    return min_hash_signatures


def lsh(signatures, rows_per_band=8):
    lsh_groups = {}
    for doc_id, signature in enumerate(signatures):
        for i in range(0, len(signature), rows_per_band):
            band_string = "".join([hex(signature[i + r]) for r in range(rows_per_band)])
            if band_string in lsh_groups:
                lsh_groups[band_string].append(doc_id)
            else:
                lsh_groups[band_string] = [doc_id]
    groups = list(filter(lambda l: len(l) > 1, list(lsh_groups.values())))
    candidate_pairs = set()
    for g in groups:
        for pair in combinations(g, 2):
            candidate_pairs.add(pair)
    return candidate_pairs


if __name__ == "__main__":

    DIM = 100
    RPB = 20
    with open("lsh.txt", "w") as f:
        print(lsh(min_hash(to_shingles(load_and_preprocess()), dim=DIM), rows_per_band=RPB), file=f)
