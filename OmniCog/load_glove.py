from gensim.models import KeyedVectors
from cogworks_data.language import get_data_path

def load_glove(*,limit=None, filename = "glove.6B.200d.txt.w2v"):
    # this takes a while to load -- keep this in mind when designing your capstone project
    glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False, limit=limit)
    return glove
