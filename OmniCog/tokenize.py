from collections import Counter
import numpy as np
import re, string

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    return punc_regex.sub('', corpus)


def tokenize(caption):
    caption = strip_punc(caption)
    return caption.lower().split()
