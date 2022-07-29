from collections import Counter
import math
from numpy import linalg

bigCount = Counter()
captions = 0

for caption_info in coco_data["annotations"]:
    captions +=1
    caption_vocab = set(tokenize(caption_info["caption"]))
    bigCount.update(caption_vocab)
        
def IDFs(counter):
    """ 
   
    Parameters
    ---------
    counter : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        captions.
    
    Returns
    -------
    IDFs: {"word1": float, "word2": float, ...} a dictionary mapping all vocab words to inverse document frequencies
    """
    
    
    vocab = bigCount.keys()
    IDFs = {}
    for word in vocab:
        IDFs[word] = math.log10(captions/bigCount[word])
    
    return IDFs
  
