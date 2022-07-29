import numpy as np
from tokenize_captions import tokenize_caption
from IDFs import IDFs

def normalize(embed):
    return embed/np.linalg.norm(embed)
  
def query_embed(query, idfs, glove):
    """
    Parameter:
        query: string
            User's input caption/query
    
    Returns:
        embed: np.array(200,)
            IDF-weighted sum of the glove-embedding for each word in the caption
    
    """
    embed = np.zeros(200,)
    tokens = tokenize_caption(query)
    
    flag = False
    for word in tokens:
        try:
            embed += idfs[word]*glove[word]
            flag = True
        except:
            pass
    return normalize(embed) if flag else embed
