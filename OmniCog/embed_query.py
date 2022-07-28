from numpy import linalg

def normalize(unnorm_embed):
    return embed/linalg.norm(embed, axis = 1)
  
def query_embed(query):
    embed = np.zeros(200,)
    tokens = tokenize(query)
    try:
        embed = np.sum(IDF(bigCount)[word]*glove[word] for word in tokens)
        return normalize(embed)
    except:
        return embed
     
