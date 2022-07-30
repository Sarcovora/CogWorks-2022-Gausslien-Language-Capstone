from IDFs import IDFs, captionCounter
from embed_query import query_embed
import numpy as np

def query_database(query, img_database, k, coco_data, glove):
    """Queries the image database and returns the top k images and their image IDs.
    
    Parameters
    ----------
    query : string
        The initial query that the user wants to find images for.
        
    img_database : dict
        The dictionary which contains image-ids mapped to the embedded image descriptors.
    
    k : integer
        The amount of images returned which match the query.
        
    Returns
    -------
    (top_k_similar, top_k_ids) : tuple
        A tuple containing a list of the top k cosine similarities and a list of the ids that correspond to the similarities.
    """
    bigCount, captions = captionCounter(coco_data)
    e_query = query_embed(query, IDFs(bigCount, captions), glove) #replace this with the actual name of Elaine's function
    #img_descriptors = np.array([img_database.values()])
    comparison_database = np.matmul(e_query, img_database.values())
    sorted_comparison_database = comparison_database.sort()[::-1]
    top_k_similar = sorted_comparison_database[:k]
    top_k_ids = []
    for img in sorted_comparison_database:
        idx = comparison_database.index(img)
        top_k_ids.append(img_database.keys()[img_database.values().index(comparison_database[idx])])
    return (top_k_similar, top_k_ids)