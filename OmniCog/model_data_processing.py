import random
import numpy as np

def train_split(image_ids, validation=0.2):
    N = len(image_ids)
    breakpoint = int(validation*N)
    
    random.shuffle(image_ids)
    
    #returns (validation, training)
    return (image_ids[:breakpoint], image_ids[breakpoint:])

def extract_triples(manager, ids, validation=False):
    """
    Gets triples every epoch
    
    Parameters
    ----------
    path : string
        Path to coco data
    
    validation : float, optional (default=0.2)
        Fraction of image objects allocated to validation
    
    Returns
    -------
    list
        Massive list of 2-tuple ids (true, caption)
    """
    
    N = len(ids)
    
    tripleList = []
    
    for imID in ids:
        if validation:
            captionIDs = manager.getCaptionIDs(imID)
        else:
            captionIDs = [random.choice(manager.getCaptionIDs(imID))]
        tripleList.extend([(imID, capID, getConfuser(imID,ids)) for capID in captionIDs])
    return np.array(tripleList)

def getConfuser(trueID, confuserPool):
    while True:
        choice = random.choice(confuserPool)
        if choice!=trueID:
            break
    return choice

