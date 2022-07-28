import random
from OmniCog import load_coco

def train_split(path, validation=0.2):
    manager = load_coco(path)
    image_ids = manager.imageIDs
    N = len(image_ids)
    breakpoint = int(validation*N)
    
    random.shuffle(image_ids)
    
    #returns (validation, training)
    return (image_ids[:breakpoint], image_ids[breakpoint:])

def extract_triples(path, ids, validation=False):
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
    manager = load_coco(path)
    image_ids = manager.imageIDs
    
    N = len(ids)
    
    tripleList = []
    
    for imID in ids:
        if validation:
            captionIDs = manager.getCaptionIDs(imID)
        else:
            captionIDs = [random.choice(manager.getCaptionIDs(imID))]
        tripleList.extend([(imID, capID, getConfuser(imID,ids)) for capID in captionIDs])
    return tripleList

def getConfuser(trueID, confuserPool):
    while True:
        choice = random.choice(confuserPool)
        if choice!=trueID:
            break
    return choice

