import numpy as np
def margin_loss(trueVec, captionVec, confuserVec, margin=1):
    """
    Loss function for training the model that weights image vectors
    
    Parameters
    ----------
    trueVec : numpy.ndarray (200,)
       The model output of the image vector we're interested in
       Should be unit vector
       
    captionVec : numpy.ndarray (200,)
       The caption vector for the image we're interested in
       Should be unit vector
       
    confuserVec : numpy.ndarray (200,)
        A model output of a confuser image vector
        Should be unit vector
    
    margin : float, optional (default=1)
        Minimum difference between true-caption similarity and confuser-caption
        similarity for the model to have 0 loss
    
    Returns
    -------
    float
        returns loss for the function (higher if trueVec is far from captionVec or
        confuserVec is close to captionVec)
    """
    
    return np.max(margin - (trueVec @ captionVec - confuserVec @ captionVec), 0)
    
def batch_accuracy(batch_losses):
    """
    Given the losses for a batch, compute the accuracy for that batch
    
    Parameters
    ----------
    batch_losses : numpy.ndarray
    
    Returns
    -------
    float
        accuracy for the given batch
    """
    
    return np.sum(batch_losses==0)/batch_losses.size
