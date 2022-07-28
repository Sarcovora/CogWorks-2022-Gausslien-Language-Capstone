def display_images(k_similarity_scores):
    """Displays the images that were returned by the query_database function. (I think)
    
    Parameters
    ----------
    k_similarity_scores : tuple
    The tuple returned by the query_database function.
    
    Returns
    -------
    PIL.Image
        The images. (hopefully)"""
    
    data = CocoDataManager(coco_data)
    urls = [data.getUrl(ids) for ids in k_similarity_scores[1]]
    for img_url in urls:
        download_image(img_url)