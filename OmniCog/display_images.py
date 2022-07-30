import io

import requests
from PIL import Image


def download_image(img_url: str) -> Image:
    """Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def display_images(k_ids, coco_data, resnet):
    """Displays the images that were returned by the query_database function. (I think)
    
    Parameters
    ----------
    k_similarity_scores : tuple
    The tuple returned by the query_database function.
    
    Returns
    -------
    PIL.Image
        The images. (hopefully)"""
    
    data = CocoDataManager(coco_data, resnet)
    urls = [data.getUrl(ids) for ids in k_ids]
    for img_url in urls:
        download_image(img_url)