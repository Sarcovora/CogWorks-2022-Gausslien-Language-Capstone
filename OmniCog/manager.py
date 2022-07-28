class CocoDataManager:
    """
    Attributes
    ----------
    images : list
        list of image dictionaries
    captions : list
        list of caption dictionaries
    imageID_to_captionID : dictionary {int : int}
        dictionary that stores image IDs as the key and caption IDs as the value
    captionID_to_imgID : dictionary {int : int}
        dictionary that stores captions IDs as the key and image IDs as the value
    captionID_to_caption : dictionary {int : str}
        dictionary that stores captions IDs as the key and a caption as the value
    caption_to_captionID : dictionary {str : int}
        dictionary that stores captions as the key and a caption IDs as the value
    imageID_to_url : dictionary {int : str}
        dictionary that stores image IDs as the key and a coco URL as the value
    imageIDs : list
        unsorted list of all image IDs
    
    Methods
    -------
    getCaptionIDs(self, img_id)
        Gives a list of caption IDs given an image ID
    getUrl(self, img_id)
        Gives a coco url given an image ID
    """
    
    def __init__ (self, coco_data):
        """
        Initialize a CocoDataManager class instance

        Parameters
        ----------
        coco_data : dictionary
            The loaded json file as a dictionary
        """
        
        self.images = coco_data["images"]
        self.captions = coco_data["annotations"]
        
        self.imageID_to_captionID = {}
        self.captionID_to_imgID = {}
        
        self.captionID_to_caption = {}
        self.caption_to_captionID = {}
        
        self.imageID_to_url = {}
        
        self.imageIDs = [i["id"] for i in self.images]
        
        for c in self.captions:
            cap_id = c["id"]
            img_id = c["image_id"]
            cap = c["caption"]
            
            # adding data to the imageID to captionID dictionary
            if (img_ID in self.imageID_to_captionID):
                self.imageID_to_captionID[img_id].append(cap_id)
            else:
                self.imageID_to_captionID[img_id] = [cap_id]
            
            # adding data to the captionID to imageID dictionary
            self.captionID_to_imgID[cap_id] = img_id
            
            # adding data to the captionID to caption dictionary
            self.captionID_to_caption[cap_id] = cap
            
        # adding data to the caption to captionID dictionary
        self.caption_to_captionID = {value:key for key, value in captionID_to_caption.items()}
        
        for img in self.images:
            img_id = img["id"]
            img_url = img["coco_url"]
            img_height = img["height"]
            img_width = img["width"]
        
            # adding data to the imageID to url dictionary
            self.imageID_to_url[img_id] = img_url
                
    def getCaptionIDs (self, img_id):
        """
        Gives a list of caption IDs given an image ID

        Parameters
        ----------
        img_id : int
            Id of the image to be accessed

        Returns
        -------
        list
            List of length 4 that contains the caption ids
        """
        return self.imageID_to_captionID[img_id]
    
    def getUrl (self, img_id):
        """
        Gives a coco url given an image ID

        Parameters
        ----------
        img_id : int
            Id of the image to be accessed

        Returns
        -------
        String
            The coco URL used to access/download an image
        """
        return self.imageID_to_url[img_id]
