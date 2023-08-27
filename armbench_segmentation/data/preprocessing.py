import os
def load_set(coco, local_filepath, image_list, load_union=False):
    # get all images containing given categories
    CATEGORIES = []
    catIds = coco.getCatIds(CATEGORIES)  # Fetch class IDs only corresponding to the Classes
    if not load_union:
        imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs together
    else:  # get images contains any of the classes
        imgIds = set()
        for cat_id in catIds:
            image_ids = coco.getImgIds(catIds=[cat_id])
            imgIds.update(image_ids)
        imgIds = list(imgIds)[:-1]  # we're missing the last image for some reason
    imgs = coco.loadImgs(imgIds)

    imgs = [img for img in imgs if img['file_name'] in image_list]
    return imgs
