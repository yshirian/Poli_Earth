import numpy as np
from PIL import Image
import urllib.request
import torch
import cv2
import requests
import json



class ImageSegmenter:
    def __init__(self):
        """
        Initializes the ImageSegmenter class by setting up the device and loading the segmentation model.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint).to(device=self.device)

    def save_anno(self, image, anns, points_per_side, location):
        """
        Saves the annotated image with overlaid masks.

        Args:
            image: The original image as a numpy array.
            anns: A list of annotations containing the segmentations and bounding boxes.
            points_per_side: The number of points per side used for segmentation.
            location: The location or name of the image.

        Returns:
            None
        """
        #plt.figure(figsize=(20,20))
        #plt.imshow(image)
        #ax = plt.gca()
        #ax.set_autoscale_on(False)
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        img = np.ones((anns[0]['segmentations'].shape[0], anns[0]['segmentations'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            mask = np.concatenate([np.random.random(3),[0.35]])
            img[m] = mask

        #ax.imshow(mask)
        save_image = mask + image
        save_img = Image.fromarray(save_img.astype(np.uint8))
        save_img.save(f"output/{location}_{points_per_side}.png")

    def convert_box_xywh_to_xyxy(self, box):
        """
        Converts a bounding box from the format [x, y, width, height] to [x1, y1, x2, y2].

        Args:
            box: The bounding box coordinates in [x, y, width, height] format.

        Returns:
            The converted bounding box coordinates in [x1, y1, x2, y2] format.
        """
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def box_to_center(self, box):
        """
        Converts a bounding box from [x, y, width, height] format to [center_x, center_y] format.

        Args:
            box: The bounding box coordinates in [x, y, width, height] format.

        Returns:
            The converted bounding box coordinates in [center_x, center_y] format.
        """
        xc = box[0] + box[2] / 2
        yc = box[1] + box[1] / 2
        return [xc, yc]



    def filter_segments(self,anns, max_area, min_area):

        """
        Filters too big or too small masks
        Args:
            anns: annotations.
            max_area: maximum mask area in pxls^2
            min_area: minimum mask area in pxls^2

        Returns:
            filtered masks annotations.
        """
        for ann in anns:
            w,h = ann['bbox'][2], ann['bbox'][3]
            if ann['segmentation'] < max_area and ann['segmentation'] > min_area:
                filtered.append(ann)
        return filtered

    def classify_lands():
        '''
        To write this later based on pytorch and alexnet clasiification


        '''
        from torchvision.io import read_image
        from torchvision.models import resnet50, ResNet50_Weights

        img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")
        pass



    def make_coco_format(image_path,anns, data_all, image_id):
        '''
            saving annotations into coco format file.

            images:
                "file_name" , "height", "width", "meta_data", "id"
            annotations
                "bbox", "image_id", "category_id", "segmentation", "area", "iscrowd"
            categories
                "id", "supercategory", "name", "isthing"


        Args:
            image_path: path to the image

            anns: annottaions in format
                    "segmentation" : the mask
                    "area" : the area of the mask in pixels
                    "bbox" : the boundary box of the mask in XYWH format
                    "predicted_iou" : the model's own prediction for the quality of the mask
                    "point_coords" : the sampled input point that generated this mask
                    "stability_score" : an additional measure of mask quality
                    "crop_box" : the crop of the image used to generate this mask in XYWH format
            data_all:
                json dictionary including "images", "annotations", "categories"
        Return:
            coco format json dictionary


        '''
        from torchvision.ops import masks_to_boxes


        img = {}
        annotations = []
        image = cv.imread(image_path)
        imgs["file_name"] = image_path.split('/')[-1]
        imgs["width"]  = image.shape[0]
        imgs["height"] = image.shape[1]
        imgs["meta_data"] = {}
        for ann in anns:
            curanns ={}
            boxes = masks_to_boxes(masks)
            curanns['bbox'] = [boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1]]
            curanns['area'] = anns['area']
            curanns['segmentations'] = anns['segmentation']
            curanns['image_id'] = image_id
            curanns['iscrowd'] = 0

            annotations.append(curanns)


        data_all['annotations'].append(annotations)
        data_all['images'].append(img)

        return data_all
