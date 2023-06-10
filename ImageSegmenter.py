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
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        mask_all = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]

            m = [m for x in range(3)]
            m = np.swapaxes(m, 0, 2)
            m = np.swapaxes(m, 0, 1)
            img = img * m
            mask_all.append(img)

        mask_all = np.array(mask_all).sum(axis=0)
        mask_all *= 255
        save_img = image / 2 + mask_all
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



    def filter_segments(self,anns, max_area, min_area, ratio):

        """



        """
        for ann in anns:
            w,h = ann['bbox'][2], ann['bbox'][3]
            if ann['segmentation'] < max_area and ann['segmentation'] > min_area:
                filtered.append(ann)
        return filtered

    def classify_lands():

    def save_masks_coco(anns):
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
