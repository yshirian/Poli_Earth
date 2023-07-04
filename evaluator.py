

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

class evaluate_metrics():




    import numpy as np

    def calculate_f1_score(ground_truth, predicted):
        '''
        calculating f1 score for segmentations

        Args:
            ground-truth: ground truth (numpy.ndarray)
            predicted: output of SAM (numpy.ndarray)

        Return:
            f1_score


        '''

        intersection = np.logical_and(ground_truth, predicted)
        true_positive = np.sum(intersection)

        ground_truth_count = np.sum(ground_truth)
        predicted_count = np.sum(predicted)

        precision = true_positive / predicted_count if predicted_count > 0 else 0
        recall = true_positive / ground_truth_count if ground_truth_count > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_score



    def calculate_f1_score(ground_truth, predicted):
        '''
        calculating f1 score for segmentations

        Args:
            ground-truth: ground truth (numpy.ndarray)
            predicted: output of SAM (numpy.ndarray)

        Return:
            f1_score


        '''

        intersection = np.logical_and(ground_truth, predicted)
        true_positive = np.sum(intersection)

        ground_truth_count = np.sum(ground_truth)
        predicted_count = np.sum(predicted)

        precision = true_positive / predicted_count if predicted_count > 0 else 0
        recall = true_positive / ground_truth_count if ground_truth_count > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_score




        def calcualte_IOU(y_pred, y_true):

            '''
                calculating IoU

                Args:

                    y_pred ( num_batch, num_classes, ....)
                    y_true (num_batch, ....)

                return
                    iou:
            '''

            cm = ConfusionMatrix(num_classes=3)
            metric = IoU(cm)
            metric.attach(default_evaluator, 'iou')

            state = default_evaluator.run([[y_pred, y_true]])

            return state['iou']
