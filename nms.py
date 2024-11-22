import numpy as np

def iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    box: [x_center, y_center, width, height]
    """
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def non_maximum_suppression(predictions, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression.
    predictions: List of predictions where each prediction is a tuple 
                 (class_id, x_center, y_center, width, height, confidence_score).
    iou_threshold: Threshold for IoU to suppress overlapping boxes.
    """
    predictions = sorted(predictions, key=lambda x: x[5], reverse=True)  # Sort by confidence score (descending)
    
    keep = []  # List to store final bounding boxes
    
    while predictions:
        chosen_box = predictions.pop(0)  # Take the box with the highest confidence
        keep.append(chosen_box)
        
        predictions = [
            box for box in predictions
            if box[0] != chosen_box[0] or iou(chosen_box[1:5], box[1:5]) <= iou_threshold
        ]
    
    return keep

def get_predictions(path_pred):
    with open(path_pred, "r") as f:
        final = [f.rstrip().split(" ") for f in f.readlines()]

        predictions = {}

        for item in final:
            if item[0] not in predictions.keys():
                predictions[item[0]] = []

            predictions[item[0]].append([
                int(item[1]), float(item[2]), float(item[3]),
                float(item[4]), float(item[5]), float(item[6])
            ])

        return predictions

# Example usage
path_pred = "./output/annotations.txt"
predictions = get_predictions(path_pred)

with open("./output/final_anno.txt", "w") as f:
    for key, item in predictions.items():
        filtered_boxes = non_maximum_suppression(item, iou_threshold=0.5)
        for box in filtered_boxes:
            f.write(key)
            for idx, subbox in enumerate(box):
                f.write(" ")
                if idx == 0:
                    f.write("{}".format(int(subbox)))
                else:
                    f.write("{:.4f}".format(float(subbox)))
            f.write("\n")
            
        print(len(filtered_boxes))
