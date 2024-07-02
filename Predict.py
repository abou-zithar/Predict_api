import os
import cv2
from ultralytics import YOLO,RTDETR
from datetime import datetime

Yolo_models_wrinkle = 'best(wrinkle-4).pt'
DETR_model_Dark_circles_Eye_bags ='best-Dark_circles-Eye_bags.pt'
DETR_model_acne = 'best(acne).pt'
parent_output_dir = 'D:/Projects Computer Vision/Computer vision Project 3 Skin-Disease-Detection/API/output_images_API'
class_index_to_predict_best_Dark_circles_Eye_bags = [2,3]
# Load the trained YOLO model
model_wrinkle = YOLO(Yolo_models_wrinkle)
model_Dark_circles = RTDETR(DETR_model_Dark_circles_Eye_bags)
model_acne = RTDETR(DETR_model_acne)
# Define a function to draw bounding boxes without labels
def draw_bounding_boxes(image, boxes, color=(255,215,0)):
    """
    Draws bounding boxes on the image without labels and with a specified color.
    
    :param image: The image on which to draw the bounding boxes.
    :param boxes: A list of bounding boxes, where each box is represented as [x_min, y_min, x_max, y_max].
    :param color: A tuple representing the color of the bounding box (B, G, R).
    """
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
    return image



def predict(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    # Perform predictions
    # Placeholder for model predictions
    results_wrinkle = model_wrinkle.predict(image_path)  # Replace with actual prediction call
    results_Dark_circles = model_Dark_circles.predict(image_path,classes=class_index_to_predict_best_Dark_circles_Eye_bags)  # Replace with actual prediction call
    results_acne = model_acne.predict(image_path)  # Replace with actual prediction call

    # Extract bounding boxes (placeholder for actual bounding box extraction)
    boxes_wrinkle = results_wrinkle[0].boxes.xyxy.cpu().numpy()
    boxes_Dark_circles = results_Dark_circles[0].boxes.xyxy.cpu().numpy()
    boxes_acne = results_acne[0].boxes.xyxy.cpu().numpy()

    # Draw bounding boxes on the original image
    image_with_boxes_W = draw_bounding_boxes(image.copy(), boxes_wrinkle)
    image_with_boxes_DC = draw_bounding_boxes(image.copy(), boxes_Dark_circles)
    image_with_boxes_A = draw_bounding_boxes(image.copy(), boxes_acne)

    # Generate a unique subdirectory name using the current timestamp
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    print(os.path.splitext(os.path.basename(image_path)))
    output_dir = os.path.join(parent_output_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{base_filename}")
    
    # Create the unique subdirectory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames for the processed images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    filename_wrinkle = f"{base_filename}_W.jpg"
    filename_DC = f"{base_filename}_DC.jpg"
    filename_acne = f"{base_filename}_A.jpg"

    # Construct output file paths
    output_img_path_W = os.path.join(output_dir, filename_wrinkle)
    output_img_path_DC = os.path.join(output_dir, filename_DC)
    output_img_path_A = os.path.join(output_dir, filename_acne)

    # Save images with bounding boxes
    cv2.imwrite(output_img_path_W, image_with_boxes_W)
    cv2.imwrite(output_img_path_DC, image_with_boxes_DC)
    cv2.imwrite(output_img_path_A, image_with_boxes_A)

    print(f"Predictions completed and saved in {output_dir}")

    return [output_img_path_W, output_img_path_DC, output_img_path_A]