import os
import numpy as np
import cv2
from cv2 import aruco

# Define paths
dataset_folder = '/home/dinesh/Room with ArUco Markers-20241024'  # Replace with your dataset path
poster_path = '/home/dinesh/Poster/Switzerland-new.jpg'  # Replace with your poster image path
output_folder = '/home/dinesh/Output Folder'  # Define output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load ArUco dictionary and detection parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Load the poster image
poster_img = cv2.imread(poster_path)
if poster_img is None:
    print("Error: Poster image not found. Check the poster_path.")
    exit()

# Get the original dimensions of the poster image
poster_height, poster_width = poster_img.shape[:2]

# Function to detect ArUco markers and overlay the poster
def process_image(image_path):
    # Read and convert the image to grayscale
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

    # Check if any markers were detected
    if ids is not None:
        print(f"Detected marker IDs: {ids.flatten()}")

        # Process each detected marker
        for marker_corners in corners:
            marker_corners = marker_corners[0].astype(np.float32)

            # Calculate the center of the marker
            marker_center = np.mean(marker_corners, axis=0)

            # Define poster corners relative to the poster's dimensions
            w, h = poster_width / 6, poster_height / 6  # Scale factors for resizing
            corner1 = (poster_width / 2 - w / 2, poster_height / 2 - h / 2)
            corner2 = (poster_width / 2 + w / 2, poster_height / 2 - h / 2)
            corner3 = (poster_width / 2 + w / 2, poster_height / 2 + h / 2)
            corner4 = (poster_width / 2 - w / 2, poster_height / 2 + h / 2)

            # Define input and output points for perspective transform
            input_corners = np.float32([corner1, corner2, corner3, corner4])
            output_corners = np.float32([marker_corners[0], marker_corners[1], marker_corners[2], marker_corners[3]])

            # Calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(input_corners, output_corners)

            # Warp the poster to fit the marker's perspective
            transformed_poster = cv2.warpPerspective(poster_img, M, (image.shape[1], image.shape[0]))

            # Create a mask from the warped poster
            pts = np.array(output_corners, np.int32)  # Convert output corners to integer
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))

            # Invert the mask to isolate the area outside the polygon
            mask_inverted = cv2.bitwise_not(mask)

            # Apply the mask to the original image
            image = cv2.bitwise_and(image, mask_inverted)

            # Overlay the transformed poster onto the original image
            image = cv2.bitwise_or(image, transformed_poster)

        return image
    else:
        print("No markers detected.")
        return image  # Return the original image if no markers are detected


# Process each image in the dataset
images = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder) if img.endswith('.jpg')]
for img_path in images:
    processed_image = process_image(img_path)
    if processed_image is not None:
        # Define the output path for the processed image
        output_path = os.path.join(output_folder, os.path.basename(img_path))

        # Save the processed image
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved at: {output_path}")
