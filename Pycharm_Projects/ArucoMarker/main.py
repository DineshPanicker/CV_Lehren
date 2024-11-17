import os
import numpy as np
import cv2
from PIL import Image
# Define paths
dataset_folder = "A:\\RWU\\First Sem\\CV\\Room with ArUco Markers-20241024"  # Replace with your dataset path
poster_path = "A:\\RWU\\First Sem\\CV\\Poster\\Switzerland-new.jpg"  # Replace with your poster image path
output_folder = "A:\\RWU\\First Sem\\CV\\Output folder"  # Define output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load ArUco dictionary and detection parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()

# Load the poster image
poster_img = cv2.imread(poster_path)
if poster_img is None:
    print("Error: Poster image not found. Check the poster_path.")
    exit()

# Get the original dimensions of the poster image
poster_height, poster_width = poster_img.shape[:2]

# Function to detect ArUco markers and overlay the scaled poster
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

        # Overlay the poster on each detected marker
        for corner in corners:
            pts_dst = corner[0].astype(np.float32)  # Corner points of the detected marker

            # Calculate the center of the marker
            marker_center = np.mean(pts_dst, axis=0)

            # Scale factor to enlarge the poster relative to the marker
            scale_factor = 3  # Adjust this factor to make the poster larger or smaller

            # Expand the destination points outward
            pts_dst_scaled = []
            for point in pts_dst:
                # Move the point outward from the center by the scale factor
                scaled_point = marker_center + scale_factor * (point - marker_center)
                pts_dst_scaled.append(scaled_point)
            pts_dst_scaled = np.array(pts_dst_scaled, dtype=np.float32)

            # Define source points from the original poster image
            pts_src = np.array([
                [0, 0],
                [poster_width - 1, 0],
                [poster_width - 1, poster_height - 1],
                [0, poster_height - 1]
            ], dtype=np.float32)

            # Calculate homography matrix with the scaled destination points
            h_matrix, _ = cv2.findHomography(pts_src, pts_dst_scaled)

            # Warp the poster to match the scaled marker's perspective
            warped_poster = cv2.warpPerspective(poster_img, h_matrix, (image.shape[1], image.shape[0]))

            # Create a mask from the warped poster
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(pts_dst_scaled), (255, 255, 255))

            # Use the mask to overlay the poster onto the original image
            image = cv2.bitwise_and(image, 255 - mask) + cv2.bitwise_and(warped_poster, mask)

        return image
    else:
        print("No markers detected.")
        return image  # Return the original image if no markers are detected


# Process each image in the dataset
images = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder) if img.endswith('.jpg')]
for img_path in images:
    ar_image = process_image(img_path)
    if ar_image is not None:
        # Define the output path for the processed image
        output_path = os.path.join(output_folder, os.path.basename(img_path))

        # Save the processed image
        cv2.imwrite(output_path, ar_image)
        print(f"Processed image saved at: {output_path}")


# Function to preview images in the default image viewer
def open_images_in_photos_app(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Loop through each file and open it in the default image viewer
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)
        img.show()  # Open the image in the default image viewer


# Preview the processed images
open_images_in_photos_app(output_folder)