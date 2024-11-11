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

# Resize the poster image to make it larger relative to the markers
poster_scale_factor = 12  # Increase the initial scale to make the poster larger overall
poster_img = cv2.resize(poster_img, (poster_width * poster_scale_factor, poster_height * poster_scale_factor))
poster_height, poster_width = poster_img.shape[:2]  # Update dimensions


# Adjust detector parameters (you can tweak these further based on your needs)
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.04
parameters.polygonalApproxAccuracyRate = 0.03


# Function to detect ArUco markers and overlay poster with larger size
def process_image(image_path):
    # Read and convert image to grayscale
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

    # Check if any markers were detected
    if ids is not None:
        print(f"Detected marker IDs: {ids.flatten()}")  # Print IDs for debugging

        # Draw detected markers
        image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

        # Overlay poster on each detected marker
        for corner in corners:
            pts_dst = corner[0].astype(np.float32)  # Corner points of detected marker

            # Calculate the center of the marker
            marker_center = np.mean(pts_dst, axis=0)

            # Scale factor to enlarge the poster on the marker
            scale_factor = 4  # Adjust this factor to make the poster larger or smaller

            # Expand the destination points outward
            pts_dst_expanded = []
            for point in pts_dst:
                # Move the point outward from the center by the scale factor
                expanded_point = marker_center + scale_factor * (point - marker_center)
                pts_dst_expanded.append(expanded_point)
            pts_dst_expanded = np.array(pts_dst_expanded, dtype=np.float32)

            # Define source points from the resized poster image
            pts_src = np.array([[0, 0], [poster_width, 0], [poster_width, poster_height], [0, poster_height]],
                               dtype=np.float32)

            # Calculate homography matrix with expanded destination points
            h, _ = cv2.findHomography(pts_src, pts_dst_expanded)

            # Warp the poster image to match the ArUco marker position
            warped_poster = cv2.warpPerspective(poster_img, h, (image.shape[1], image.shape[0]))

            # Create a mask from the warped poster
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(pts_dst_expanded), (255, 255, 255))

            # Use mask to overlay poster onto the original image
            image_with_markers = cv2.bitwise_and(image_with_markers, 255 - mask) + cv2.bitwise_and(warped_poster, mask)

        return image_with_markers
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

folder_path = "A:\\RWU\\First Sem\\CV\\Output folder"
def open_images_in_photos_app(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Loop through each file and open it in the default image viewer
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)
        img.show()  # This opens the image in the default image viewer


open_images_in_photos_app(folder_path)
