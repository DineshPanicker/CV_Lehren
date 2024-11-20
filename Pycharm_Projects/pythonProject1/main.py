import cv2
import numpy as np


def manual_convolution(image, kernel):
    # Get the size of the kernel
    n = kernel.shape[0]
    m = (n + 1) // 2

    # Get the dimensions of the image
    rows, cols = image.shape

    # Initialize an output image
    output = np.zeros_like(image, dtype=np.float32)

    # Pad the image to handle borders
    padded_image = np.pad(image, pad_width=m - 1, mode='constant', constant_values=0)

    # Perform the convolution
    for x in range(rows):
        for y in range(cols):
            # Compute the sum over the kernel and the corresponding image patch
            value = 0
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    value += padded_image[x + i - m, y + j - m] * kernel[i - 1, j - 1]
            output[x, y] = value

    # Clip the values to be in the valid range for images
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


# Example usage
if __name__ == "__main__":
    # Load a grayscale image
    image = cv2.imread(r"A:\RWU\First Sem\CV\CV_Lehren\albert-einstein_gray.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image file not found.")
    else:
        # Define a 3x3 mean kernel
        kernel = np.ones((5, 5), dtype=np.float32) /25

        # Apply the convolution
        result = manual_convolution(image, kernel)

        # Display the results
        cv2.imshow("Original Image", image)
        cv2.imshow("Convoluted Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
