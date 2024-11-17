import os
import cv2
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(image, mean=0, std=50):
    """
    Add Gaussian noise to an image.

    Args:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with added noise.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(
        np.uint8
    )  # Clip values to valid range
    return noisy_image


def process_folder(input_folder, output_folder, noise_mean=0, noise_std=25):
    """
    Add Gaussian noise to all images in a folder and save them to a new folder.

    Args:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        noise_mean (float): Mean of the Gaussian noise.
        noise_std (float): Standard deviation of the Gaussian noise.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Could not read {input_path}")
                continue

            # Add noise
            noisy_image = add_gaussian_noise(image, mean=noise_mean, std=noise_std)

            # Save the noisy image
            cv2.imwrite(output_path, noisy_image)


# Define paths
input_folder = "./valid"
output_folder = "./valid_noisy"

# Add noise to images
process_folder(input_folder, output_folder, noise_mean=0, noise_std=25)
