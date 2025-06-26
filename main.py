import os
from typing import Dict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImageUtils:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError
        return np.array(Image.open(image_path))

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    @staticmethod
    def save_image(image_path: str, image: np.ndarray, do_normalization: bool = False) -> None:
        if image.dtype != np.uint8:
            if do_normalization:
                image = ImageUtils.normalize_image(image).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return Image.fromarray(image).save(image_path)


class LBPUtils:
    @staticmethod
    def _is_uniform_pattern(binary_pattern: list) -> bool:
        """Check if the binary pattern is uniform (≤2 transitions)"""
        transitions = 0
        for i in range(8):
            if binary_pattern[i] != binary_pattern[(i + 1) % 8]:
                transitions += 1
        return transitions <= 2

    @staticmethod
    def _get_uniform_pattern_id(binary_pattern: list) -> int:
        """Get uniform pattern ID (0-57) or 58 for non-uniform"""
        if not LBPUtils._is_uniform_pattern(binary_pattern):
            return 58  # Non-uniform bin

        # Count number of 1s in the pattern
        ones_count = sum(binary_pattern)
        return ones_count

    @staticmethod
    def calculate_lbp(
            image: np.ndarray,
            rotation_invariant_lbp: bool = False,
            uniform_patterns: bool = False
    ) -> np.ndarray:
        """
        Simple Local Binary Pattern implementation
        """
        h, w = image.shape
        lbp_image = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # Process each pixel (excluding border)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]

                # Get 8 neighbors in clockwise order
                neighbors = [
                    image[i - 1, j - 1],  # top-left
                    image[i - 1, j],  # top
                    image[i - 1, j + 1],  # top-right
                    image[i, j + 1],  # right
                    image[i + 1, j + 1],  # bottom-right
                    image[i + 1, j],  # bottom
                    image[i + 1, j - 1],  # bottom-left
                    image[i, j - 1]  # left
                ]

                # Create a binary pattern
                binary_pattern = []
                for neighbor in neighbors:
                    if neighbor >= center:
                        binary_pattern.append(1)
                    else:
                        binary_pattern.append(0)

                # Calculate LBP value based on method
                if uniform_patterns:
                    lbp_value = LBPUtils._get_uniform_pattern_id(binary_pattern)

                elif rotation_invariant_lbp:
                    min_value = float('inf')

                    for rotation in range(8):
                        current_value = 0
                        for k in range(8):
                            current_value += binary_pattern[(k + rotation) % 8] * (2 ** k)
                        min_value = min(min_value, current_value)
                    lbp_value = min_value

                else:
                    lbp_value = 0
                    for k in range(8):
                        lbp_value += binary_pattern[k] * (2 ** k)

                lbp_image[i - 1, j - 1] = lbp_value

        return lbp_image

    @staticmethod
    def calculate_lbp_histogram(lbp_image, uniform_patterns: bool = False):
        """Calculate histogram of LBP values"""
        if uniform_patterns:
            hist = np.zeros(59)  # 59 bins for uniform LBP
        else:
            hist = np.zeros(256)  # 256 bins for regular LBP

        for i in range(lbp_image.shape[0]):
            for j in range(lbp_image.shape[1]):
                hist[lbp_image[i, j]] += 1
        return hist


class Helper:
    @staticmethod
    def create_directories(directories: Dict[str, str]) -> None:
        for path in directories.values():
            os.makedirs(path, exist_ok=True)
        return


def main():
    directories = {
        "main_path": "Images",
    }
    original_image_path = os.path.join(directories["main_path"], "original.bmp")

    # ---------------------------------------------------------------------------------
    # Load Original Image
    # ---------------------------------------------------------------------------------
    original_image = ImageUtils.load_image(original_image_path)
    print(50 * "=", "\nOriginal Image Array:\n", original_image)

    # Apply different LBP methods
    lbp_basic = LBPUtils.calculate_lbp(original_image, rotation_invariant_lbp=False, uniform_patterns=False)
    lbp_uniform = LBPUtils.calculate_lbp(original_image, rotation_invariant_lbp=False, uniform_patterns=True)
    lbp_ri = LBPUtils.calculate_lbp(original_image, rotation_invariant_lbp=True, uniform_patterns=False)

    # Calculate histograms
    hist_basic = LBPUtils.calculate_lbp_histogram(lbp_basic, uniform_patterns=False)
    hist_uniform = LBPUtils.calculate_lbp_histogram(lbp_uniform, uniform_patterns=True)
    hist_ri = LBPUtils.calculate_lbp_histogram(lbp_ri, uniform_patterns=False)

    # Calculate original image histogram
    hist_original = np.histogram(original_image.flatten(), bins=256, range=(0, 256))[0]

    # Display results
    plt.figure(figsize=(16, 10))

    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.plot(hist_original)
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Basic LBP
    plt.subplot(2, 4, 2)
    plt.imshow(lbp_basic, cmap='gray')
    plt.title('Basic LBP')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.plot(hist_basic)
    plt.title('Basic LBP Histogram (256 bins)')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')

    # Uniform LBP
    plt.subplot(2, 4, 3)
    plt.imshow(lbp_uniform, cmap='gray')
    plt.title('Uniform LBP')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.bar(range(59), hist_uniform)
    plt.title('Uniform LBP Histogram (59 bins)')
    plt.xlabel('Pattern ID')
    plt.ylabel('Frequency')

    # Rotation-invariant LBP
    plt.subplot(2, 4, 4)
    plt.imshow(lbp_ri, cmap='gray')
    plt.title('Rotation-Invariant LBP')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.plot(hist_ri)
    plt.title('RI LBP Histogram (256 bins)')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print(f"Original image shape: {original_image.shape}")
    print(f"Basic LBP histogram sum: {np.sum(hist_basic)}")
    print(f"Uniform LBP histogram sum: {np.sum(hist_uniform)} (59 bins)")
    print(f"Rotation-invariant LBP histogram sum: {np.sum(hist_ri)}")

    # Show uniform pattern statistics
    uniform_count = np.sum(hist_uniform[:58])  # Uniform patterns (bins 0-57)
    non_uniform_count = hist_uniform[58]  # Non-uniform patterns (bin 58)
    total_pixels = uniform_count + non_uniform_count
    print(f"Uniform patterns: {uniform_count}/{total_pixels} ({uniform_count / total_pixels * 100:.1f}%)")
    print(f"Non-uniform patterns: {non_uniform_count}/{total_pixels} ({non_uniform_count / total_pixels * 100:.1f}%)")


if __name__ == '__main__':
    main()