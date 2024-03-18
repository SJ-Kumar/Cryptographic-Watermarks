import numpy as np
from PIL import Image

def block_partition(image, block_size):
    """
    Divide an image into non-overlapping blocks of a specified size.

    Args:
    - image: Input image (numpy array or PIL Image object).
    - block_size: Size of the blocks (tuple of width and height).

    Returns:
    - List of block images (numpy arrays).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]
    block_width, block_height = block_size
    num_blocks_horizontal = width // block_width
    num_blocks_vertical = height // block_height

    blocks = []
    for y in range(num_blocks_vertical):
        for x in range(num_blocks_horizontal):
            block = image[y * block_height: (y + 1) * block_height,
                          x * block_width: (x + 1) * block_width]
            blocks.append(block)

    return blocks

def create_binary_watermark(watermark_image, threshold=128):
    """
    Generate a binary watermark image based on a grayscale watermark image and a threshold.

    Args:
    - watermark_image: Grayscale watermark image (numpy array or PIL Image object).
    - threshold: Threshold value for binarization (default is 128).

    Returns:
    - Binary watermark image (numpy array).
    """
    if isinstance(watermark_image, Image.Image):
        watermark_image = np.array(watermark_image.convert("L"))

    binary_watermark = (watermark_image > threshold).astype(np.uint8) * 255

    return binary_watermark
