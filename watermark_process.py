import cv2
import numpy as np

# Step 1: Image Block Partitioning
def block_partition(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)
    return blocks

# Step 2: Binary Watermark Image Creation
def create_binary_watermark(image_shape):
    watermark = np.random.randint(0, 2, image_shape[:2], dtype=np.uint8) * 255
    return watermark

# Step 3: Watermark Image Block Partitioning
def watermark_block_partition(watermark, block_size):
    return block_partition(watermark, block_size)

# Step 4: Embedding Watermark into Image (Simple LSB substitution)
def embed_watermark(image, watermark_blocks):
    watermarked_image = image.copy()
    for i, block in enumerate(watermark_blocks):
        x = (i * block.shape[1]) % image.shape[1]
        y = (i * block.shape[0]) // image.shape[1] * block.shape[0]
        
        # Check if the block fits within the image dimensions
        if y + block.shape[0] <= image.shape[0] and x + block.shape[1] <= image.shape[1]:
            for row in range(block.shape[0]):
                for col in range(block.shape[1]):
                    # Replace LSB of each pixel in the block with the corresponding bit from the watermark
                    pixel_value = image[y + row, x + col]
                    watermark_pixel = block[row, col]
                    watermarked_pixel = pixel_value & 0b11111110 | (watermark_pixel >> 7)
                    watermarked_image[y + row, x + col] = watermarked_pixel
    return watermarked_image


# Load input image
input_image = cv2.imread("input_3.jpg", cv2.IMREAD_COLOR)

# Step 1: Image Block Partitioning
block_size = 8  # You can adjust this block size
image_blocks = block_partition(input_image, block_size)

# Step 2: Binary Watermark Image Creation
binary_watermark = create_binary_watermark(input_image.shape)

# Step 3: Watermark Image Block Partitioning
watermark_blocks = watermark_block_partition(binary_watermark, block_size)

# Step 4: Embedding Watermark into Image
watermarked_image = embed_watermark(input_image, watermark_blocks)

# Save the watermarked image
cv2.imwrite("watermarked_image.jpg", watermarked_image)
