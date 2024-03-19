from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def embed_watermark(original_img_path, watermark_img_path, output_img_path, extracted_watermark_path):
    # Load images
    original_img = Image.open(original_img_path)
    watermark_img = Image.open(watermark_img_path).convert("L")  # Convert to grayscale

    # Convert images to numpy arrays
    original_np = np.array(original_img)
    watermark_np = np.array(watermark_img)

    # Resize watermark image to match the size of the original image
    watermark_resized = watermark_img.resize(original_img.size, Image.LANCZOS)
    watermark_np = np.array(watermark_resized)

    # Convert watermark image to binary
    threshold = 128  # Adjust as needed
    binary_watermark = (watermark_np > threshold).astype(np.uint8)

    # Embedding process
    watermarked_np = np.copy(original_np)
    block_size = 8  # Adjust block size as needed
    for y in range(0, original_np.shape[0], block_size):
        for x in range(0, original_np.shape[1], block_size):
            block = original_np[y:y+block_size, x:x+block_size]
            watermark_block = binary_watermark[y:y+block_size, x:x+block_size]
            # Check if block exceeds image dimensions
            block_height, block_width = watermark_block.shape
            if block_height < block_size or block_width < block_size:
                continue  # Skip if block is smaller than block size
            # Embed watermark into LSB of block
            for i in range(block_size):
                for j in range(block_size):
                    if watermark_block[i, j] == 1:
                        watermarked_np[y+i, x+j] |= 1  # Set LSB to 1
                    else:
                        watermarked_np[y+i, x+j] &= 254  # Set LSB to 0

    # Save watermarked image
    watermarked_img = Image.fromarray(watermarked_np)
    watermarked_img.save(output_img_path, format="TIFF")
    
    # Visualization of original, watermarked, and difference images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(watermarked_np, cmap='gray')
    axes[1].set_title('Watermarked Image')
    axes[1].axis('off')

    difference_img = np.abs(original_np - watermarked_np).astype(np.uint8)
    axes[2].imshow(difference_img, cmap='gray')
    axes[2].set_title('Difference Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Save the difference image
    diff_img = Image.fromarray(np.abs(original_np - watermarked_np).astype(np.uint8))
    diff_img.save(extracted_watermark_path)

def verify_watermark(watermarked_img_path, original_img_path, extracted_watermark_path):
    # Load images
    watermarked_img = Image.open(watermarked_img_path)
    original_img = Image.open(original_img_path)

    # Convert images to numpy arrays
    watermarked_np = np.array(watermarked_img)
    original_np = np.array(original_img)
    
    if watermarked_np.shape != original_np.shape:
        raise ValueError("Watermarked and original images must have the same dimensions. Tampering detected.")

    # Calculate the difference between watermarked and original images
    extracted_watermark_np = original_np - watermarked_np 

    # Threshold the difference to obtain binary watermark information
    threshold = 50  # Adjust as needed
    extracted_watermark_binary = np.where(extracted_watermark_np > threshold, 1, 0)

    # Save the extracted watermark image
    extracted_watermark_img = Image.fromarray(extracted_watermark_binary.astype(np.uint8) * 255)  # Convert to 8-bit
    extracted_watermark_img.save(extracted_watermark_path)
    
    return extracted_watermark_binary

def matching_percentage(extracted_watermark_np, grayscale_watermark_path):
    # Load grayscale watermark image
    grayscale_watermark_img = Image.open(grayscale_watermark_path).convert("L")

    # Resize grayscale watermark image to match the size of the extracted watermark
    extracted_watermark_img = Image.fromarray(extracted_watermark_np.astype(np.uint8) * 255)  # Convert to 8-bit
    grayscale_watermark_resized = grayscale_watermark_img.resize(extracted_watermark_img.size, Image.LANCZOS)

    # Convert extracted watermark to grayscale
    extracted_watermark_gray = extracted_watermark_img.convert("L")

    # Calculate the number of matching pixels
    matching_pixels = np.sum(np.array(extracted_watermark_gray) == np.array(grayscale_watermark_resized))

    # Calculate matching percentage
    total_pixels = extracted_watermark_img.size[0] * extracted_watermark_img.size[1]

    # Calculate matching percentage
    percentage = (matching_pixels / total_pixels) * 100

    # Plotting comparison between extracted and original watermarks
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(extracted_watermark_img, cmap='gray')
    axes[0].set_title('Extracted Watermark')
    axes[0].axis('off')

    axes[1].imshow(grayscale_watermark_resized, cmap='gray')
    axes[1].set_title('Original Watermark')
    axes[1].axis('off')

    difference_img = np.abs(np.array(extracted_watermark_gray) - np.array(grayscale_watermark_resized))
    axes[2].imshow(difference_img, cmap='gray')
    axes[2].set_title('Difference')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    if percentage > 80:
        print("Watermark is verified, matching percentage:", percentage)
    else:
        print("Watermark is not verified, matching percentage:", percentage)

# Example usage
def tamper_detection(input, original_img_path, extracted_watermark_path, grayscale_watermark_path):
    try:
        extracted_watermark_np = verify_watermark(input, original_img_path, extracted_watermark_path)
    except ValueError as e:
        print(e)
        return
    matching_percentage(extracted_watermark_np, grayscale_watermark_path)
    
original_img_path = "./main_images/input_image.jpg"
watermark_img_path = "./main_images/watermark_image.png"
output_img_path = "./temp_files/watermarked_image.TIFF"
extracted_watermark_path = "./temp_files/extracted_watermark.png"
grayscale_watermark_path = "./main_images/watermark_image.png"
other_img = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
embed_watermark(original_img_path, watermark_img_path, output_img_path, extracted_watermark_path)
tamper_detection(output_img_path, original_img_path, extracted_watermark_path, grayscale_watermark_path)
print("------------------------------------------------")
print("------------------------------------------------")
print("------------------------------------------------")
for i in other_img:
    tamper_detection("test_images/"+i, original_img_path, extracted_watermark_path, grayscale_watermark_path)
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("------------------------------------------------")