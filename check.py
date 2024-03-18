import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images side by side
def display_images(img1, img2, title1='Original', title2='Watermarked'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.show()

# Function to compare histograms
def compare_histograms(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    plt.plot(hist1, color='b', label='Original')
    plt.plot(hist2, color='r', label='Watermarked')
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Load original and watermarked images
original_image = cv2.imread("input_3.jpg")
watermarked_image = cv2.imread("watermarked_image.jpg")

# Visual Inspection
display_images(original_image, watermarked_image)

# Histogram Analysis
compare_histograms(original_image, watermarked_image)
