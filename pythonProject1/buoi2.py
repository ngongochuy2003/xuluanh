import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('phongcanh.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = np.uint8(sobel)

# Apply Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(image, (3, 3), 0)
log = cv2.Laplacian(blurred, cv2.CV_64F)
log = np.uint8(np.absolute(log))

# Display the results
titles = ['Original Image', 'Sobel Edge Detection', 'LoG Edge Detection']
images = [image, sobel, log]

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()