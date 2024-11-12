import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh
image = cv2.imread('ggmap3.png', cv2.IMREAD_GRAYSCALE)

# Bộ lọc Gaussian để giảm nhiễu
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Toán tử Sobel
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Theo chiều x
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Theo chiều y
sobel = cv2.magnitude(sobelx, sobely)  # Độ lớn của gradient

# Toán tử Prewitt (dùng convolution kernels cho Prewitt)
prewittx = cv2.filter2D(blurred, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))  # Chiều x
prewitty = cv2.filter2D(blurred, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  # Chiều y
prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))

# Toán tử Roberts (dùng convolution kernels cho Roberts)
robertsx = cv2.filter2D(blurred, -1, np.array([[1, 0], [0, -1]]))
robertsy = cv2.filter2D(blurred, -1, np.array([[0, 1], [-1, 0]]))
roberts = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))

# Toán tử Canny
canny = cv2.Canny(blurred, 100, 200)

# Hiển thị kết quả
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Ảnh gốc')
axes[0].axis('off')

axes[1].imshow(sobel, cmap='gray')
axes[1].set_title('Sobel')
axes[1].axis('off')

axes[2].imshow(prewitt, cmap='gray')
axes[2].set_title('Prewitt')
axes[2].axis('off')

axes[3].imshow(roberts, cmap='gray')
axes[3].set_title('Roberts')
axes[3].axis('off')

axes[4].imshow(canny, cmap='gray')
axes[4].set_title('Canny')
axes[4].axis('off')

axes[5].imshow(blurred, cmap='gray')
axes[5].set_title('Gaussian Blurred')
axes[5].axis('off')

plt.tight_layout()
plt.show()