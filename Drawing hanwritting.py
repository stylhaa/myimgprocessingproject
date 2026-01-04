import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_multiotsu

image_path = r"C:\Users\user\.spyder-py3\drawing.jpg"
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    print("Error: Image not found. Please check the file path.")
    exit()

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, otsu = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

thresholds = threshold_multiotsu(blur, classes=3)
regions = np.digitize(blur, bins=thresholds)

multi_otsu = (regions * (255 / regions.max())).astype(np.uint8)

pixels = blur.reshape((-1, 1)).astype(np.float32)
K = 3

_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

kmeans_img = centers[labels.flatten()].reshape(gray.shape).astype(np.uint8)

fig, axes = plt.subplots(1, 6, figsize=(24, 6))

axes[0].set_title("Original Grayscale")
axes[0].imshow(gray, cmap="gray")
axes[0].axis("off")

axes[1].set_title("Gaussian Blur")
axes[1].imshow(blur, cmap="gray")
axes[1].axis("off")

axes[2].set_title("Single Otsu\n(One Stroke Class)")
axes[2].imshow(otsu, cmap="gray")
axes[2].axis("off")

stroke_patch = mpatches.Patch(color='black', label='Foreground (Stroke)')
background_patch = mpatches.Patch(color='white', label='Background (Paper)')
axes[2].legend(
    handles=[stroke_patch, background_patch],
    loc="lower right",
    fontsize=9,
    frameon=True
)

axes[3].set_title("Morphology Cleaned")
axes[3].imshow(cleaned, cmap="gray")
axes[3].axis("off")

axes[4].set_title("Multi-level Otsu\n(Tonal Regions)")
axes[4].imshow(multi_otsu, cmap="gray")
axes[4].axis("off")

dark_patch = mpatches.Patch(color='black', label='Dark Tone')
mid_patch = mpatches.Patch(color='gray', label='Medium Tone')
light_patch = mpatches.Patch(color='white', label='Light Tone')
axes[4].legend(
    handles=[dark_patch, mid_patch, light_patch],
    loc="lower right",
    fontsize=9,
    frameon=True
)

axes[5].set_title("K-Means (K=3)\nOver-segmentation")
axes[5].imshow(kmeans_img, cmap="gray")
axes[5].axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("grayscale.png", gray)
cv2.imwrite("single_otsu.png", otsu)
cv2.imwrite("cleaned_ink.png", cleaned)
cv2.imwrite("multi_otsu.png", multi_otsu)
cv2.imwrite("kmeans_comparison.png", kmeans_img)

print("Processing completed successfully.")
print("Multi-level Otsu thresholds:", thresholds)


print("Processing completed successfully.")
print("Multi-level Otsu thresholds:", thresholds)

