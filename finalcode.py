import cv2
import numpy as np
import matplotlib.pyplot as plt

def cartoonize(image_path, K=5, max_dim=800, attempts=10, random_seed=42):
    # load
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path!r}")
    # resize for speed while preserving aspect ratio
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Working in BGR for OpenCV filters, but converting to RGB for display later
    # 1) Smooth colors while keeping edges 
    # applying bilateral several times to get strong edge-preserving smoothing
    smoothed = img.copy()
    for _ in range(3):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) Color quantization via k-means
    # convert to float32 for kmeans
    Z = smoothed.reshape((-1, 3)).astype(np.float32)

    # kmeans criteria and flags
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS  # better initialization
    # set rng seed for reproducibility
    cv2.setRNGSeed(random_seed)
    compactness, label, center = cv2.kmeans(Z, K, None, criteria, attempts, flags)

    center = np.uint8(center)
    quantized = center[label.flatten()].reshape(img.shape)

    # 3) Edge mask using adaptive thresholding on a median-blurred grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=15,
                                  C=5)
    # edges is binary (255 = background, 0 = lines) suitable as mask to produce black lines

    # 4) Combine: keep quantized color everywhere but zero-out lines using the mask
    cartoon_bgr = cv2.bitwise_and(quantized, quantized, mask=edges)

    # convert to RGB for matplotlib display
    cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return original_rgb, cartoon_rgb

if __name__ == "__main__":
    orig, cartoon = cartoonize("fun.jpg", K=5)
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(orig)
    plt.title("Original (resized if large)")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(cartoon)
    plt.title("Cartoonized")
    plt.axis("off")

    plt.show()
