import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MOTION BLUR
# ---------------------------------------------------------------------------

def apply_motion_blur(image, kernel_size):
    """
    Applies a single pass of linear motion blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    kernel_size : int
        Size of the motion blur kernel (odd number recommended). Larger
        values produce a stronger blur.

    Returns
    -------
    np.ndarray
        The motion-blurred image.
    """
    kernel_size = max(1, int(kernel_size))

    # Horizontal motion blur kernel: a single row of 1/kernel_size values
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel_size

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def generate_motion_blur_levels(image):
    """
    Generates 3 versions of the image with increasing motion blur strength.

    Returns
    -------
    list of (label, np.ndarray) tuples, ordered mild -> severe.
    """
    kernel_sizes = [7, 15, 25]
    labels = ["Motion Blur - Mild", "Motion Blur - Medium", "Motion Blur - Strong"]

    results = []
    for label, k in zip(labels, kernel_sizes):
        results.append((label, apply_motion_blur(image, k)))

    return results


# ---------------------------------------------------------------------------
# LOW LIGHT
# ---------------------------------------------------------------------------

def apply_low_light(image, brightness_factor, gamma=1.5):
    """
    Applies a single pass of low-light simulation to an image by scaling
    down brightness (V channel in HSV) and applying a gamma curve to
    darken midtones, mimicking underexposed / night footage.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    brightness_factor : float
        Value in (0, 1]. Lower = darker. 1.0 = no change.
    gamma : float
        Gamma > 1 darkens midtones further for a more natural low-light look.

    Returns
    -------
    np.ndarray
        The darkened image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Scale down the Value (brightness) channel
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Apply a gamma curve for a more realistic non-linear darkening
    normalized = darkened.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, gamma) * 255.0

    return gamma_corrected.astype(np.uint8)


def generate_low_light_levels(image):
    """
    Generates 3 versions of the image with increasing low-light severity.

    Returns
    -------
    list of (label, np.ndarray) tuples, ordered mild -> severe.
    """
    brightness_factors = [0.7, 0.45, 0.25]
    labels = ["Low Light - Mild", "Low Light - Medium", "Low Light - Strong"]

    results = []
    for label, factor in zip(labels, brightness_factors):
        results.append((label, apply_low_light(image, factor)))

    return results


# ---------------------------------------------------------------------------
# RAIN
# ---------------------------------------------------------------------------

def apply_rain(image, density=0.02, length=20, thickness=1, angle=-70, blur_ksize=5, seed=None):
    """
    Applies a single pass of synthetic rain to an image by generating
    random streak noise and blending it onto the image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    density : float
        Fraction of pixels used as rain-streak seeds (higher = more rain).
    length : int
        Length of each rain streak in pixels.
    thickness : int
        Thickness of each rain streak.
    angle : float
        Angle of the rain streaks in degrees (0 = vertical, negative = leaning left).
    blur_ksize : int
        Kernel size used to slightly blur the streak layer for realism.
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    np.ndarray
        The image with rain overlaid.
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = image.shape[:2]

    # Create a blank noise layer and seed it with random bright points.
    # `density` is the fraction of total pixels used as rain-streak seeds.
    rain_layer = np.zeros((height, width), dtype=np.uint8)
    num_drops = int(density * height * width)

    xs = np.random.randint(0, width, num_drops)
    ys = np.random.randint(0, height, num_drops)
    rain_layer[ys, xs] = 255

    # Stretch the points into streaks using a motion-blur-style kernel
    # oriented along the rain angle
    kernel_size = max(length, 3)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = np.deg2rad(angle)
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.sin(angle_rad))
        y = int(center + offset * np.cos(angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()

    streaks = cv2.filter2D(rain_layer, -1, kernel)

    if thickness > 1:
        dilate_kernel = np.ones((thickness, thickness), np.uint8)
        streaks = cv2.dilate(streaks, dilate_kernel)

    streaks = cv2.GaussianBlur(streaks, (blur_ksize, blur_ksize), 0)

    # Blend the rain streaks (as white/light gray) onto the original image
    streaks_bgr = cv2.cvtColor(streaks, cv2.COLOR_GRAY2BGR)
    rainy_image = cv2.addWeighted(image, 1.0, streaks_bgr, 0.85, 0)

    # Slightly desaturate and darken to mimic overcast/rainy atmosphere
    hsv = cv2.cvtColor(rainy_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8   # reduce saturation
    hsv[:, :, 2] = hsv[:, :, 2] * 0.9   # slightly dim
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rainy_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rainy_image


def generate_rain_levels(image):
    """
    Generates 3 versions of the image with increasing rain severity.

    Returns
    -------
    list of (label, np.ndarray) tuples, ordered mild -> severe.
    """
    settings = [
        {"density": 0.008, "length": 15, "thickness": 1},  # mild
        {"density": 0.020, "length": 22, "thickness": 2},  # medium
        {"density": 0.040, "length": 30, "thickness": 3},  # strong
    ]
    labels = ["Rain - Mild", "Rain - Medium", "Rain - Strong"]

    results = []
    for label, params in zip(labels, settings):
        results.append((label, apply_rain(image, seed=42, **params)))

    return results


# ---------------------------------------------------------------------------
# MAIN - 3x3 GRID DISPLAY
# ---------------------------------------------------------------------------

def display_augmentation_grid(image):
    """
    Builds a 3x3 grid: rows are augmentation types (Motion Blur, Low Light,
    Rain), columns are severity levels (mild, medium, strong).
    """
    motion_blur_results = generate_motion_blur_levels(image)
    low_light_results = generate_low_light_levels(image)
    rain_results = generate_rain_levels(image)

    rows = [motion_blur_results, low_light_results, rain_results]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for row_idx, row_results in enumerate(rows):
        for col_idx, (label, aug_image) in enumerate(row_results):
            ax = axes[row_idx, col_idx]
            ax.imshow(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
            ax.set_title(label)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    image_path = r"C:\Users\alonk\GitHub\Project_digital_image_processing__Amit_Alon\data\1_data_lane_detection_low_level\0a0a0b1a-7c39d841.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: image not found -> {image_path}")
        return

    display_augmentation_grid(image)


if __name__ == "__main__":
    main()