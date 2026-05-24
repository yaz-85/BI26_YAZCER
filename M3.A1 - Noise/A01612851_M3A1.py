import numpy as np
import matplotlib.pyplot as plt
import pydicom


np.random.seed(42)
dicom_path = "ID_0091_AGE_0072_CONTRAST_0_CT.dcm"
ds = pydicom.dcmread(dicom_path)
image = ds.pixel_array.astype(np.float32)

image = (image - image.min()) / (image.max() - image.min())

def salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(img)
    rows, cols = img.shape

    num_salt = int(rows * cols * salt_prob)
    salt_coords = (
        np.random.randint(0, rows, num_salt),
        np.random.randint(0, cols, num_salt)
    )
    noisy[salt_coords] = 1

    num_pepper = int(rows * cols * pepper_prob)
    pepper_coords = (
        np.random.randint(0, rows, num_pepper),
        np.random.randint(0, cols, num_pepper)
    )
    noisy[pepper_coords] = 0
    return noisy

def gaussian_noise(img, mean=0, sigma=0.05):
    gaussian = np.random.normal(mean, sigma, img.shape)
    noisy = img + gaussian
    noisy = np.clip(noisy, 0, 1)
    return noisy

def speckle_noise(img):
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise * 0.2
    noisy = np.clip(noisy, 0, 1)
    return noisy

def poisson_noise(img):
    vals = 256
    noisy = np.random.poisson(img * vals) / float(vals)
    noisy = np.clip(noisy, 0, 1)
    return noisy

def periodic_noise(img, amplitude=0.15, frequency=15):
    rows, cols = img.shape
    x = np.arange(cols)
    sinusoidal_pattern = amplitude * np.sin(2 * np.pi * x / frequency)
    noise_matrix = np.tile(sinusoidal_pattern, (rows, 1))
    noisy = img + noise_matrix
    noisy = np.clip(noisy, 0, 1)
    return noisy

sp_image = salt_pepper_noise(image)
gauss_image = gaussian_noise(image)
speckle_image = speckle_noise(image)
poisson_image = poisson_noise(image)
periodic_image = periodic_noise(image)

titles = [
    "Original",
    "Salt & Pepper",
    "Gaussian",
    "Speckle",
    "Poisson",
    "Periodic Noise"
]

images = [
    image,
    sp_image,
    gauss_image,
    speckle_image,
    poisson_image,
    periodic_image
]

plt.figure(figsize=(15, 10))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()