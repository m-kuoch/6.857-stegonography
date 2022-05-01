from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pywt


def load_image(path):
    return np.array(Image.open(path).convert("L"))


def show_image(image):
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.show()


def qr(image):
    qs, rs = np.linalg.qr(image)
    return qs, rs


def dwt(image, wavelet="db1"):
    LL, other_cover = pywt.dwt2(image, wavelet)
    return LL, other_cover

def stego_pipeline(cover_path, secret_path, method: function, row):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    
    cover = load_image(cover_path)
    secret = load_image(secret_path)

    # Preprocess images (scale to 0-1, pad asymmetrically with zeros)
    cover = cover / 255
    secret = secret / 255
    cover = np.concatenate((cover, np.zeros((cover.shape[0], 1))), axis=1)
    cover = np.concatenate((cover, np.zeros((1, cover.shape[1]))), axis=0)
    secret = np.concatenate((secret, np.zeros((secret.shape[0], 1))), axis=1)
    secret = np.concatenate((secret, np.zeros((1, secret.shape[1]))), axis=0)
    
    # Show the cover image
    axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
    axes[row][0].set_title('Cover')
    
    # Show the secret image
    axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
    axes[row][1].set_title('Secret')
    
    stego, recovered = method(cover, secret)

    # Get the stego image
    axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
    axes[row][2].set_title('Stego')
    axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
    axes[row][3].set_title('Recovered')


def qr_hide(cover, secret, alpha):
    qc, rc = np.linalg.qr(cover)
    qs, rs = np.linalg.qr(secret)

    # Combine cover and secret, generate stego
    r_combined = rc + (alpha * rs)
    stego = qc @ r_combined

    # Extract secret image
    qsi, rsi = np.linalg.qr(stego)
    r_extracted = (rsi - rc) / alpha
    recovered = qs @ r_extracted

    return stego, recovered



