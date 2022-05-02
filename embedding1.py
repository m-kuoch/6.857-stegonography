from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import evaluation

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    im = Image.open('lady.jpg').convert('L')
    cover = np.array(im)  # 2D array
    secret = np.array(Image.open('secret_grayscale.jpg').convert('L'))

    cover = cover / 255
    secret = secret / 255

    # Pad asymmetrically with zeros
    cover = np.concatenate((cover, np.zeros((cover.shape[0], 1))), axis=1)
    cover = np.concatenate((cover, np.zeros((1, cover.shape[1]))), axis=0)
    secret = np.concatenate((secret, np.zeros((secret.shape[0], 1))), axis=1)
    secret = np.concatenate((secret, np.zeros((1, secret.shape[1]))), axis=0)

    print(cover.shape, secret.shape)

    def qr_hide(cover, secret, axes, row=0):
        """QR method with no DWT"""
        alpha = 0.01

        axes[0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Cover')
        qc, rc = np.linalg.qr(cover)

        axes[1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Secret')
        qs, rs = np.linalg.qr(secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        axes[2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Stego')

        # Extract secret image
        qsi, rsi = np.linalg.qr(stego)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        axes[3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[3].set_title('Recovered')

        evaluation_dict = evaluation.evaluate_images(np.uint8(cover*255), np.uint8(secret*255),np.uint8(stego*255),np.uint8(recovered*255))
        print(evaluation_dict)

    qr_hide(cover, secret, axes, row=0)
    axes[0].set_ylabel('QR method')

    for axes in axes.flat:
        axes.set_yticklabels([])
        axes.set_xticklabels([])

    plt.tight_layout()

    plt.show()