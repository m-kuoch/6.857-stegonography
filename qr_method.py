from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

    """QR method with no DWT"""

    alpha = 0.01

    im = Image.open('lady.jpg').convert('L')

    cover = np.array(im)  # 2D array
    axes[0][0].imshow(cover, cmap='gray', vmin=0, vmax=255)
    axes[0][0].set_title('Cover')
    qc, rc = np.linalg.qr(cover)

    secret = np.array(Image.open('secret_grayscale.jpg').convert('L'))
    axes[0][1].imshow(secret, cmap='gray', vmin=0, vmax=255)
    axes[0][1].set_title('Secret')
    qs, rs = np.linalg.qr(secret)

    # Combine cover and secret, generate stego
    r_combined = rc + (alpha * rs)
    stego = qc @ r_combined
    print(stego)
    axes[0][2].imshow(np.uint8(stego), cmap='gray', vmin=0, vmax=255)
    axes[0][2].set_title('Stego')

    # Extract secret image
    qsi, rsi = np.linalg.qr(stego)
    r_extracted = (rsi - rc) / alpha
    recovered = qs @ r_extracted
    axes[0][3].imshow(np.uint8(recovered), cmap='gray', vmin=0, vmax=255)
    axes[0][3].set_title('Recovered')

    """QR method with DWT"""


    plt.show()



