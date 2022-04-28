from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pywt

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

    """QR method with no DWT"""

    im = Image.open('lady.jpg').convert('L')
    cover = np.array(im)  # 2D array
    secret = np.array(Image.open('secret_grayscale.jpg').convert('L'))
    def qr_hide(cover, secret, axes, row=0):
        alpha = 0.01

        axes[row][0].imshow(cover, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')
        qc, rc = np.linalg.qr(cover)

        axes[row][1].imshow(secret, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        qs, rs = np.linalg.qr(secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        print(stego)
        axes[row][2].imshow(np.uint8(stego), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        qsi, rsi = np.linalg.qr(stego)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        axes[row][3].imshow(np.uint8(recovered), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

    qr_hide(cover, secret, axes, row=0)

    """QR method with DWT"""
    def qr_hide_dwt(cover, secret, axes, row=0):
        alpha = 0.1

        axes[row][0].imshow(cover, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')


        LL, other_cover = pywt.dwt2(cover, 'coif1')
        qc, rc = np.linalg.qr(LL)
        #secret = secret[:LL.shape[0], :LL.shape[1]]

        axes[row][1].imshow(secret, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        LL_secret, other_secret = pywt.dwt2(secret, 'coif1')
        qs, rs = np.linalg.qr(LL_secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        stego = pywt.idwt2((stego, other_cover), 'coif1')
        axes[row][2].imshow(np.uint8(stego), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        LL, other = pywt.dwt2(stego, 'coif1')
        qsi, rsi = np.linalg.qr(LL)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        recovered = pywt.idwt2((recovered, other_secret), 'coif1')
        axes[row][3].imshow(np.uint8(recovered), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')
    qr_hide_dwt(cover, secret, axes, row=1)

    plt.show()



