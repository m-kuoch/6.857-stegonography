from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pywt

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
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

        axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')
        qc, rc = np.linalg.qr(cover)

        axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        qs, rs = np.linalg.qr(secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        qsi, rsi = np.linalg.qr(stego)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

    qr_hide(cover, secret, axes, row=0)

    def qr_hide_dwt(cover, secret, axes, row=0):
        """QR method with DWT"""
        alpha = 0.01

        axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')


        LL, other_cover = pywt.dwt2(cover, 'db1')
        qc, rc = np.linalg.qr(LL)
        #secret = secret[:LL.shape[0], :LL.shape[1]]

        axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        LL_secret, other_secret = pywt.dwt2(secret, 'db1')
        qs, rs = np.linalg.qr(LL_secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        stego = pywt.idwt2((stego, other_cover), 'db1')
        axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        LL, other = pywt.dwt2(stego, 'db1')
        qsi, rsi = np.linalg.qr(LL)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        recovered = pywt.idwt2((recovered, other_secret), 'db1')
        axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')
    qr_hide_dwt(cover, secret, axes, row=1)

    def qr_hide_dwt2(cover, secret, axes, row=0):
        """QR method with DWT after QR decomposition (histo paper)"""
        alpha = 0.01

        axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')
        axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')

        # QR decomposition of secreet image
        qs, rs = np.linalg.qr(secret)

        # DWT of cover image and QR decomposition of secret image
        LL_secret, other_secret = pywt.dwt2(secret, 'db1')
        LL_cover, other_cover = pywt.dwt2(cover, 'db1')

        # Combine cover and secret, generate stego
        r_combined = LL_cover + (alpha * LL_secret)
        stego = pywt.idwt2((r_combined, other_cover), 'db1')
        axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        rsi, other_recovered = pywt.dwt2(stego, 'db1')
        r_extracted = (rsi - LL_cover) / alpha
        r_extracted = pywt.idwt2((r_extracted, other_secret), 'db1')
        recovered = qs @ r_extracted
        axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')
    qr_hide_dwt2(cover, secret, axes, row=2)

    plt.show()



