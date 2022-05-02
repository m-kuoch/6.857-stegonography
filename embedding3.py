from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pywt

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
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


    def qr_hide_dwt2(cover, secret, axes, row=0):
        """QR method with DWT after QR decomposition (histo paper)"""
        alpha = 0.01
        wavelet = 'db1'

        axes[row][0].imshow(cover * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')
        axes[row][1].imshow(secret * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')

        # QR decomposition of secret image
        qs, rs = np.linalg.qr(secret)

        # DWT of cover image and QR decomposition of secret image
        LL_secret, other_secret = pywt.dwt2(rs, wavelet)
        LL_cover, other_cover = pywt.dwt2(cover, wavelet)

        # Combine cover and secret, generate stego
        r_combined = LL_cover + (alpha * LL_secret)

        other_combined = []
        for i in range(len(other_secret)):
            other_combined.append(other_cover[i] + (alpha * other_secret[i]))
        other_combined = tuple(other_combined)

        stego = pywt.idwt2((r_combined, other_combined), wavelet)  # replace other_combined with other_cover
        axes[row][2].imshow(np.uint8(stego * 255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        rsi, other_recovered = pywt.dwt2(stego, wavelet)
        r_extracted = (rsi - LL_cover) / alpha
        r_extracted = pywt.idwt2((r_extracted, other_secret), wavelet)
        recovered = qs @ r_extracted
        axes[row][3].imshow(np.uint8(recovered * 255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

    qr_hide_dwt2(cover, secret, axes, row=0)
    axes[0][0].set_ylabel('Histo paper, dwt')


    def qr_hide_fft_v2(cover, secret, axes, row=0):
        """QR method with FFT after QR decomposition (similar to histo paper)"""
        alpha = 0.01

        axes[row][0].imshow(cover * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')
        axes[row][1].imshow(secret * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')

        # QR decomposition of secret image
        qs, rs = np.linalg.qr(secret)

        # DWT of cover image and QR decomposition of secret image
        LL_secret = np.fft.fft2(rs)
        LL_cover = np.fft.fft2(cover)

        # Combine cover and secret, generate stego
        r_combined = LL_cover + (alpha * LL_secret)
        stego = np.fft.ifft2(r_combined)
        axes[row][2].imshow(np.uint8(stego * 255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        rsi = np.fft.fft2(stego)
        r_extracted = (rsi - LL_cover) / alpha
        r_extracted = np.fft.ifft2(r_extracted)
        recovered = qs @ r_extracted
        axes[row][3].imshow(np.uint8(recovered * 255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

    qr_hide_dwt2(cover, secret, axes, row=1)
    axes[1][0].set_ylabel('Histo paper, fft')

    plt.show()