from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pywt
import evaluation

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

    def qr_hide_dwt(cover, secret, axes, row=0):
        """QR method with DWT"""
        alpha = 0.01
        wavelet = 'db1'

        axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')

        LL, other_cover = pywt.dwt2(cover, wavelet)
        qc, rc = np.linalg.qr(LL)
        #secret = secret[:LL.shape[0], :LL.shape[1]]

        axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        LL_secret, other_secret = pywt.dwt2(secret, wavelet)
        qs, rs = np.linalg.qr(LL_secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        stego = pywt.idwt2((stego, other_cover), wavelet)
        axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        LL, other = pywt.dwt2(stego, wavelet)
        qsi, rsi = np.linalg.qr(LL)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        recovered = pywt.idwt2((recovered, other_secret), wavelet)
        axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

        evaluation_dict = evaluation.evaluate_images(np.uint8(cover*255), np.uint8(secret*255),np.uint8(stego*255),np.uint8(recovered*255))
        print(evaluation_dict)
    qr_hide_dwt(cover, secret, axes, row=0)
    axes[0][0].set_ylabel('QR with DWT (db1)')

    def qr_hide_dft(cover, secret, axes, row=0):
        """QR method with DFT"""
        alpha = 0.01

        axes[row][0].imshow(cover*255, cmap='gray', vmin=0, vmax=255)
        axes[row][0].set_title('Cover')


        LL = np.fft.fft2(cover)
        qc, rc = np.linalg.qr(LL)

        axes[row][1].imshow(secret*255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title('Secret')
        LL_secret = np.fft.fft2(secret)
        qs, rs = np.linalg.qr(LL_secret)

        # Combine cover and secret, generate stego
        r_combined = rc + (alpha * rs)
        stego = qc @ r_combined
        stego = np.fft.ifft2(stego)
        axes[row][2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title('Stego')

        # Extract secret image
        LL = np.fft.fft2(stego)
        qsi, rsi = np.linalg.qr(LL)
        r_extracted = (rsi - rc) / alpha
        recovered = qs @ r_extracted
        recovered = np.fft.ifft2(recovered)
        axes[row][3].imshow(np.uint8(recovered*255), cmap='gray', vmin=0, vmax=255)
        axes[row][3].set_title('Recovered')

        evaluation_dict = evaluation.evaluate_images(np.uint8(cover*255), np.uint8(secret*255),np.uint8(stego*255),np.uint8(recovered*255))
        print(evaluation_dict)
    qr_hide_dft(cover, secret, axes, row=1)
    axes[1][0].set_ylabel('QR with DFT')

    for axes in axes.flat:
        axes.set_yticklabels([])
        axes.set_xticklabels([])

    plt.tight_layout()
    plt.show()