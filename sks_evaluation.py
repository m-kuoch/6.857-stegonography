from evaluation import evaluate_images
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import pywt
from scipy.stats import ortho_group
import skimage


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6))
    test_type = 'dwt_qr'

    save_path = f'./evaluation/sks/{test_type}_results.pkl'

    results = {
        'stego_psnr': [],
        'recovered_psnr': [],
        'stego_ssim': [],
        'recovered_ssim': [],
        'stego_ks': [],
        'recovered_ks': [],
        'stego_p_val': [],
        'recovered_p_val': [],
        'stego_ws': [],
        'recovered_ws': [],
    }

    wavelet = 'db1'

    # Test with a fixed random key
    COVER_DIR_PATH = os.path.join("images", "cover")
    SECRET_IMG_PATH = os.path.join("images", "secret", "test.jpg")
    secret = np.array(Image.open(SECRET_IMG_PATH).convert('L')) / 255
    if test_type == 'qr':
        transformed_secret = secret
    elif test_type == 'dwt_qr':
        transformed_secret, other_secret = pywt.dwt2(secret, wavelet)
    elif test_type == 'dft_qr':
        transformed_secret = np.fft.fft2(secret)

    np.random.seed(0)
    if test_type == 'dwt_qr':
        key = ortho_group.rvs(dim=512)  # dwt halves the image size
    else:
        key = ortho_group.rvs(dim=1024)
    alpha = 0.001

    # Repeat for many cover images
    for i, filename in enumerate(os.listdir(COVER_DIR_PATH)):
        if i % 10 == 0:
            print(f'doing {i}')
        img_num = filename.replace('.jpg', '')
        if filename == "baboon.jpg" or filename == "lady.jpg":
            continue
        #filename = '099902.jpg'

        cover = np.array(Image.open(os.path.join(COVER_DIR_PATH, filename)).convert('L')) / 255
        if test_type == 'qr':
            qc, rc = np.linalg.qr(cover)
        elif test_type == 'dwt_qr':
            LL_cover, other_cover = pywt.dwt2(cover, 'db1')
            qc, rc = np.linalg.qr(LL_cover)
        elif test_type == 'dft_qr':
            qc, rc = np.linalg.qr(np.fft.fft2(cover))

        # Create stego image
        ts = key @ transformed_secret
        r_combined = rc + (alpha * ts)
        stego = qc @ r_combined
        if test_type == 'qr':
            stego_untransformed = stego
        elif test_type == 'dwt_qr':
            stego = pywt.idwt2((stego, other_cover), wavelet)
            stego_untransformed, other_recovered = pywt.dwt2(stego, wavelet)
        elif test_type == 'dft_qr':
            stego = np.fft.ifft2(stego)
            stego_untransformed = np.fft.fft2(stego)

        # Recover secret image
        r1 = np.array(np.asmatrix(qc).H @ stego_untransformed)
        ts_extracted = (r1 - rc) / alpha
        recovered = key.T @ ts_extracted
        #print(transformed_secret - recovered)
        # Undo domain transform if necessary
        if test_type == 'dwt_qr':
            recovered = pywt.idwt2((recovered, other_secret), wavelet)
        elif test_type == 'dft_qr':
            recovered = np.fft.ifft2(recovered)

        #print(np.uint8(secret * 255))
        #print(np.uint8(recovered * 255))

        #secret = np.clip(secret, 0, 1)
        #recovered = np.clip(recovered, 0, 1)

        im_results = evaluate_images(
            np.uint8(cover * 255),
            np.uint8(secret * 255),
            np.uint8(stego * 255),
            np.uint8(recovered * 255),
        )

        # axes[0].imshow(np.uint8(cover*255), cmap='gray', vmin=0, vmax=255)
        # axes[1].imshow(np.uint8(secret*255), cmap='gray', vmin=0, vmax=255)
        # axes[2].imshow(np.uint8(stego*255), cmap='gray', vmin=0, vmax=255)
        # axes[3].imshow(v, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        #
        # exit()

        for field in results:
            results[field].append(im_results[field])
        #print(im_results['recovered_ssim'])

    pickle.dump(results, open(save_path, 'wb'))




