from matplotlib import pyplot as plt
import evaluation as eval
from PIL import Image
import numpy as np
import pywt


def _load_image(path):
    return np.array(Image.open(path).convert("L"))


def _show_image(image):
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.show()


def qr_only(cover, secret, alpha=0.01):
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


def dwt_qr(cover, secret, alpha=0.01, wavelet="db1"):
    """QR method with Discrete Wavelet Transform (DWT)"""
    LL, other_cover = pywt.dwt2(cover, wavelet)
    qc, rc = np.linalg.qr(LL)

    LL_secret, other_secret = pywt.dwt2(secret, wavelet)
    qs, rs = np.linalg.qr(LL_secret)

    # Combine cover and secret, generate stego
    r_combined = rc + (alpha * rs)
    stego = qc @ r_combined
    stego = pywt.idwt2((stego, other_cover), wavelet)

    # Extract secret image
    LL, other = pywt.dwt2(stego, wavelet)
    qsi, rsi = np.linalg.qr(LL)
    r_extracted = (rsi - rc) / alpha
    recovered = qs @ r_extracted
    recovered = pywt.idwt2((recovered, other_secret), wavelet)

    return stego, recovered


def fwt_qr(cover, secret, alpha=0.01):
    """QR method with Discrete FOurier Transform (DFT)"""
    LL = np.fft.fft2(cover)
    qc, rc = np.linalg.qr(LL)

    LL_secret = np.fft.fft2(secret)
    qs, rs = np.linalg.qr(LL_secret)

    # Combine cover and secret, generate stego
    r_combined = rc + (alpha * rs)
    stego = qc @ r_combined
    stego = np.fft.ifft2(stego)

    # Extract secret image
    LL = np.fft.fft2(stego)
    qsi, rsi = np.linalg.qr(LL)
    r_extracted = (rsi - rc) / alpha
    recovered = qs @ r_extracted
    recovered = np.fft.ifft2(recovered)

    return stego, recovered


def qr_dwt(cover, secret, alpha=0.01, wavelet="db1"):
    """QR method with DWT after QR decomposition (histo paper)"""
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

    stego = pywt.idwt2(
        (r_combined, other_combined), wavelet
    )  # replace other_combined with other_cover

    # Extract secret image
    rsi, other_recovered = pywt.dwt2(stego, wavelet)
    r_extracted = (rsi - LL_cover) / alpha
    r_extracted = pywt.idwt2((r_extracted, other_secret), wavelet)
    recovered = qs @ r_extracted

    return stego, recovered


def qr_fwt(cover, secret, alpha=0.01):
    """QR method with DWT after QR decomposition (histo paper)"""
    # QR decomposition of secret image
    qs, rs = np.linalg.qr(secret)

    # DWT of cover image and QR decomposition of secret image
    LL_secret = np.fft.fft2(rs)
    LL_cover = np.fft.fft2(cover)

    # Combine cover and secret, generate stego
    r_combined = LL_cover + (alpha * LL_secret)

    stego = np.fft.ifft2(r_combined)

    # Extract secret image
    rsi = np.fft.fft2(stego)
    r_extracted = (rsi - LL_cover) / alpha
    r_extracted = np.fft.ifft2(r_extracted)
    recovered = qs @ r_extracted

    return stego, recovered

def qr_both_dwt(cover, secret, alpha=0.01, wavelet='db1'):
    # QR decomposition of cover image
    qc, rc = np.linalg.qr(cover)
    # QR decomposition of secret image
    qs, rs = np.linalg.qr(secret)

    # DWT of cover image and QR decomposition of secret image
    LL_secret, other_secret = pywt.dwt2(rs, wavelet)
    LL_cover, other_cover = pywt.dwt2(rc, wavelet)

    # Combine cover and secret, generate stego
    r_combined = LL_cover + (alpha * LL_secret)

    other_combined = []
    for i in range(len(other_secret)):
        other_combined.append(other_cover[i] + (alpha * other_secret[i]))
    other_combined = tuple(other_combined)

    stego = pywt.idwt2((r_combined, other_combined), wavelet)  # replace other_combined with other_cover
    stego = qc @ stego  # transform back to original space

    # Extract secret image
    q_secret, r_secret = np.linalg.qr(stego)
    rsi, other_recovered = pywt.dwt2(r_secret, wavelet)
    r_extracted = (rsi - LL_cover) / alpha
    r_extracted = pywt.idwt2((r_extracted, other_secret), wavelet)
    recovered = qs @ r_extracted
    
    return stego, recovered

def qr_both_fwt(cover, secret, alpha=0.01):
    """QR method with FFT after QR decomposition (both)"""
    # QR decomposition of cover image
    qc, rc = np.linalg.qr(cover)
    # QR decomposition of secret image
    qs, rs = np.linalg.qr(secret)

    # DWT of cover image and QR decomposition of secret image
    LL_secret = np.fft.fft2(rs)
    LL_cover = np.fft.fft2(rc)

    # Combine cover and secret, generate stego
    r_combined = LL_cover + (alpha * LL_secret)
    stego = np.fft.ifft2(r_combined)
    stego = qc @ stego  # transform back to original space

    # Extract secret image
    q_secret, r_secret = np.linalg.qr(stego)
    rsi = np.fft.fft2(r_secret)
    r_extracted = (rsi - LL_cover) / alpha
    r_extracted = np.fft.ifft2(r_extracted)
    recovered = qs @ r_extracted
    
    return stego, recovered


METHODS_MAP = {
    "qr": qr_only,
    "dwt_qr": dwt_qr,
    "fwt_qr": fwt_qr,
    "qr_dwt": qr_dwt,
    "qr_fwt": qr_fwt,
    "qr_both_dwt": qr_both_dwt,
    "qr_both_fwt": qr_both_fwt,
}


def run_comparison(
    cover_path: str,
    secret_path: str,
    methods,
    skip_evaluation: bool = False,
):
    evaluations = {}

    # Preprocess images (scale to 0-1, pad asymmetrically with zeros)
    cover = _load_image(cover_path)
    secret = _load_image(secret_path)
    cover = cover / 255
    secret = secret / 255

    # cover = np.concatenate((cover, np.zeros((cover.shape[0], 1))), axis=1)
    # cover = np.concatenate((cover, np.zeros((1, cover.shape[1]))), axis=0)
    # secret = np.concatenate((secret, np.zeros((secret.shape[0], 1))), axis=1)
    # secret = np.concatenate((secret, np.zeros((1, secret.shape[1]))), axis=0)
    


    num_methods = len(methods)
    fig, axes = plt.subplots(nrows=num_methods, ncols=4, figsize=(12, 12))

    for i, method in enumerate(methods):

        # Show the cover image
        cover_display = np.uint8(cover * 255)
        axes[i][0].imshow(cover_display, cmap="gray", vmin=0, vmax=255)

        # Show the secret image
        secret_display = np.uint8(secret * 255)
        axes[i][1].imshow(secret_display, cmap="gray", vmin=0, vmax=255)

        method_name = method
        exe_config = {}
        label = method
        if isinstance(method, dict):
            method_name = method['name']
            exe_config = method.get('exe_config', {})
            label = method.get('label', method_name)
        try:
            method_func = METHODS_MAP[method_name]
        except KeyError:
            print(f'Warning: method "{method_name}" not found! Skipping...')
            continue
        
        stego, recovered = method_func(cover, secret, **exe_config)


        # Show the stego image
        stego_display = np.uint8(stego * 255)
        axes[i][2].imshow(stego_display, cmap="gray", vmin=0, vmax=255)

        # Show the recovered image
        recovered_display = np.uint8(recovered * 255)
        axes[i][3].imshow(recovered_display, cmap="gray", vmin=0, vmax=255)


        if label == method_name:
            axes[i][0].set_ylabel(f"{label} ({exe_config})")
        else:
            axes[i][0].set_ylabel(f"{label}")

        if skip_evaluation:
            continue

        evaluation_dict = eval.evaluate_images(
            cover_display,
            secret_display,
            stego_display,
            recovered_display,
        )
        evaluations[f"{label} ({exe_config})"] = evaluation_dict
        print('Evaluation for "{}":'.format(label))
        print(evaluation_dict)

    for axes in axes.flat:
        axes.set_yticklabels([])
        axes.set_xticklabels([])

    axes[0][0].set_title("Cover")
    axes[0][1].set_title("Secret")
    axes[0][2].set_title("Stego")
    axes[0][3].set_title("Recovered")

    plt.tight_layout()
    plt.show()
    return evaluations