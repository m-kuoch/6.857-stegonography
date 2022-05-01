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


METHODS_MAP = {
    "qr": qr_only,
    "dwt_qr": dwt_qr,
    "fwt_qr": fwt_qr,
    "qr_dwt": qr_dwt,
    "qr_fwt": qr_fwt,
}


def run_comparison(
    cover_path: str,
    secret_path: str,
    methods,
    exe_config: dict = {},
    display_config: dict = {},
):
    evaluations = {}

    # Preprocess images (scale to 0-1, pad asymmetrically with zeros)
    cover = _load_image(cover_path)
    secret = _load_image(secret_path)
    cover = cover / 255
    secret = secret / 255
    cover = np.concatenate((cover, np.zeros((cover.shape[0], 1))), axis=1)
    cover = np.concatenate((cover, np.zeros((1, cover.shape[1]))), axis=0)
    secret = np.concatenate((secret, np.zeros((secret.shape[0], 1))), axis=1)
    secret = np.concatenate((secret, np.zeros((1, secret.shape[1]))), axis=0)

    num_methods = len(methods)
    fig, axes = plt.subplots(nrows=num_methods, ncols=4, figsize=(12, 12))

    for i, method in enumerate(methods):
        method_func = METHODS_MAP[method]
        execution_config = exe_config.get(method, {})
        stego, recovered = method_func(cover, secret, **execution_config)

        cover_display = np.uint8(cover * 255)
        secret_display = np.uint8(secret * 255)
        stego_display = np.uint8(stego * 255)
        recovered_display = np.uint8(recovered * 255)

        # Show the cover image
        axes[i][0].imshow(cover_display, cmap="gray", vmin=0, vmax=255)
        axes[i][0].set_title("Cover")

        # Show the secret image
        axes[i][1].imshow(secret_display, cmap="gray", vmin=0, vmax=255)
        axes[i][1].set_title("Secret")

        # Show the stego image
        axes[i][2].imshow(stego_display, cmap="gray", vmin=0, vmax=255)
        axes[i][2].set_title("Stego")

        # Show the recovered image
        axes[i][3].imshow(recovered_display, cmap="gray", vmin=0, vmax=255)
        axes[i][3].set_title("Recovered")

        try:
            label = display_config[method]["label"]
        except KeyError:
            label = method
        axes[i][0].set_ylabel(label)

        evaluation_dict = eval.evaluate_images(
            cover_display,
            secret_display,
            stego_display,
            recovered_display,
        )
        evaluations[method] = evaluation_dict
        print('Evaluation for "{}":'.format(method))
        print(evaluation_dict)

    plt.show()
    return evaluations
