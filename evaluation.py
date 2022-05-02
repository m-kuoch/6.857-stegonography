from scipy import stats
import skimage.metrics
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
NMRSE
https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
'''

'''
PSNR 
https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio 
'''


'''
SSIM
https://en.wikipedia.org/wiki/Structural_similarity
'''

def evaluate_images(cover_image,secret_image,stego_image,recovered_image):
    '''
    takes in cover, secret, stego, recovered image, and evaluate them

    :param cover_image: The cover image
    :param secret_image: The secret image to hide
    :param stego_image: The output steganography
    :param recovered_image: The recovered image
    :return: A dictionary with various types of scores and their values.
    '''

    # print(cover_image)
    # print(secret_image)
    # print(stego_image)
    # print(recovered_image)
    hist_cover = cv2.calcHist([cover_image], [0], None, [256], [0, 256])
    hist_secret = cv2.calcHist([secret_image], [0], None, [256], [0, 256])
    hist_stego = cv2.calcHist([stego_image], [0], None, [256], [0, 256])
    hist_recovered = cv2.calcHist([recovered_image], [0], None, [256], [0, 256])

    # score_mse_stego = skimage.metrics.mean_squared_error(cover_image, stego_image)
    # score_mse_recovered = skimage.metrics.mean_squared_error(secret_image,recovered_image)
    # # NMRSE https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    # score_nmrse_stego = skimage.metrics.normalized_root_mse(cover_image, stego_image)
    # score_nmrse_recovered = skimage.metrics.normalized_root_mse(secret_image,recovered_image)

    # PSNR https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    score_psnr_stego = skimage.metrics.peak_signal_noise_ratio(cover_image, stego_image)
    score_psnr_recovered = skimage.metrics.peak_signal_noise_ratio(secret_image,recovered_image)

    # SSIM https://en.wikipedia.org/wiki/Structural_similarity

    score_ssim_stego = skimage.metrics.structural_similarity(cover_image, stego_image)
    score_ssim_recovered = skimage.metrics.structural_similarity(secret_image,recovered_image)

    # Kolmogorov-Smirnov test https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    # print(hist_cover.reshape(-1))
    stats_ks_stego, p_val_ks_stego = stats.ks_2samp(hist_cover.reshape(-1), hist_stego.reshape(-1))
    stats_ks_recovered, p_val_ks_recovered = stats.ks_2samp(hist_secret.reshape(-1),hist_recovered.reshape(-1))

    # return {'stego_mse':score_mse_stego, 'recovered_mse': score_mse_recovered,
    #         'stego_nmrse':score_nmrse_stego, 'recovered_nmrse': score_nmrse_recovered,
    #         'stego_psnr':score_psnr_stego, 'recovered_psnr': score_psnr_recovered,
    #         'stego_ssim':score_ssim_stego, 'recovered_ssim': score_ssim_recovered}
    return {
            'stego_psnr': score_psnr_stego, 'recovered_psnr': score_psnr_recovered,
            'stego_ssim': score_ssim_stego, 'recovered_ssim': score_ssim_recovered,
            'stego_ks': stats_ks_stego, 'recovered_ks': stats_ks_recovered,
            'stego_p_val': p_val_ks_stego, 'recovered_p_val': p_val_ks_recovered
    }


