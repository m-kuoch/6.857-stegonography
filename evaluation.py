from skimage.measure import compare_mse,compare_nrmse,compare_psnr,compare_ssim
import argparse
import imutils
import cv2




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


if __name__ == '__main__':
    # 3. Load the two input images
    coverImagePath = './images/cover/lady.jpg'
    stegoImagePath = './images/secret/secret_grayscale.jpg'
    outputImagePath = './images/stego/breuh.jpg'
    recoveredImagePath = ''
    coverImage = cv2.imread(coverImagePath)
    stegoImage = cv2.imread(stegoImagePath)
    # recoveredImage = cv2.imread(recoveredImagePath)


    # MSE https://en.wikipedia.org/wiki/Mean_squared_error
    score_mse_cover_stego = compare_mse(coverImage, outputImagePath)
    # score_mse_stego_recovered = skimage.measure.compare_mse(recoveredImage, stegoImagePath)
    # NMRSE https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    score_nmrse_cover_stego = compare_nrmse(coverImage, outputImagePath)
    # score_nmrse_stego_recovered = skimage.measure.compare_nrmse(recoveredImage, stegoImagePath)

    # PSNR https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    score_psnr_cover_stego = compare_psnr(coverImage, outputImagePath)
    # score_psnr_stego_recovered = skimage.measure.compare_psnr(recoveredImage, stegoImagePath)

    # SSIM https://en.wikipedia.org/wiki/Structural_similarity

    (score_ssim_cover_stego, diff) = compare_ssim(coverImage, outputImagePath, full=True)
    # (score_ssim_stego_recovered, diff_recovered) = skimage.measure.compare_ssim(recoveredImage, stegoImagePath, full=True)
    # score_psnr_cover_stego = (diff * 255).astype("uint8")



    print("Cover vs Stego \n MSE: {} \n NMRSE: {} \n PNSR:{} \n SSIM: {}  ".format(score_mse_cover_stego, score_nmrse_cover_stego, score_psnr_cover_stego, score_ssim_cover_stego))
    # print("Recovered vs Stego \n MSE: {} \n NMRSE: {} \n PNSR:{} \n SSIM: {}  ".format(score_mse_cover_stego, score_nmrse_cover_stego, score_psnr_cover_stego, score_ssim_cover_stego))
