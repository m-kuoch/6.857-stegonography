
import numpy as np




def lsb(cover, secret):
    data_c = np.array(cover)

    # Convert image to 1-bit pixel, black and white and resize to cover image

    data_s = np.array(secret, dtype=np.uint8)

    # Rewrite LSB
    stego = data_c & ~1

    # new_img.save("cover-secret.png")
    # new_img.show()

    # Recover Secret
    recovered = np.array(stego) & 1

    return stego, recovered