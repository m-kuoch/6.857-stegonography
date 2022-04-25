from stegano import lsb


def stegano_hide(cover,message,output_file):
    secret = lsb.hide(cover,message)
    secret.save(output_file)

def reveal(stego):
    return lsb.reveal(stego)


