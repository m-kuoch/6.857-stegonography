from stegano import lsb

# source code: https://git.sr.ht/~cedric/stegano/tree/master/item/stegano/lsb/lsb.py

def stegano_hide(cover,message,output_file):
    secret = lsb.hide(cover,message)
    secret.save(output_file)

def reveal(stego):
    return lsb.reveal(stego)


