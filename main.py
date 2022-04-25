import stega

if __name__ == '__main__':
    stega.stegano_hide('./lady.png',"hello world", "breuh.png")
    print(stega.reveal("breuh.png"))