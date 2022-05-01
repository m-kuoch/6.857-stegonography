import stega


def main() -> None:
    stega.stegano_hide("./lady.png", "hello world", "breuh.png")
    print(stega.reveal("breuh.png"))
    pass


if __name__ == "__main__":
    main()
