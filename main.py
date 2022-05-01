import pipeline as p
import stega
import os


def main() -> None:
    # stega.stegano_hide("./lady.png", "hello world", "breuh.png")
    # print(stega.reveal("breuh.png"))
    evaluation = p.run_comparison(
        cover_path=os.path.join("images", "cover", "lady.jpg"),
        secret_path=os.path.join("images", "secret", "secret_grayscale.jpg"),
        methods=["qr", "dwt_qr", "fwt_qr", "qr_dwt","qr_fwt"],
        exe_config={"qr_dwt": {"wavelet": "db1"}},
        display_config={
            "qr": {"label": "QR Method Only"},
            "dwt_qr": {"label": "QR Method with DWT"},
            "fwt_qr": {"label": "QR with FFT2"},
        },
    )
    print(evaluation)


if __name__ == "__main__":
    main()
