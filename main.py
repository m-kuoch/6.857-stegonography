import pipeline as p
import stega
import os


def main() -> None:
    # stega.stegano_hide("./lady.png", "hello world", "breuh.png")
    # print(stega.reveal("breuh.png"))
    evaluation = p.run_comparison(
        cover_path=os.path.join("images", "cover", "lady.jpg"),
        secret_path=os.path.join("images", "secret", "secret_grayscale.jpg"),
        methods=[
            {
                "name": "qr",
                "label": "QR Method only (alpha 0.05)",
                "exe_config": {"alpha": 0.05},
            },
            {
                "name": "qr",
                "label": "QR Method only (default alpha)",
            },
        ],
    )
    print(evaluation)


if __name__ == "__main__":
    main()
