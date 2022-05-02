import numpy as np

import pipeline as p
import stega
import os
import pandas as pd
import dataframe_image as dfi


def main() -> None:
    # stega.stegano_hide("./lady.png", "hello world", "breuh.png")
    # print(stega.reveal("breuh.png"))
    # evaluation = p.run_comparison(
    #     cover_path=os.path.join("images", "cover", "lady.jpg"),
    #     secret_path=os.path.join("images", "secret", "secret_grayscale.jpg"),
    #     methods=[
    #         {
    #             "name": "qr",
    #             "label": "QR Method only (alpha 0.05)",
    #             "exe_config": {"alpha": 0.05},
    #         },
    #         {
    #             "name": "qr",
    #             "label": "QR Method only (default alpha)",
    #         },
    #     ],
    # )
    evaluation = p.run_comparison(
        cover_path=os.path.join("images", "cover", "099900.jpg"),
        secret_path=os.path.join("images", "cover", "099901.jpg"),
        methods=[
            "qr",
            "dwt_qr",
            "fwt_qr",
            "qr_dwt",
            "qr_fwt",
            "qr_both_dwt",
            "qr_both_fwt",
        ],
        skip_evaluation=True,
    )
    print(evaluation)
    df = pd.DataFrame(evaluation)
    # dfi.save_dataframe_image(df, "evaluation2.png")
    df = pd.DataFrame.from_dict(evaluation, orient="index")
    # df.style.apply(highlight_max).to_html("results.html")
    df_styled = df.style.highlight_max(color="lightgreen", axis=0).highlight_min(
        color="lightcoral", axis=0
    )
    dfi.export(df_styled, "dataframe2.png")
    print(df)
    print(evaluation)


if __name__ == "__main__":
    main()
