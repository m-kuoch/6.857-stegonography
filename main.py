import dataframe_image as dfi
import pipeline as p
import pandas as pd
import numpy as np
import os

# import stega
def analysis():
    df = pd.read_csv(os.path.join("evaluation", "eval-dataframe.csv"))
    grouped_by_method = df.groupby("method").mean()
    print(grouped_by_method)
    grouped_by_method.to_csv(os.path.join("evaluation", "eval-dataframe-mean-grouped.csv"))


def main() -> None:
    COVER_DIR_PATH = os.path.join("images", "cover")
    SECRET_IMG_PATH = os.path.join("images", "secret", "test.jpg")
    running_df = pd.DataFrame()
    for filename in os.listdir(COVER_DIR_PATH):
        img_num = filename.replace('.jpg', '')
        if filename == "baboon.jpg" or filename == "lady.jpg":
            continue

        print(
            f"Running evaluation on {filename} with secret image {SECRET_IMG_PATH}..."
        )
        evaluation = p.run_comparison(
            os.path.join(COVER_DIR_PATH, filename),
            SECRET_IMG_PATH,
            methods=[
                "qr",
                "dwt_qr",
                "dft_qr",
                "qr_dwt",
                "qr_dft",
                "qr_both_dwt",
                "qr_both_dft",
            ],
            save_fig=True,
        )

        df = pd.DataFrame.from_dict(evaluation, orient="index")
        running_df = pd.concat([running_df, df])
    
    running_df['method'] = running_df.index
    running_df.to_csv(os.path.join("evaluation", "eval-dataframe.csv"), index=False)
    # evaluation = p.run_comparison(
    #     cover_path=os.path.join("images", "cover", "099902.jpg"),
    #     secret_path=os.path.join("images", "secret", "test.jpg"),
    #     methods=[
    #         "qr",
    #         "dwt_qr",
    #         "dft_qr",
    #         "qr_dwt",
    #         "qr_dft",
    #         "qr_both_dwt",
    #         "qr_both_dft",
    #     ],
    #     skip_evaluation=False,
    # )
    # print(evaluation)
    # df = pd.DataFrame(evaluation)
    # # dfi.save_dataframe_image(df, "evaluation2.png")
    # df = pd.DataFrame.from_dict(evaluation, orient="index")
    # # df.style.apply(highlight_max).to_html("results.html")
    # df_styled = df.style.highlight_max(color="lightgreen", axis=0).highlight_min(
    #     color="lightcoral", axis=0
    # )
    # dfi.export(df_styled, "dataframe2.png")
    # print(df)
    # print(evaluation)


if __name__ == "__main__":
    # main()
    analysis()
