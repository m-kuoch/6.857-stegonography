import dataframe_image as dfi
import pipeline as p
import pandas as pd
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

METRIC_MAP = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "ks": "KS",
    "p_val": "P-Value",
    "ws": "Wasserstein Metric",
    "stego_psnr": "PSNR (Stego)",
    "stego_ssim": "SSIM (Stego)",
    "stego_ws": "Wasserstein (Stego)",
    "recovered_psnr": "PSNR (Recovered)",
    "recovered_ssim": "SSIM (Recovered)",
    "recovered_ws": "Wasserstein (Recovered)",
}

METHOD_LABEL_MAP = {
    "qr ({})": "QR Only",
    "dwt_qr ({})": "DWT -> QR",
    "dft_qr ({})": "FFT -> QR",
    "qr_dwt ({})": "QR -> DWT",
    "qr_dft ({})": "QR -> FFT",
    "qr_both_dwt ({})": "QR (both) -> DWT",
    "qr_both_dft ({})": "QR (both) -> FFT",
}

# import stega
def analysis():
    df = pd.read_csv(os.path.join("evaluation", "eval-dataframe.csv"))
    # Melt the df to get access to the different types
    df = df.melt(id_vars=["method"], value_vars=df.columns[:-1], var_name="eval_metric")
    print(METHOD_LABEL_MAP)
    df["method"] = df["method"].replace(
        METHOD_LABEL_MAP.keys(), METHOD_LABEL_MAP.values()
    )
    # df['eval_metric'] = df['eval_metric'].replace(METRIC_MAP.keys(), METRIC_MAP.values())
    # Aggregate by getting the median value for each method and metric
    aggregated = df.groupby(["method", "eval_metric"]).median().reset_index()
    print(aggregated)

    # px.sunburst(df, path=["eval_metric", "method"], values='value').show()
    # for eval_metric in ["psnr", "ssim", "ws"]:
    #     specific_df = df[df["eval_metric"].str.contains(eval_metric)]
    #     specific_df['eval_metric'] = specific_df['eval_metric'].replace(METRIC_MAP.keys(), METRIC_MAP.values())
        
    #     print(specific_df)
    #     fig = px.box(
    #         specific_df,
    #         x="method",
    #         y="value",
    #         color="eval_metric",
    #         facet_col="eval_metric",
    #         # barmode="group",
    #         title=f"{METRIC_MAP[eval_metric]} Value per Stego Method",
    #     )

    #     fig.update_xaxes(
    #         tickangle=45,
    #         title_text="Stegonography Method",
    #     )
    #     # Updat 
    #     # fig.update_yaxes(
    #     #     title_text=f"{METRIC_MAP[eval_metric]} Value", title_font={"size": 20}
    #     # )
    #     fig.show()

    #     continue

    # different_methods = [y for x, y in df.groupby('method', as_index=False)]
    # for method in different_methods:
    #     method_name = method.iloc[0]['method']
    #     stats = method.describe()
    #     stats.to_csv(os.path.join("evaluation", f"{method_name}-summary-stats.csv"))

    # grouped_by_method = df.groupby("method").mean()
    # print(grouped_by_method)
    # grouped_by_method.to_csv(os.path.join("evaluation", "eval-dataframe-mean-grouped.csv"))


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
                "lsb",
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
    running_df.to_csv(os.path.join("evaluation", "eval-dataframe-compression.csv"), index=False)

    # evaluation = p.run_comparison(
    #     cover_path=os.path.join("images", "cover", "099902.jpg"),
    #     secret_path=os.path.join("images", "secret", "test.jpg"),
    #     methods=[
    #         "lsb",
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
    # dfi.export(df_styled, "dataframe169.png")
    # print(df)
    # print(evaluation)


if __name__ == "__main__":
    # main()
    # df = pd.read_csv("./evaluation/eval-dataframe-mean-grouped.csv", index_col=0)
    # # df.style.apply(highlight_max).to_html("results.html")
    # df_styled = df.style.highlight_max(color="lightgreen", axis=0).highlight_min(
    #     color="lightcoral", axis=0)
    # dfi.export(df_styled, "dataframe4.png")
    analysis()
