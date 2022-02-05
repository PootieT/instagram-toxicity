import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def weekly_breakdown(content: str, label: str = "toxicity", aggregate: bool = False):
    df = pd.read_csv(f"data/{content}_toxicity.csv")
    if content != "caption":
        df2 = pd.read_csv(f"data/caption_toxicity.csv")
        df = df.merge(df2[["postdate", "imagename"]], left_on="postid", right_on="imagename", how="left")
    df["postdate"] = pd.to_datetime(df["postdate"])
    if aggregate:
        gb = df.groupby(pd.Grouper(key="postdate", freq="1W"))\
            .apply(lambda x: sum(
            (x[f"original_{label}"]>0.5) | \
            (x[f"unbiased_{label}"]>0.5) | \
            (x[f"multilingual_{label}"]>0.5))
        )
    else:
        gb = df.groupby(pd.Grouper(key="postdate", freq="1W")) \
            .apply(lambda x: sum(
            (x[f"original_{label}"] > 0.5)
        ))

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.ylabel(f"Number of Toxic {content[0].upper()+content[1:]}s")
    plt.xticks(rotation=90)
    plt.bar(gb.index, gb.values, color="red", width=3)
    plt.legend(["Toxicity"])
    xfmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    pass


if __name__ == "__main__":
    weekly_breakdown("caption", "toxicity")
    # weekly_breakdown("comment", "toxicity")
