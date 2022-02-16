import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def weekly_breakdown_emotions(content: str, graph_type: str="bar"):
    df = pd.read_csv(f"data/{content}_toxicity.csv")
    if content != "caption":
        df2 = pd.read_csv(f"data/caption_toxicity.csv")
        df = df.merge(df2[["postdate", "imagename"]], left_on="postid", right_on="imagename", how="left")
    df["postdate"] = pd.to_datetime(df["postdate"])
    gbs = []
    emotions = ["anger", "fear", "sadness", "joy"]
    for emo in emotions:
        gb = df.groupby(pd.Grouper(key="postdate", freq="1W")) \
            .apply(lambda x: (sum(x[f"Topic_{emo}"]>0.5) / (len(x)+1e-10))*100)
        gbs.append(gb)

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.ylabel(f"Emotions Breakdown in {content[0].upper()+content[1:]}s by Percentage (%)")
    plt.xticks(rotation=90)
    for i, emo in enumerate(emotions):
        if graph_type == "bar":
            if i > 0:
                plt.bar(gbs[i].index, gbs[i].values, width=3, label=emo, bottom=accum)
                accum += gbs[i].values
            else:
                plt.bar(gbs[i].index, gbs[i].values, width=3, label=emo)
                accum = gbs[i].values
        else:
            plt.plot(gbs[i].index, gbs[i].values,'-o', label=emo)
    plt.legend()
    xfmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    pass


def weekly_breakdown_toxicity(content: str, label: str = "toxicity", aggregate: bool = False):
    df = pd.read_csv(f"data/{content}_toxicity.csv")
    if content != "caption":
        df2 = pd.read_csv(f"data/caption_toxicity.csv")
        df = df.merge(df2[["postdate", "imagename"]], left_on="postid", right_on="imagename", how="left")
    df["postdate"] = pd.to_datetime(df["postdate"])
    if aggregate:
        gb = df.groupby(pd.Grouper(key="postdate", freq="1W"))\
            .apply(lambda x: sum(
            (x[f"original_{label}"]>0.5) | (x[f"multilingual_{label}"]>0.5)
        ))
        gb2 = df.groupby(pd.Grouper(key="postdate", freq="1W")) \
            .apply(lambda x: sum(
            (x[f"original_identity_attack"] > 0.5) | (x[f"multilingual_identity_attack"] > 0.5)
        ))
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
    plt.bar(gb2.index, gb2.values, color="purple", width=3)
    plt.legend(["Toxicity", "Identity Attack"])
    xfmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    pass


def combine_original_and_multilingual(content: str):
    df = pd.read_csv(f"data/{content}_toxicity.csv")
    for label in ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]:
        df[f"original_multilingual_{label}"] = (df[f"original_{label}"]>0.5) | (df[f"multilingual_{label}"]>0.5)

    df.to_csv(f"data/{content}_toxicity.csv")


if __name__ == "__main__":
    # weekly_breakdown("caption", "toxicity")
    weekly_breakdown_toxicity("comment", "toxicity", aggregate=True)
    # combine_original_and_multilingual("comment")
    # weekly_breakdown_emotions("comment")
    # weekly_breakdown_emotions("comment", graph_type="line")