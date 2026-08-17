import config as C
import pandas as pd


def create_balanced_dataset(df):
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n_spam, random_state=1009)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=1009).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(C.SPAM_FILE, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv(C.SPAM_TRAIN_CSV, index=None)
    val_df.to_csv(C.SPAM_VAL_CSV, index=None)
    test_df.to_csv(C.SPAM_TEST_CSV, index=None)
