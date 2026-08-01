import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_split(split_name, features_path=None):
    """
    Load a specific split ('train', 'dev', or 'eval') from the cached
    RIR features CSV.

    Returns: (X, y)
      X = DataFrame with feature columns [t60, decay_slope, drr]
      y = Series with binary labels (1 = real, 0 = fake)
    """
    if features_path is None:
        features_path = project_root / "dataset" / "rir_features.csv"

    df = pd.read_csv(features_path)
    split_df = df[df["split"] == split_name].copy()

    feature_cols = ["t60", "decay_slope", "drr"]
    X = split_df[feature_cols]
    y = (split_df["label"] == "real").astype(int)   # real=1, fake=0

    return X, y


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")
    X_eval, y_eval = load_split("eval")

    print("Train:", X_train.shape, "Label counts:", y_train.value_counts().to_dict())
    print("Dev:  ", X_dev.shape, "Label counts:", y_dev.value_counts().to_dict())
    print("Eval: ", X_eval.shape, "Label counts:", y_eval.value_counts().to_dict())