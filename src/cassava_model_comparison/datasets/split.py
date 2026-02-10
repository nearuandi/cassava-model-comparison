# Standard library
from typing import Tuple

# Third-party
import pandas as pd
from sklearn.model_selection import train_test_split


# split_train_val_test
# train 70%, val 20%, test 10%
def split_train_val_test(
    df: pd.DataFrame,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=random_seed
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=1/3,
        stratify=holdout_df["label"],
        random_state=random_seed
    )

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))