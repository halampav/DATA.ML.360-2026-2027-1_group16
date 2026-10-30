from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    t_val: int
    t_test: int

    def summary(self) -> str:
        lines = []

        for name in ("train", "val", "test"):
            df = getattr(self, name)

            start = pd.to_datetime(df["ts"].min(), unit="s").date()
            end = pd.to_datetime(df["ts"].max(), unit="s").date()
            
            lines.append(f"{name:5s} {len(df):8d} rows  {start} -> {end}")
        
        return "\n".join(lines)
