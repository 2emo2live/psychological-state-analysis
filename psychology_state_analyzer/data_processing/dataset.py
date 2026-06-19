from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class MentalHealthDataset(Dataset):
    def __init__(self, root_path: str, split: str):
        super().__init__()
        file_name = split + ".csv"
        data_root = Path(root_path) / file_name
        self.split = split

        if not data_root.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_root}\n"
                "Use the `load_data` function to prepare the dataset before using this class."
            )

        df = pd.read_csv(data_root)
        self.texts = df["statement"]
        self.labels = df["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        return text, label
