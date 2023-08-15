from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

class SMKRedditDataset(Dataset):
    """Solomonk’s Reddit Mental Health (SMK)"""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the .csvs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.adhd_csv = pd.read_csv(root_dir + "/adhd.csv")
        self.aspergers_csv = pd.read_csv(root_dir + "/aspergers.csv")
        self.depression_csv = pd.read_csv(root_dir + "/depression.csv")
        self.ocd_csv = pd.read_csv(root_dir + "/ocd.csv")
        self.ptsd_csv = pd.read_csv(root_dir + "/ptsd.csv")
        
        self.root_dir = root_dir
        self.data=[]
        for item in self.adhd_csv["body"]:
            self.data.append([item, "adhd"])

        for item in self.aspergers_csv["body"]:
            self.data.append([item, "aspergers"])
        
        for item in self.depression_csv["body"]:
            self.data.append([item, "depression"])
        
        for item in self.ocd_csv["body"]:
            self.data.append([item, "ocd"])

        for item in self.ptsd_csv["body"]:
            self.data.append([item, "ptsd"])
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

## write a test for this class
# dataset = SMKRedditDataset(root_dir="./Dataset/reddit_mental_health_posts/")
# print(dataset[0])
# print(len(dataset))
