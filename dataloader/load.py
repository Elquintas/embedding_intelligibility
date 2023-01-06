import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

class load_data(Dataset):
    def __init__(self, filename, datadir):

        self.filename = filename
        self.datadir = datadir
        xy = pd.read_csv(filename,header=None)

        self.file = xy.values[:,0]
        self.INT = xy.values[:,1]
        self.SEV = xy.values[:,2]
        self.V  =  xy.values[:,3]
        self.R  =  xy.values[:,4]
        self.P  =  xy.values[:,5]
        self.PD =  xy.values[:,6]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        return (self.file[idx], )
