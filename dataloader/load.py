import pandas as pd
import torch
import pickle
import yaml
from torch.utils.data import Dataset,DataLoader

#embedding_path = './data/embeddings/xvec/'

with open("./configs/parameters.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

#embedding_path = cfg['embedding_path'] 


class load_data(Dataset):
    def __init__(self, filename):

        self.filename = filename
        #self.datadir = datadir
        xy = pd.read_csv(filename,header=None)
        
        self.emb_file = xy.values[:,0]

        self.INT = xy.values[:,1]
        self.SEV = xy.values[:,2]
        self.V  =  xy.values[:,3]
        self.R  =  xy.values[:,4]
        self.P  =  xy.values[:,5]
        self.PD =  xy.values[:,6]

        self.n_samples = xy.shape[0]


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        emb = torch.load(cfg['embedding_path'] + self.emb_file[idx])

        return emb,self.emb_file[idx],self.INT[idx],self.SEV[idx],\
                self.V[idx],self.R[idx],self.P[idx],self.PD[idx]
