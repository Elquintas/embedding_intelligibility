import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F 

with open("./configs/parameters.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)


class model_embedding_snn(nn.Module):
    def __init__(self):
        super(model_embedding_snn,self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(cfg['dropout'])

        self.batch_norm1 = nn.BatchNorm1d(cfg['first_layer'])
        self.batch_norm2 = nn.BatchNorm1d(cfg['second_layer'])
        self.batch_norm3 = nn.BatchNorm1d(cfg['third_layer'])

        self.fc1 = nn.Linear(cfg['first_layer'],cfg['second_layer'])
        self.fc2 = nn.Linear(cfg['second_layer'],cfg['third_layer'])
        
        self.fc_voix = nn.Linear(cfg['third_layer'],1)
        self.fc_res = nn.Linear(cfg['third_layer'],1)
        self.fc_pros = nn.Linear(cfg['third_layer'],1)
        self.fc_pd = nn.Linear(cfg['third_layer'],1)

        self.fc_int = nn.Linear(cfg['third_layer'],1)
        

    def forward(self, input_embs):

        x = self.batch_norm1(input_embs)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.batch_norm3(x)

        v = self.fc_voix(x)
        v = self.relu(v)
        r = self.fc_res(x)
        r = self.relu(r)
        p = self.fc_pros(x)
        p = self.relu(p)
        pd = self.fc_pd(x)
        pd = self.relu(pd)

        INT = self.fc_int(x)
        INT = self.relu(INT)

        return INT, v, r, p, pd
