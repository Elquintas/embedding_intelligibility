import sys
import torch
import torch.nn as nn
import yaml
import models.model
import dataloader.load

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print('Error reading the config file')


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = './configs/parameters.yaml'
    cfg = load_config(config_path)

    model_snn = models.model.model_embedding_snn().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_snn.parameters())

    train_filename = cfg['data_path']+cfg['traininig_set_file']
    print(train_filename)

    for ep in range(cfg['epochs']):
        print(ep)
