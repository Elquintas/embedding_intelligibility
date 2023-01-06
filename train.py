import sys
import torch
import torch.nn as nn
import yaml
import models.model
import dataloader.load

from torch.utils.data import DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
    optimizer = torch.optim.Adam(model_snn.parameters(),lr=cfg['learning_rate'])

    sequential = dataloader.load.load_data
    train_set = sequential(cfg['train_set_file'])
    test_set = sequential(cfg['test_set_file'])

    trainloader = DataLoader(dataset=train_set, batch_size=cfg['batch_size'],\
            shuffle=True,drop_last=True)

    testloader = DataLoader(dataset=test_set, batch_size=1)

    for ep in range(cfg['epochs']):
        print('EPOCH: {0}/{1}'.format(ep+1,cfg['epochs']))
        
        model_snn.train()
        
        for a, b in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            at,bt,ct,dt,et = model_snn(b[0].squeeze().cuda())
            
            loss = criterion(at.double(),b[2].unsqueeze(1).cuda())
            
            loss.backward()
            optimizer.step()

            print(loss)

            
