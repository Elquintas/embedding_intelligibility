import sys
import torch
import torch.nn as nn
import yaml
import models.model
import utils.plotter as plotter
import dataloader.load
import numpy as np

from torch.utils.data import DataLoader
from scipy import stats

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

    testloader = DataLoader(dataset=test_set, batch_size=27)

    for ep in range(cfg['epochs']):
        
        TRAIN_LOSS = []
        TEST_LOSS = []

        print('EPOCH: {0}/{1}'.format(ep+1,cfg['epochs']))
        
        model_snn.train()
        
        for a, b in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            #[b0] --> filename
            #[b1] --> severity
            #[b2] --> intelligibility
            #[b3] --> voix
            #[b4] --> resonance
            #[b5] --> phonemic distortions

            sev_,int_,v_,r_,p_,pd_ = model_snn(b[0].squeeze().cuda())
            
            loss = criterion(int_.double(),b[2].unsqueeze(1).cuda())
            TRAIN_LOSS.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            #print(loss)

        model_snn.eval()
        for at, bt in enumerate(testloader):
            optimizer.zero_grad()
            
            sev_test,int_test,v_test,r_test,p_test,pd_test = model_snn(\
                    bt[0].squeeze().cuda())

            loss_test = criterion(int_test.double(),bt[2].unsqueeze(1).cuda())
            TEST_LOSS.append(loss_test.cpu().detach().numpy())
      
            if ep+1 == cfg['epochs']:
                predicted = int_test.squeeze().tolist()
                reference = bt[2].tolist()
                print(predicted,reference)
        

        print('Train Loss: {} ----- Test Loss: {}'.format(np.mean(TRAIN_LOSS),\
                np.mean(TEST_LOSS)))
        
    correlation = stats.spearmanr(predicted,reference)[0]
    print("Correlation (Spearman's): {}".format(correlation))
    plotter.plot_graph(predicted,reference,correlation,1)



