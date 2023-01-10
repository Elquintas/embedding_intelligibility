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

def rmse(a,b):
    return np.sqrt(((a-b)**2).mean())

def multi_task_loss(s,i,v,r,p,pd,b):

    l_s = criterion(s.double(),b[2].unsqueeze(1).cuda())
    l_i = criterion(i.double(),b[3].unsqueeze(1).cuda())

    return (0.5*l_s+0.5*l_i)

def return_metrics(a,b):
     
    corr = stats.spearmanr(a,b)[0]
    error = rmse(a,b)
    return corr,error




if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = './configs/parameters.yaml'
    cfg = load_config(config_path)

    model_snn = models.model.model_embedding_snn().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_snn.parameters(),lr=cfg['learning_rate'])

    sequential = dataloader.load.load_data
    train_set = sequential(cfg['train_set_file'])
    validation_set = sequential(cfg['validation_set_file'])

    trainloader = DataLoader(dataset=train_set, batch_size=cfg['batch_size'],\
            shuffle=True,drop_last=True)

    val_loader = DataLoader(dataset=validation_set, batch_size=27)

    for ep in range(cfg['epochs']):
        
        TRAIN_LOSS = []
        VAL_LOSS = []

        print('EPOCH: {0}/{1}'.format(ep+1,cfg['epochs']))
        
        model_snn.train()
        
        for a, b in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            #b[0] --> embedding
            #b[1] --> filename
            #b[2] --> severity
            #b[3] --> intelligibility
            #b[4] --> voix
            #b[5] --> resonance
            #b[6] --> prosody
            #b[7] --> phonemic distortions

            sev_,int_,v_,r_,p_,pd_ = model_snn(b[0].squeeze().cuda())

            #loss = criterion(int_.double(),b[3].unsqueeze(1).cuda())

            loss = multi_task_loss(sev_,int_,v_,r_,p_,pd_,b)

            TRAIN_LOSS.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            #print(loss)

        model_snn.eval()
        for at, bt in enumerate(val_loader):
            optimizer.zero_grad()
            
            sev_val,int_val,v_val,r_val,p_val,pd_val = model_snn(\
                    bt[0].squeeze().cuda())

            #loss_test = criterion(int_test.double(),bt[3].unsqueeze(1).cuda())
            
            loss_val = multi_task_loss(sev_val,int_val,v_val,r_val,p_val,\
                    pd_val,bt)

            VAL_LOSS.append(loss_val.cpu().detach().numpy())
      
            if ep+1 == cfg['epochs']:
                pred_sev = np.array(sev_val.squeeze().tolist())
                ref_sev = np.array(bt[2].tolist())
                pred_int = np.array(int_val.squeeze().tolist())
                ref_int = np.array(bt[3].tolist())
                
                torch.save(model_snn.state_dict(), cfg['model_path']+'model_snn_'+\
                        cfg['embedding_type'])                
        

        print('Train Loss: {} ----- Validation Loss: {}'.format(np.mean(TRAIN_LOSS),\
                np.mean(VAL_LOSS)))
    

    corr_sev,rmse_sev = return_metrics(pred_sev,ref_sev)
    corr_int,rmse_int = return_metrics(pred_int,ref_int)

    print("Correlation SEV(Spearman's): {} ----- RMSE: {}".format(corr_sev,rmse_sev))
    print("Correlation INT(Spearman's): {} ----- RMSE: {}".format(corr_int,rmse_int))
    
    #plotter.plot_graph(predicted,reference,correlation,rmse)



