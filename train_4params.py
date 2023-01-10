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

def multi_task_loss(v,r,p,pd,b):

    l_v = criterion(v.double(),b[4].unsqueeze(1).cuda())
    l_r = criterion(r.double(),b[5].unsqueeze(1).cuda())
    l_p = criterion(p.double(),b[6].unsqueeze(1).cuda())
    l_pd = criterion(pd.double(),b[7].unsqueeze(1).cuda())

    return (0.163*l_v+0.014*l_r+0.314*l_p+0.660*l_pd)

def return_metrics(a,b):

    corr = stats.spearmanr(a,b)[0]
    error = rmse(a,b)
    return corr,error

def return_results(a,b):
    
    pred = np.array(a.squeeze().tolist())
    ref = np.array(b.tolist())
    
    return pred, ref


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
            
            #b[0] --> embedding
            #b[1] --> filename
            #b[2] --> severity
            #b[3] --> intelligibility
            #b[4] --> voix
            #b[5] --> resonance
            #b[6] --> prosody
            #b[7] --> phonemic distortions

            sev_,int_,v_,r_,p_,pd_ = model_snn(b[0].squeeze().cuda())

            loss = multi_task_loss(v_,r_,p_,pd_,b)

            TRAIN_LOSS.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()


        model_snn.eval()
        for at, bt in enumerate(testloader):
            optimizer.zero_grad()
            
            sev_test,int_test,v_test,r_test,p_test,pd_test = model_snn(\
                    bt[0].squeeze().cuda())
            
            loss_test = multi_task_loss(v_test,r_test,p_test,pd_test,bt)

            TEST_LOSS.append(loss_test.cpu().detach().numpy())
      
            if ep+1 == cfg['epochs']:                
                pred_v,ref_v = return_results(v_test,bt[4])
                pred_r,ref_r = return_results(r_test,bt[5])
                pred_p,ref_p = return_results(p_test,bt[6])
                pred_pd,ref_pd = return_results(pd_test,bt[7])

                pred_int = pred_v+pred_r+pred_p+pred_pd
                ref_int = np.array(bt[3].tolist())
        

        print('Train Loss: {} ----- Test Loss: {}'.format(np.mean(TRAIN_LOSS),\
                np.mean(TEST_LOSS)))
    


    corr_v,rmse_v = return_metrics(pred_v,ref_v)
    corr_r,rmse_r = return_metrics(pred_r,ref_r)
    corr_p,rmse_p = return_metrics(pred_p,ref_p)
    corr_pd,rmse_pd = return_metrics(pred_pd,ref_pd)

    corr_int,rmse_int = return_metrics(pred_int,ref_int)

    print("Correlation V(Spearman's): {} ----- RMSE: {}".format(corr_v,rmse_v))
    print("Correlation R(Spearman's): {} ----- RMSE: {}".format(corr_r,rmse_r))
    print("Correlation P(Spearman's): {} ----- RMSE: {}".format(corr_p,rmse_p))
    print("Correlation PD(Spearman's): {} ----- RMSE: {}".format(corr_pd,rmse_pd))

    print("Correlation INT(Spearman's): {} ----- RMSE: {}".format(corr_int,rmse_int))

    #plotter.plot_graph(predicted,reference,correlation,rmse)



