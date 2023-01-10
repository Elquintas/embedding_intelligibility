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

def return_metrics(a,b):

    corr = stats.spearmanr(a,b)[0]
    error = rmse(a,b)
    return corr,error


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = './configs/parameters.yaml'
    cfg = load_config(config_path)

    # MODEL LOADING
    model_snn = models.model.model_embedding_snn().cuda()
    model_snn.load_state_dict(torch.load(cfg['model_path']+'model_snn_'\
            +cfg['embedding_type']))
    model_snn.eval()

    # TEST DATASET LOADER
    sequential = dataloader.load.load_data
    test_set = sequential(cfg['test_set_file'])
    testloader = DataLoader(dataset=test_set, batch_size=27)

    for at, bt in enumerate(testloader):

        sev_test,int_test,v_test,r_test,p_test,pd_test = model_snn(\
                    bt[0].squeeze().cuda())

        pred_sev = np.array(sev_test.squeeze().tolist())
        ref_sev = np.array(bt[2].tolist())
        
        pred_int = np.array(int_test.squeeze().tolist())
        ref_int = np.array(bt[3].tolist())
        

    corr_sev,rmse_sev = return_metrics(pred_sev,ref_sev)
    corr_int,rmse_int = return_metrics(pred_int,ref_int)

    print("pred_sev",pred_sev)
    print("ref_sev",ref_sev)

    print("pred_int", pred_int)
    print("ref_int",ref_int)

    print("Test File: {}".format(cfg['test_set_file']))

    print("Correlation SEV(Spearman's): {} ----- RMSE: {}".format(corr_sev,rmse_sev))
    print("Correlation INT(Spearman's): {} ----- RMSE: {}".format(corr_int,rmse_int))

    plotter.plot_graph(pred_sev,ref_sev,corr_sev,rmse_sev)
    plotter.plot_graph(pred_int,ref_int,corr_int,rmse_int)



