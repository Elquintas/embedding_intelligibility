import torch
#import yaml
import model
import torchaudio

from speechbrain.pretrained import EncoderClassifier


#with open("./configs/parameters.yaml", "r") as ymlfile:
#    cfg = yaml.safe_load(ymlfile)

MODEL_PATH = './model_snn_x-vector'


class PythonPredictor():

    def __init__(self):
        super(PythonPredictor,self).__init__()

        self.model = model.model_embedding_snn().cuda()
        #self.model.load_state_dict(torch.load(cfg['model_path']+'model_snn_'\
        #    +cfg['embedding_type']))
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

        self.emb_extractor = EncoderClassifier.from_hparams(source=\
                "speechbrain/spkrec-xvect-voxceleb", savedir=\
                "pretrained_models/spkrec-xvect-voxceleb")

    def predict(self,wavfile):

        signal, fs =torchaudio.load(wavfile)
        emb = self.emb_extractor.encode_batch(signal)
        emb = emb.squeeze().unsqueeze(0).cuda()
        
        sev_,int_,v_,r_,p_,pd_=self.model(emb)
              
        print('\nFilename: {}\n'.format(wavfile))

        print('Speech Disorder Severity:   {}'.format(round(sev_.item(),3)))
        print('Speech Intelligibility:     {}\n'.format(round(int_.item(),3)))
        
