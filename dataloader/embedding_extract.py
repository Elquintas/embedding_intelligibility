import torchaudio
import torch
import sys
import os
import yaml
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


# LOADS CONFIGURATION .YAML FILE
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print('Error reading the config file')

# RETURNS THE EMBEDDING EXTRACTO
# TWO SUPPORTED TYPES: X-VECTORS AND ECAPA_TDNN
def embedding_type():
    if config['embedding_type'] == 'x-vector':
        classifier = EncoderClassifier.from_hparams(source=\
                "speechbrain/spkrec-xvect-voxceleb", savedir=\
                "pretrained_models/spkrec-xvect-voxceleb")
        extension = 'xvec'
    elif config['embedding_type'] == 'ecapa_tdnn':
        classifier = EncoderClassifier.from_hparams(source=\
                "speechbrain/spkrec-ecapa-voxceleb", savedir=\
                "pretrained_models/spkrec-ecapatdnn-voxceleb")
        extension = 'ecapa'
    else:
        print('Unknown embedding type')
        sys.exit()

    return classifier, extension

# EXTRACTS THE SPEAKER EMBEDDING BASED ON THE USED CLASSIFIER
# CLASSIFIERS OBTAINED FROM HUGGINGFACE
def extractor(wavfile, classifier):
    
    signal, fs =torchaudio.load(wavfile)
    embedding = classifier.encode_batch(signal)

    return embedding




if __name__=="__main__":

    config_path = '../configs/parameters.yaml'
    config = load_config(config_path)

    extraction_files = os.listdir(config['wav_path'])

    classifier, ext = embedding_type()
    saving_path = config['embedding_path']+ext+'/'
    
    # CREATES SPECIFIC EMBEDDING SUBDIR IF IT DOESN'T EXIST
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)


    print("\nExtracting speaker embeddings...")
    with tqdm(total=len(extraction_files)) as pbar:
        for files in os.listdir(config['wav_path']):
            if files.endswith('.wav'):
                embedding = extractor(config['wav_path'] + files, classifier)
                buf = os.path.splitext(files)
                new_f = buf[0]+'_'+ext+'_emb.pickle'
            
                torch.save(embedding, saving_path+new_f)
                pbar.update(1)
            else:
                pbar.update(1)
                continue



