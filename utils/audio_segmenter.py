from pydub import AudioSegment
import os
import sys
import wave
import contextlib
from tqdm import tqdm

fname = AudioSegment.from_wav("./file1.wav")


# RETURNS THE DURATION OF AN AUDIOFILE IN MILISECS
def get_file_duration(wavfile):
    with contextlib.closing(wave.open(wavfile,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration*1000    # DURATION IN MILISECS FOR AUDIOSEGMENT

# SEGMENTS THE AUDIO FILE IN EQUAL LENGTH SEGMENTS
def segmenter(wavpath, wavfile, segs, savepath):

    duration = get_file_duration(wavpath+wavfile)
    buf = os.path.splitext(wavfile)
    i = 1
    j = 0
    tblock = duration/segs

    while (j+tblock) <= duration:
        newAudio = AudioSegment.from_wav(wavpath+wavfile)
        newAudio = fname[j:j+tblock]
        newAudio.export(savepath+buf[0]+'_seg'+\
                str(i)+'.wav', format='wav')
        j += tblock
        i += 1
    

if __name__=="__main__":

    segments = 3
    wavfile = './file1.wav'
    
    wav_path = '../data/wavs/'
    saving_path = '../data/wavs/segmented/'
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    total_files = os.listdir(wav_path)
    
    with tqdm(total=len(total_files)) as pbar:
        for files in os.listdir(wav_path):
            if files.endswith('.wav'):
                segmenter(wav_path,files,segments,saving_path)
                pbar.update(1)
            else:
                pbar.update(1)
                continue

    #segmenter(wavfile, segments)

    






