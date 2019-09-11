import glob
import tqdm
from utils.display import *
from utils.dsp import *
import torch
import os
from pase.models.frontend import wf_builder
#import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
from utils.hparams import HParams
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
import json


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', default=hp.wav_path, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--cfg', type=str, default=None)
parser.add_argument('--mel_guia', type=str, default=None)
parser.add_argument('--wav_guia', type=str, default=None)
args = parser.parse_args()

extension = args.extension
path = args.path

hp = HParams(args.cfg)


def convert_file(in_path, out_path):
    # load the output waveform to be predicted
    y = load_wav(out_path)
    if len(y) < hp.voc_seq_len * 3:
        # skip too short files
        return None, None
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    # load the input waveform from which melspec is computed
    x = load_wav(in_path)
    mel = melspectrogram(x)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path):
    if 'in_path' in path:
        in_path = path['in_path']
    else:
        in_path = path['out_path']
    path = path['out_path']
    id = path.split('/')[-1][:-4]
    m, x = convert_file(in_path, path)
    if m is None:
        return None, None
    np.save(f'{paths.mel}{id}.npy', m, allow_pickle=False)
    np.save(f'{paths.quant}{id}.npy', x, allow_pickle=False)
    return id, m.shape[-1]

if args.wav_guia is not None:
    wav_files = []
    # read a list for cheaper computation
    with open(args.wav_guia, 'r') as guia_f:
        if args.mel_guia is not None:
            # There are other input wav files out of which
            # melspecs are extracted, include those in wav_files
            with open(args.mel_guia, 'r') as mguia_f:
                mel_files = dict((os.path.basename(l.rstrip()), l.rstrip()) for l in mguia_f)
            for l in guia_f:
                l = l.rstrip()
                basename = os.path.basename(l)
                # look for this baename in the mel_guia
                # if it does not exist, raise Error
                if basename not in mel_files:
                    raise ValueError('File {} not found in mel guia {}'
                                     ''.format(basename, args.mel_guia))
                wav_files.append({'in_path':mel_files[basename], 
                                  'out_path':l})
        else:
            wav_files = [l.rstrip() for l in guia_f]
else:
    wav_files = get_files(path, extension)

paths = Paths(hp.data_path, hp.voc_model_id, '')#hp.tts_model_id)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

if len(wav_files) == 0:

    print('Please point wav_path in hparams.py to your dataset,')
    print('or use the --path option.\n')

else:

    simple_table([('Sample Rate', hp.sample_rate),
                  ('Bit Depth', hp.bits),
                  ('Mu Law', hp.mu_law),
                  ('Hop Length', hp.hop_length),
                  ('CPU Count', cpu_count())])
    num_workers = hp.num_workers
    dataset = []
    if num_workers > 0:
        pool = Pool(processes=cpu_count())

        for i, (id, length) in tqdm.tqdm(enumerate(pool.imap_unordered(process_wav, wav_files), 1),
                                         total=len(wav_files)):
            if id is not None:
                dataset += [(id, length)]
                #bar = progbar(i, len(wav_files))
                #message = f'{i}/{len(wav_files)} '
                #stream(message)
    else:
        for i, wav in tqdm.tqdm(enumerate(wav_files, 1), total=len(wav_files)):
            id, length = process_wav(wav)
            if id is not None:
                dataset += [(id, length)]
                #bar = progbar(i, len(wav_files))
                #message = f'{bar} {i}/{len(wav_files)} '
                #stream(message)

    with open(f'{paths.data}dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
