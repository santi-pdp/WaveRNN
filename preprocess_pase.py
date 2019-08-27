import glob
import tqdm
import soundfile as sf
from utils.display import *
from utils.dsp import *
import torch
from pase.models.frontend import wf_builder
import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
from torch.utils.data import Dataset, DataLoader


def collater(batch):
    maxlen = 0
    for sample in batch:
        wav, _, _ = sample
        maxlen = max(len(wav), maxlen)

    wavs = []
    ids = []
    uttnames =[]
    slens = []
    for sample in batch:
        wav, id_, uttname = sample
        if len(wav) < hp.voc_seq_len * 5:
            # skip too short files
            continue
        P = maxlen - len(wav)
        slens.append(len(wav))
        if P > 0:
            wav = torch.cat((wav, torch.zeros(P)), dim=0)
        wavs.append(wav.view(1, -1))
        ids.append(id_)
        uttnames.append(uttname)
    return torch.cat(wavs, dim=0), ids, uttnames, slens

class WavDataset(Dataset):

    def __init__(self, wav_list, paths):
        super().__init__()
        self.wav_list = wav_list
        self.paths = paths


    def __getitem__(self, index):
        path = self.wav_list[index]
        uttname = path
        id = path.split('/')[-1][:-4]
        #y = load_wav(path)
        y, rate = sf.read(path)
        peak = np.abs(y).max()
        if hp.peak_norm or peak > 1.0:
            y /= peak
        if hp.voc_mode == 'RAW':
            quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = float_2_label(y, bits=16)
        np.save(f'{self.paths.quant}{id}.npy', quant, allow_pickle=False)
        return torch.FloatTensor(y), id, uttname

    def __len__(self):
        return len(self.wav_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
    parser.add_argument('--path', '-p', default=hp.wav_path, help='directly point to dataset path (overrides hparams.wav_path')
    parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    extension = args.extension
    path = args.path
    wav_files = get_files(path, extension)
    if hp.pase_cfg is None:
        raise ValueError
        assert hp.pase_ckpt is not None
    CUDA = torch.cuda.is_available() and hp.cuda
    hp.device = 'cuda' if CUDA else 'cpu'
    # Load pase model
    pase = wf_builder(hp.pase_cfg)
    pase.load_pretrained(hp.pase_ckpt, load_last=True, verbose=True)
    pase.to(hp.device)
    pase.eval()
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

    if len(wav_files) == 0:
        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')
    else:
        if not hp.ignore_tts:

            text_dict = ljspeech(path)

            with open(f'{paths.data}text_dict.pkl', 'wb') as f:
                pickle.dump(text_dict, f)

        simple_table([('Sample Rate', hp.sample_rate),
                      ('Bit Depth', hp.bits),
                      ('Mu Law', hp.mu_law),
                      ('Hop Length', hp.hop_length),
                      ('CPU Count', cpu_count())])
        num_workers = hp.num_workers
        dataloader = DataLoader(WavDataset(wav_files, paths), 
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                collate_fn=collater)
        dataset = []
        for bidx, batch in tqdm.tqdm(enumerate(dataloader, start=1),
                                     total=len(dataloader)):

            wav, ids, uttnames, slens = batch
            with torch.no_grad():
                wav = wav.to(hp.device)
                mel = pase(wav.unsqueeze(1)).cpu().data.numpy()
            assert mel.shape[1] == hp.num_mels, mel.shape
            for mi in range(mel.shape[0]):
                id = ids[mi]
                uttname = uttnames[mi]
                max_mel_len = slens[mi] // 160
                mel_ = mel[mi, :, :max_mel_len]
                np.save(f'{paths.mel}{id}.npy', mel_, allow_pickle=False)
                dataset += [(id, uttname, max_mel_len)]
            
        with open(f'{paths.data}dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
