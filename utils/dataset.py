import pickle
import random
import tqdm
import glob
import os
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
#import hparams as hp
from utils.text import text_to_sequence
import random
random.seed(1)

###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class GEnhancementOnlineDataset(Dataset):
    
    def __init__(self, data_root, hp, transform, cache=False,
                 test=None):
        super().__init__()
        self.cache = cache
        self.wavs = glob.glob(os.path.join(data_root, '*.wav'))
        if len(self.wavs) == 0:
            raise ValueError('No wavs found in {}'.format(data_root))
        if cache:
            self.data = []
            for wfile in tqdm.tqdm(self.wavs, total=len(self.wavs)):
                self.data.append({'wav':sf.read(wfile, dtype='int16')[0],
                                  'uttname':wfile})
        self.hp = hp
        self.test = test
        self.transform = transform

    def discretize(self, x):
        hp = self.hp
        if hp.voc_mode == 'RAW':
            x = encode_mu_law(x, mu=2 ** hp.bits) if hp.mu_law else \
                    float_2_label(x, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            x = float_2_label(x, bits=16)
        x = x.astype(np.int64)
        return x

    #@profile
    def __getitem__(self, index):
        hp = self.hp
        # pick clean file
        if hasattr(self, 'data'):
            cwav = self.data[index]['wav']
            cfile = self.data[index]['uttname']
        else:
            cfile = self.wavs[index]
            cwav, rate = sf.read(cfile)
        cwav = cwav.astype(np.float32) / (2 ** 15)
        # pick corresponding noisy
        y = self.discretize(cwav)
        # chunk
        if not self.test:
            mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
            T = len(cwav) // hp.hop_length
            max_offset = T -2 - (mel_win + 2 * hp.voc_pad)
            assert max_offset > 0, max_offset
            mel_offset = np.random.randint(0, max_offset)
            sig_offset = (mel_offset + hp.voc_pad) * hp.hop_length 
            # select random signal window
            nwav = cwav
            m = nwav[sig_offset:sig_offset + mel_win * hp.hop_length]
            pkg = {'chunk':torch.FloatTensor(m), 'uttname':cfile}
            m = self.transform(pkg)['chunk']
            if hp.context_size is not None:
                if len(nwav) < hp.context_size:
                    P_ = hp.context_size - len(nwav)
                    nwav = np.concatenate((nwav, np.zeros(P_,)), axis=0)
                    longidx = 0
                elif len(nwav) == hp.context_size:
                    longidx = 0
                else:
                    longidx = np.random.randint(0, len(nwav) - hp.context_size)
                longm = nwav[longidx:longidx + hp.context_size]
                pkg = {'chunk':torch.FloatTensor(longm), 'uttname':cfile}
                longm = self.transform(pkg)['chunk']
                m = (m, longm)
            lab = torch.tensor(y[sig_offset:sig_offset + hp.voc_seq_len + 1]).long()
            # AR input and output
            x = lab[:hp.voc_seq_len]
            y = lab[1:]
            # convert x to float (possibly with mu law)
            x = label_2_float(x.float(), hp.bits)
            if hp.voc_mode == 'MOL':
                y = label_2_float(y.float(), hp.bits)
            return x, y, m
        else:
            nwav = self.transform({'chunk':torch.FloatTensor(cwav),
                                   'uttname':cfile})['chunk']
            bits = 16 if hp.voc_mode == 'MOL' else hp.bits
            x = y
            if hp.mu_law and hp.voc_mode != 'MOL':
                x = decode_mu_law(x, 2 ** bits, from_labels=True)
            else:
                x = label_2_float(x, bits)
            xm = nwav.float()
            if hp.context_size is not None:
                xlm = nwav.float()
                xm = (xm, xlm)
            return x, xm

    def __len__(self):
        return len(self.wavs)


class GEnhancementDataset(Dataset):

    def __init__(self, clean, noisy, hp, test=False):
        super().__init__()
        self.clean_wavs = glob.glob(os.path.join(clean, '*.wav'))
        if len(self.clean_wavs) == 0:
            raise ValueError('No clean wavs found')
        self.noisy = noisy
        self.hp = hp
        self.test = test

    def discretize(self, x):
        hp = self.hp
        if hp.voc_mode == 'RAW':
            x = encode_mu_law(x, mu=2 ** hp.bits) if hp.mu_law else \
                    float_2_label(x, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            x = float_2_label(x, bits=16)
        x = x.astype(np.int64)
        return x

    def __getitem__(self, index):
        hp = self.hp
        # pick clean file
        cfile = self.clean_wavs[index]
        cwav, rate = sf.read(cfile)
        # pick corresponding noisy
        bname = os.path.basename(cfile)
        nfile = os.path.join(self.noisy, bname)
        nwav , rate = sf.read(nfile)
        assert len(nwav) == len(cwav)
        y = self.discretize(cwav)
        # chunk
        if not self.test:
            mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
            T = len(nwav) // hp.hop_length
            max_offset = T -2 - (mel_win + 2 * hp.voc_pad)
            mel_offset = np.random.randint(0, max_offset)
            sig_offset = (mel_offset + hp.voc_pad) * hp.hop_length 
            # select random signal window
            m = torch.FloatTensor(nwav[sig_offset:sig_offset + mel_win * hp.hop_length])
            if len(nwav) < hp.context_size:
                P_ = hp.context_size - len(nwav)
                nwav = np.concatenate((nwav, np.zeros(P_,)), axis=0)
                longidx = 0
            elif len(nwav) == hp.context_size:
                longidx = 0
            else:
                longidx = np.random.randint(0, len(nwav) - hp.context_size)
            longm = torch.FloatTensor(nwav[longidx:longidx + hp.context_size])
            lab = torch.tensor(y[sig_offset:sig_offset + hp.voc_seq_len + 1]).long()
            # AR input and output
            x = lab[:hp.voc_seq_len]
            y = lab[1:]
            # convert x to float (possibly with mu law)
            x = label_2_float(x.float(), hp.bits)
            if hp.voc_mode == 'MOL':
                y = label_2_float(y.float(), hp.bits)
            return x, y, m, longm
        else:
            bits = 16 if hp.voc_mode == 'MOL' else hp.bits
            x = y
            if hp.mu_law and hp.voc_mode != 'MOL':
                x = decode_mu_law(x, 2 ** bits, from_labels=True)
            else:
                x = label_2_float(x, bits)
            xm = torch.FloatTensor(nwav) 
            xlm = torch.FloatTensor(nwav)
            return x, xm, xlm

    def __len__(self):
        return len(self.clean_wavs)


#def get_genh_datasets(train_clean, train_noisy, valid_clean, valid_noisy,
#                      test_clean, test_noisy, batch_size, hparams, 
#                      num_workers=1):
def get_genh_datasets(train_clean, valid_clean, 
                      test_clean, batch_size, hparams, 
                      transform,
                      cache,
                      num_workers=1):


    #train_dataset = GEnhancementDataset(train_clean, train_noisy, hparams)
    train_dataset = GEnhancementOnlineDataset(train_clean, hparams, transform,
                                              cache=cache)

    valid_dataset = GEnhancementOnlineDataset(valid_clean, hparams, transform,
                                        cache=cache)

    test_dataset = GEnhancementOnlineDataset(test_clean, hparams, transform,
                                       cache=cache,
                                       test=True)

    train_set = DataLoader(train_dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=False,
                          pin_memory=True)

    valid_set = DataLoader(valid_dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=False,
                           pin_memory=True)

    return train_set, valid_set, test_set

class VocoderDataset(Dataset):
    def __init__(self, ids, path, uttnames=None, 
                 transforms=None, train_gta=False,
                 test=False):
        self.metadata = ids
        self.uttnames = uttnames
        if uttnames is not None:
            assert len(self.uttnames) == len(ids)
        self.mel_path = f'{path}gta/' if train_gta else f'{path}mel/'
        self.quant_path = f'{path}quant/'
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index):
        id = self.metadata[index]
        if self.uttnames is not None:
            uttname = self.uttnames[index]
            wav, rate = sf.read(uttname)
            # randomly chunk a piece of the wav to make
            # a neighbor from the same speaker as ID representative
            ridx = np.random.randint(0, len(wav) - hp.voc_seq_len)
            neigh_wav = torch.FloatTensor(wav[ridx:ridx+hp.voc_seq_len])
        else:
            neigh_wav = torch.zeros(hp.voc_seq_len)
        m = np.load(f'{self.mel_path}{id}.npy')
        x = np.load(f'{self.quant_path}{id}.npy')
        if not self.test:
            mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
            #max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
            max_offset = m.shape[-1] -2 - (mel_win + 2 * hp.voc_pad)
            mel_offset = np.random.randint(0, max_offset)
            sig_offset = (mel_offset + hp.voc_pad) * hp.hop_length 
            # select random mel window
            m = torch.tensor(m[:, mel_offset:mel_offset + mel_win]).float()
            # select random label window (signal level)
            labels = torch.tensor(x[sig_offset:sig_offset + hp.voc_seq_len + 1]).long()
        else:
            m = torch.tensor(m).float()
        # Create the mel span version in time in case feature extraction
        # is done on the fly with additional distortions
        bits = 16 if hp.voc_mode == 'MOL' else hp.bits
        if self.test:
            if hp.mu_law and hp.voc_mode != 'MOL':
                x = decode_mu_law(x, 2 ** bits, from_labels=True)
            else:
                x = label_2_float(x, bits)
            xm = torch.FloatTensor(x) 
            if self.transforms is not None:
                xm = self.transforms({'chunk':xm})['chunk'].float()
                #xm = xm / torch.max(torch.abs(xm))
        else:
            xm = x[sig_offset:sig_offset + hp.voc_seq_len + (2 * hp.voc_pad * \
                                                             hp.hop_length)]
            if hp.mu_law and hp.voc_mode != 'MOL':
                # decode mu x for xm
                xm = decode_mu_law(xm, 2 ** bits, from_labels=True)
            xm = torch.FloatTensor(xm)
            if self.transforms is not None:
                xm = self.transforms({'chunk':xm})['chunk'].float()
                #xm = xm / torch.max(torch.abs(xm))
            x = labels[:hp.voc_seq_len]
            y = labels[1:]
            # convert x to float (possibly with mu law)
            x = label_2_float(x.float(), bits)
            if hp.voc_mode == 'MOL':
                y = label_2_float(y.float(), bits)

        if self.test:
            return m, xm, x, neigh_wav
        else:
            return m, xm, x, y, neigh_wav

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path, batch_size, train_gta, num_workers=1,
                         transforms=None, spk2split=None):

    with open(f'{path}dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    dataset_uttnames = []
    for x in dataset:
        if len(x) == 3:
            dataset_ids.append(x[0])
            dataset_uttnames.append(x[1])
        else:
            assert len(x) == 2
            dataset_ids.append(x[0])
            dataset_uttnames = None

    random.seed(1234)
    random.shuffle(dataset_ids)

    print(dataset_ids[:5])
    if spk2split is None:
        test_ids = dataset_ids[-hp.voc_test_samples:]
        train_ids = dataset_ids[:-hp.voc_test_samples]
        if dataset_uttnames is not None:
            test_uttnames = dataset_uttnames[-hp.voc_test_samples:]
            train_uttnames = dataset_uttnames[:-hp.voc_test_samples]
        else:
            test_uttnames = train_uttnames = None
        valid_ids = None
    else:
        # map each id to the diff split
        splits = {'id':{'train':[], 'valid':[], 'test':[]},
                  'uttname':{'train':[], 'valid':[], 'test':[]}}
        for idi, id_ in enumerate(dataset_ids):
            spkid = id_.split('_')[0]
            splits['id'][spk2split[spkid]].append(id_)
            if dataset_uttnames is not None:
                uttname = dataset_uttnames[idi]
                splits['uttname'][spk2split[spkid]].append(uttname)

        train_ids = splits['id']['train']
        test_ids = splits['id']['test']
        valid_ids = splits['id']['valid']
        if dataset_uttnames is not None:
            train_uttnames = splits['uttname']['train']
            test_uttnames = splits['uttname']['test']
            valid_uttnames = splits['uttname']['valid']
        else:
            train_uttnames = test_uttnames = valid_uttnames = None


    train_dataset = VocoderDataset(train_ids, path, train_uttnames,
                                   transforms=transforms, train_gta=train_gta)

    test_dataset = VocoderDataset(test_ids, path, test_uttnames, 
                                  transforms=transforms, 
                                  train_gta=train_gta,
                                  test=True)
    if valid_ids is not None:
        valid_dataset = VocoderDataset(valid_ids, path, valid_uttnames,
                                       transforms=transforms, train_gta=train_gta)

    train_set = DataLoader(train_dataset,
                           #collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)
    if valid_ids is not None:
        valid_set = DataLoader(valid_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=False,
                               pin_memory=True)
    else:
        valid_set = None

    return train_set, test_set, valid_set


"""
def collate_vocoder(batch):
    # raw wavs
    wavs = [x[2] for x in batch]
    # build neighbors list too
    neighs = [x[3] for x in batch]


    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)
    neighs = np.stack(neighs).astype(np.float32)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()
    neighs = torch.tensor(neighs)



    return x, y, mels, neighs
"""


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################


def get_tts_dataset(path, batch_size, r):

    with open(f'{path}dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []

    for (id, len) in dataset:
        if len <= hp.tts_max_mel_len:
            dataset_ids += [id]
            mel_lengths += [len]

    with open(f'{path}text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    train_dataset = TTSDataset(path, dataset_ids, text_dict)

    sampler = None

    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=1,
                           pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))
    attn_example = dataset_ids[longest]

    # print(attn_example)

    return train_set, attn_example


class TTSDataset(Dataset):
    def __init__(self, path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        id = self.metadata[index]
        x = text_to_sequence(self.text_dict[id], hp.tts_cleaner_names)
        mel = np.load(f'{self.path}mel/{id}.npy')
        mel_len = mel.shape[-1]
        return x, mel, id, mel_len

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, ids, mel_lens


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)














