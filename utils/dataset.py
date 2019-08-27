import pickle
import random
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
import hparams as hp
from utils.text import text_to_sequence
import random
random.seed(1)

###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset):
    def __init__(self, ids, path, uttnames, 
                 transforms=None, train_gta=False,
                 test=False):
        self.metadata = ids
        self.uttnames = uttnames
        assert len(self.uttnames) == len(ids)
        self.mel_path = f'{path}gta/' if train_gta else f'{path}mel/'
        self.quant_path = f'{path}quant/'
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index):
        id = self.metadata[index]
        uttname = self.uttnames[index]
        wav, rate = sf.read(uttname)
        # randomly chunk a piece of the wav to make
        # a neighbor from the same speaker as ID representative
        ridx = np.random.randint(0, len(wav) - hp.voc_seq_len)
        neigh_wav = wav[ridx:ridx+hp.voc_seq_len]
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
                xm = xm / torch.max(torch.abs(xm))
        else:
            xm = x[sig_offset:sig_offset + hp.voc_seq_len + (2 * hp.voc_pad * \
                                                             hp.hop_length)]
            if hp.mu_law and hp.voc_mode != 'MOL':
                # decode mu x for xm
                xm = decode_mu_law(xm, 2 ** bits, from_labels=True)
            xm = torch.FloatTensor(xm)
            if self.transforms is not None:
                xm = self.transforms({'chunk':xm})['chunk'].float()
                xm = xm / torch.max(torch.abs(xm))
            x = labels[:hp.voc_seq_len]
            y = labels[1:]
            # convert x to float (possibly with mu law)
            x = label_2_float(x.float(), bits)
            if hp.voc_mode == 'MOL':
                y = label_2_float(y.float(), bits)

        if self.test:
            return m, xm, x, torch.FloatTensor(neigh_wav)
        else:
            return m, xm, x, y, torch.FloatTensor(neigh_wav)

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path, batch_size, train_gta, num_workers=1,
                         transforms=None):

    with open(f'{path}dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    dataset_uttnames = []
    for x in dataset:
        dataset_ids.append(x[0])
        dataset_uttnames.append(x[1])

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    test_uttnames = dataset_uttnames[-hp.voc_test_samples:]
    train_uttnames = dataset_uttnames[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(train_ids, path, train_uttnames,
                                   transforms=transforms, train_gta=train_gta)
    test_dataset = VocoderDataset(test_ids, path, test_uttnames, 
                                  transforms=transforms, 
                                  train_gta=train_gta,
                                  test=True)

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

    return train_set, test_set


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














