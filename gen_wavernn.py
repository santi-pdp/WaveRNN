from utils.dataset import get_vocoder_datasets
from utils.dsp import *
from models.fatchord_version import WaveRNN, PASEInjector
from utils.paths import GEnhancementPaths
import soundfile as sf
from utils.hparams import *
from pase.models.frontend import wf_builder
from utils.display import simple_table
import torch
import argparse


def gen_genh_testset(model, test_set, samples, batched, target, overlap, save_path,
                     hp=None, device='cpu'):

    k = model.get_step() // 1000
    trg = None
    hp.pase.eval()

    for i, (x, xm, xlm) in enumerate(test_set, 1):

        if i > samples: break

        print('\n| Generating: %i/%i' % (i, samples))
        x = x[0].numpy()
        xm = xm.unsqueeze(1).to(device)
        xlm = xlm.unsqueeze(1).to(device)
        with torch.no_grad():
            m = hp.pase(xm, xlm)
        xm = xm[0, 0].cpu().data.numpy()

        save_wav(x, f'{save_path}{k}k_steps_{i}_target.wav')
        save_wav(xm, f'{save_path}{k}k_steps_{i}_xm.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}{k}k_steps_{i}_{batch_str}.wav'


        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)

def gen_testset(model, test_set, samples, batched, target, overlap, save_path,
                hp=None, device='cpu'):

    k = model.get_step() // 1000
    trg = None

    for i, (m, xm, x, neigh) in enumerate(test_set, 1):

        if i > samples: break

        print('\n| Generating: %i/%i' % (i, samples))
        x = x[0].numpy()
        xm = xm.to(device)
        neigh = neigh.to(device)

        if hp.pase_cntnt is not None:
            hp.pase_cntnt.eval()
            with torch.no_grad():
                m = hp.pase_cntnt(xm.unsqueeze(1))
        if hp.conversion:
            if hp.pase_id is not None:
                hp.pase_id.eval()
                with torch.no_grad():
                    # speed up discarding grad info to backtrack the graph
                    trg = hp.pase_id(neigh.unsqueeze(1))

        save_wav(x, f'{save_path}{k}k_steps_{i}_target.wav')
        save_wav(xm[0].cpu().data.numpy(), f'{save_path}{k}k_steps_{i}_xm.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}{k}k_steps_{i}_{batch_str}.wav'


        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law,
                           trg_mel=trg)

def gen_genh_from_file(model, load_path, save_path, batched, target, overlap, 
                       hp=None, device='cpu'):

    k = model.get_step() // 1000
    file_name = load_path.split('/')[-1]
    hp.pase.eval()

    #wav = load_wav(load_path)
    wav, rate = sf.read(load_path)
    save_wav(wav, f'{save_path}__{file_name}__{k}k_steps_target.wav')

    if hp.pase is not None:
        wav = torch.FloatTensor(wav).view(1, 1, -1)
        wav = wav.to(device)
        #wav = wav / torch.max(torch.abs(wav))
        with torch.no_grad():
            mel = pase(wav)
        wav = wav.cpu().squeeze().data.numpy()
    else:
        mel = melspectrogram(wav)
        mel = torch.tensor(mel).unsqueeze(0)

    batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
    save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}.wav'

    _ = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)


def gen_from_file(model, load_path, save_path, batched, target, overlap,
                  pase_cntnt=None, pase_id=None, conversion_ref=None, device='cpu'):

    k = model.get_step() // 1000
    file_name = load_path.split('/')[-1]

    wav = load_wav(load_path)
    save_wav(wav, f'{save_path}__{file_name}__{k}k_steps_target.wav')
    
    if pase_cntnt is not None:
        wav = torch.FloatTensor(wav).view(1, 1, -1)
        wav = wav.to(device)
        wav = wav / torch.max(torch.abs(wav))
        with torch.no_grad():
            mel = pase_cntnt(wav)
        wav = wav.cpu().squeeze().data.numpy()
    else:
        mel = melspectrogram(wav)
        mel = torch.tensor(mel).unsqueeze(0)

    if conversion_ref is not None:
        trg_wav = load_wav(conversion_ref)
        save_wav(trg_wav, f'{save_path}__{file_name}__{k}k_steps_target-conversion.wav')
        if pase_id is not None:
            trg_wav = torch.FloatTensor(trg_wav).view(1, 1, -1)
            trg_wav = trg_wav.to(device)
            trg_wav = trg_wav / torch.max(torch.abs(trg_wav))
            with torch.no_grad():
                trg_mel = pase_id(trg_wav)
            trg_wav = trg_wav.cpu().squeeze().data.numpy()
        else:
            raise NotImplementedError
        if hp.conversion_mode == 0:
            # operate the mel data
            trg_avg = torch.mean(trg_mel, dim=2, keepdim=True)
            src_avg = torch.mean(mel, dim=2, keepdim=True)
            for alpha in [0.5, 1.0, 1.5, 2.0]:
                mel = mel - alpha * (src_avg - trg_avg)

                batch_str = f'alpha{alpha}_gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
                save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}.wav'

                _ = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)
        elif hp.conversion_mode == 1:
            # handle the target mel to the model
            batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
            save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}.wav'
            _ = model.generate(mel, save_str, batched, target, overlap,
                               hp.mu_law,
                               trg_mel=trg_mel)
    else:
        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}.wav'

        _ = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--samples', '-s', type=int, help='[int] number of utterances to generate')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--file', '-f', type=str, nargs='+', help='[string/path] for '
                        'testing a wav outside dataset', default=None)
    parser.add_argument('--conversion_ref', '-k', type=str, help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--weights', '-w', type=str, help='[string/path] checkpoint file to load weights from')
    parser.add_argument('--gta', '-g', dest='use_gta', action='store_true', help='Generate from GTA testset')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--pase_cfg', type=str, help='[string/path] checkpoint file to load weights from',
                        default=None)
    parser.add_argument('--pase_ckpt', type=str, help='[string/path] checkpoint file to load weights from',
                        default=None)

    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(samples=hp.voc_gen_at_checkpoint)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)
    parser.set_defaults(file=None)
    parser.set_defaults(weights=None)
    parser.set_defaults(gta=False)

    args = parser.parse_args()

    hp = HParams(args.hparams)

    batched = args.batched
    samples = args.samples
    target = args.target
    overlap = args.overlap
    file = args.file
    gta = args.gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.pase_feats,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)


    #paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    paths = GEnhancementPaths(hp.voc_model_id)

    restore_path = args.weights if args.weights else paths.voc_latest_weights

    model.restore(restore_path)

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    assert args.file is not None
    #_, test_set = get_vocoder_datasets(paths.data, 1, gta)

    # Load pase model
    print('Building PASE...')
    if args.pase_cfg is not None:
        """
        # 2 PASEs: (1) Identifier extractor, (2) Content extractor
        pase_cntnt = wf_builder(args.pase_cfg)
        pase_cntnt.load_pretrained(args.pase_cntnt_ckpt, load_last=True, verbose=True)
        pase_cntnt.to(device)
        pase_cntnt.eval()
        if conversion:
            pase_id = wf_builder(args.pase_cfg)
            pase_id.load_pretrained(args.pase_id_ckpt, load_last=True, verbose=True)
            pase_id.to(device)
        else:
            pase_id = None
        hp.pase_cntnt = pase_cntnt
        if conversion:
            pase_id.eval()
            hp.pase_id = pase_id
        """
        pase = PASEInjector(args.pase_cfg, None, False,
                            hp.num_mels, hp.pase_feats,
                            paths.voc_checkpoints, global_mode=hp.global_pase)
        pase.load_pretrained(args.pase_ckpt, load_last=True, verbose=True)
        pase.to(device)
        hp.pase = pase
        hp.pase.eval()
    else:
        hp.pase = None
        #hp.pase_id = hp.pase_cntnt = None
        #pase_id = pase_cntnt = None
    if file is not None:
        for f in file:
            #gen_from_file(model, f, paths.voc_output, batched, target, overlap,
            #              pase_cntnt=pase_cntnt, pase_id=pase_id, 
            #              conversion_ref=args.conversion_ref, device=device)
            gen_genh_from_file(model, f, paths.voc_output, batched, target,
                               overlap, hp=hp, device=device)
    else:
        gen_testset(model, test_set, samples, batched, target, overlap,
                    paths.voc_output, device=device)

    print('\n\nExiting...\n')
