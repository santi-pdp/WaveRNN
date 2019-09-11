import time
import numpy as np
import random
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.hparams import HParams
from pase.models.frontend import wf_builder
from pase.models.modules import Saver
from pase.transforms import *
from utils.dsp import *
from utils.dataset import get_vocoder_datasets
from utils.distribution import discretized_mix_logistic_loss
#import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import json
import argparse
import radam


def voc_train_loop(model, loss_func, optimiser, train_set, eval_set, test_set,
                   lr, total_steps, device, hp):

    for p in optimiser.param_groups: p['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1
    trg = None
    patience = hp.patience
    min_val_loss = np.inf

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.
        running_pase_reg_loss = 0.
        running_nll_loss = 0.
        pase_reg_loss = None

        for i, (m, xm, x, y, neigh) in enumerate(train_set, 1):
            m, xm, x, y, neigh = m.to(device), xm.to(device), x.to(device), y.to(device), neigh.to(device)

            if hp.pase_cntnt is not None:
                if hp.pase_cntnt_ft:
                    m_clean = m
                    m = hp.pase_cntnt(xm.unsqueeze(1))
                    if hp.pase_lambda > 0:
                        # use an MSE loss weighted with pase_lamda
                        # that tights the distorted PASE output
                        # to the clean PASE soft-labels (loaded in m)
                        pase_reg_loss = hp.pase_lambda * F.mse_loss(m, m_clean)
                else:
                    with torch.no_grad():
                        m = hp.pase_cntnt(xm.unsqueeze(1))
            if hp.conversion:
                if hp.pase_id is not None:
                    if hp.pase_id_ft:
                        trg = hp.pase_id(neigh.unsqueeze(1))
                    else:
                        with torch.no_grad():
                            # speed up discarding grad info to backtrack the graph
                            trg = hp.pase_id(neigh.unsqueeze(1))

            y_hat = model(x, m, trg)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)


            loss = loss_func(y_hat, y)

            running_nll_loss += loss.item()

            optimiser.zero_grad()
            if pase_reg_loss is not None:
                total_loss = loss + pase_reg_loss
                running_pase_reg_loss += pase_reg_loss.item()
                pase_reg_avg_loss = running_pase_reg_loss / i
            else:
                total_loss = loss
            total_loss.backward()
            optimiser.step()
            running_loss += total_loss.item()

            speed = i / (time.time() - start)
            nll_avg_loss = running_nll_loss / i
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_write_every == 0:
                hp.writer.add_scalar('train/nll', avg_loss, step)
                if pase_reg_loss is not None:
                    hp.writer.add_scalar('train/pase_reg_loss', pase_reg_avg_loss,
                                         step)

            if step % hp.voc_checkpoint_every == 0:
                if eval_set is not None:
                    print('Validating')
                    # validate the model
                    val_loss = voc_eval_loop(model, loss_func, eval_set, device)
                    if val_loss <= min_val_loss:
                        patience = hp.patience
                        print('Val loss improved: {:.4f} -> '
                              '{:.4f}'.format(min_val_loss, val_loss))
                        min_val_loss = val_loss
                    else:
                        patience -= 1
                        print('Val loss did not improve. Patience '
                              '{}/{}'.format(patience, hp.patience))
                        if patience == 0:
                            print('Out of patience. Breaking the loop')
                            break
                    # set to train mode again
                    model.train()
                # generate some test samples
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output,
                            hp=hp, device=device)
                model.checkpoint(paths.voc_checkpoints)
                if hp.pase_cntnt is not None and hp.pase_cntnt_ft:
                    hp.pase_cntnt.train()
                    hp.pase_cntnt.save(paths.voc_checkpoints, step)
                if hp.conversion:
                    if hp.pase_id is not None and hp.pase_id_ft:
                        hp.pase_id.train()
                        hp.pase_id.save(paths.voc_checkpoints, step)

            if pase_reg_loss is None:
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | NLLoss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            else:
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Total Loss: {avg_loss:.4f} | NLLoss: {avg_nll_loss:.4f} | PASE reg loss: {pase_reg_avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        model.save(paths.voc_latest_weights)
        model.log(paths.voc_log, msg)
        print(' ')

def voc_eval_loop(model, loss_func, eval_set, device):

    total_iters = len(eval_set)
    trg = None
    model.eval()

    with torch.no_grad():
        start = time.time()
        running_loss = 0.
        running_pase_reg_loss = 0.
        running_nll_loss = 0.
        pase_reg_loss = None

        for i, (m, xm, x, y, neigh) in enumerate(eval_set, 1):
            m, xm, x, y, neigh = m.to(device), xm.to(device), x.to(device), y.to(device), neigh.to(device)

            if hp.pase_cntnt is not None:
                m_clean = m
                m = hp.pase_cntnt(xm.unsqueeze(1))
                if hp.pase_lambda > 0:
                    # use an MSE loss weighted with pase_lamda
                    # that tights the distorted PASE output
                    # to the clean PASE soft-labels (loaded in m)
                    pase_reg_loss = hp.pase_lambda * F.mse_loss(m, m_clean)
            if hp.conversion:
                trg = hp.pase_id(neigh.unsqueeze(1))

            y_hat = model(x, m, trg)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)

            loss = loss_func(y_hat, y)

            running_nll_loss += loss.item()

            if pase_reg_loss is not None:
                total_loss = loss + pase_reg_loss
                running_pase_reg_loss += pase_reg_loss.item()
                pase_reg_avg_loss = running_pase_reg_loss / i
            else:
                total_loss = loss

            running_loss += total_loss.item()

            speed = i / (time.time() - start)
            nll_avg_loss = running_nll_loss / i
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if pase_reg_loss is None:
                msg = f'| EVAL {i}/{total_iters} | NLLoss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            else:
                msg = f'| EVAL {i}/{total_iters} | Total Loss: {avg_loss:.4f} | NLLoss: {avg_nll_loss:.4f} | PASE reg loss: {pase_reg_avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        hp.writer.add_scalar('eval/nll', nll_avg_loss, step)
        if pase_reg_loss is not None:
            hp.writer.add_scalar('eval/pase_reg_loss', pase_reg_avg_loss,
                                 step)

        print(' ')
        return avg_loss


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override '
                        'hparams.py learning rate', default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, 
                        help='[int] override hparams.py batch size',
                        default=32)
    parser.add_argument('--num_workers', '-w', type=int, help='[int]',
                        default=1)
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--cfg', type=str, default=None)

    args = parser.parse_args()
    num_workers = args.num_workers

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr
    
    if args.cfg is None:
        raise ValueError('Please specify a config file')

    hp = HParams(args.cfg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    adaptnet = hp.conversion_mode == 1
    conversion = (not hp.conversion_mode == 2)
    hp.conversion = conversion

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        adaptnet=adaptnet,
                        mode=hp.voc_mode).to(device)

    print(voc_model)
    trainable_params = list(voc_model.parameters())

    paths = Paths(hp.data_path, hp.voc_model_id, '')

    # Load pase model
    print('Building PASE...')
    if hp.pase_cfg is not None:
        # 2 PASEs: (1) Identifier extractor, (2) Content extractor
        pase_cntnt = wf_builder(hp.pase_cfg)
        if hp.pase_ckpt is not None:
            pase_cntnt.load_pretrained(hp.pase_ckpt, load_last=True, verbose=True)
        pase_cntnt.to(device)
        if conversion:
            pase_id = wf_builder(hp.pase_cfg)
            if hp.pase_ckpt is not None:
                pase_id.load_pretrained(hp.pase_ckpt, load_last=True, verbose=True)
            pase_id.to(device)
        if hp.pase_cntnt_ft:
            print('Setting Content PASE in TRAIN mode')
            pase_cntnt.train()
            # assign a saver to the model
            pase_cntnt.saver = Saver(pase_cntnt, paths.voc_checkpoints,
                                     prefix='PASE_cntnt')
            trainable_params += list(pase_cntnt.parameters())
        else:
            print('Setting Content PASE in EVAL mode')
            pase_cntnt.eval()
        hp.pase_cntnt = pase_cntnt
        if conversion:
            if hp.pase_id_ft:
                print('Setting ID PASE in TRAIN mode')
                pase_id.train()
                pase_id.saver = Saver(pase_id, paths.voc_checkpoints,
                                      prefix='PASE_id')
                trainable_params += list(pase_id.parameters())
            else:
                print('Setting ID PASE in EVAL mode')
                pase_id.eval()
            hp.pase_id = pase_id
    else:
        hp.pase_id = hp.pase_cntnt = None
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length



    voc_model.restore(paths.voc_latest_weights)

    optimiser = optim.Adam(trainable_params)
    #optimiser = radam.RAdam(trainable_params)

    if hp.distortions_cfg is not None:
        # Build distortion pipeline
        with open(hp.distortions_cfg, 'r') as dtr_cfg:
            dtr = json.load(dtr_cfg)
            trans = config_distortions(**dtr)
            print(trans)
    else:
        trans = None

    if hasattr(hp, 'spk2split'):
        with open(hp.spk2split, 'r') as f:
            spk2split = json.load(f)
    else:
        spk2split = None


    train_set, test_set, valid_set = get_vocoder_datasets(paths.data, batch_size,
                                                          train_gta,
                                                          num_workers=num_workers,
                                                          transforms=trans,
                                                          spk2split=spk2split)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    writer = SummaryWriter(paths.voc_checkpoints)
    hp.writer = writer

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(voc_model, loss_func, optimiser, train_set, valid_set, test_set, lr,
                   total_steps, device, hp)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')
