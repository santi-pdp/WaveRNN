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
from utils.dataset import *
from utils.distribution import discretized_mix_logistic_loss
#import hparams as hp
from models.fatchord_version import WaveRNN, PASEInjector
from gen_wavernn import gen_genh_testset
from utils.paths import *
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

        for i, (x, y, xm, xlm) in enumerate(train_set, 1):
            x, y, xm, xlm = x.to(device), y.to(device), xm.to(device), xlm.to(device)
            xm = xm.unsqueeze(1)
            xlm = xlm.unsqueeze(1)

            if hp.pase_ft:
                m = hp.pase(xm, xlm)
            else:
                with torch.no_grad():
                    m = hp.pase(xm, xlm)
            if hp.pase_lambda > 0:
                raise NotImplementedError
                # use an MSE loss weighted with pase_lamda
                # that tights the distorted PASE output
                # to the clean PASE soft-labels (loaded in m)
                pase_reg_loss = hp.pase_lambda * F.mse_loss(m, m_clean)

            y_hat = model(x, m)

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
                gen_genh_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                                 hp.voc_target, hp.voc_overlap, paths.voc_output,
                                 hp=hp, device=device)
                model.checkpoint(paths.voc_checkpoints)
                if hp.pase_ft:
                    hp.pase.train()
                    hp.pase.save(paths.voc_checkpoints, step)

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

        for i, (x, y, xm, xlm) in enumerate(eval_set, 1):
            x, y, xm, xlm = x.to(device), y.to(device), xm.to(device), xlm.to(device)
            xm = xm.unsqueeze(1)
            xlm = xlm.unsqueeze(1)

            m = hp.pase(xm, xlm)
            if hp.pase_lambda > 0:
                raise NotImplementedError
                # use an MSE loss weighted with pase_lamda
                # that tights the distorted PASE output
                # to the clean PASE soft-labels (loaded in m)
                pase_reg_loss = hp.pase_lambda * F.mse_loss(m, m_clean)

            y_hat = model(x, m)

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
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--cfg', type=str, default=None)

    args = parser.parse_args()
    num_workers = args.num_workers

    batch_size = args.batch_size
    force_train = args.force_train
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

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
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

    print(voc_model)
    trainable_params = list(voc_model.parameters())

    paths = GEnhancementPaths(hp.voc_model_id)

    # Load pase model
    print('Building PASE...')
    if hp.pase_cfg is not None:
        pase = PASEInjector(hp.pase_cfg, hp.pase_ckpt, hp.pase_ft,
                            hp.num_mels, hp.pase_feats,
                            paths.voc_checkpoints, global_mode=hp.global_pase)
        pase.to(device)
        if hp.pase_ft:
            print('Setting PASE in TRAIN mode')
            trainable_params += list(pase.parameters())
        else:
            print('Setting PASE in EVAL mode')
            pase.eval()
        hp.pase = pase
    else:
        raise ValueError
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    #voc_model.restore(paths.voc_latest_weights)
    #print('WARNING: PASE is not restored as in WaveRNN')

    optimiser = optim.Adam(trainable_params)

    train_set, valid_set, test_set = get_genh_datasets(hp.train_clean, 
                                                       hp.train_noisy,
                                                       hp.valid_clean,
                                                       hp.valid_noisy,
                                                       hp.test_clean,
                                                       hp.test_noisy,
                                                       batch_size,
                                                       hparams=hp,
                                                       num_workers=num_workers)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len)])

    writer = SummaryWriter(paths.voc_checkpoints)
    hp.writer = writer

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(voc_model, loss_func, optimiser, train_set, valid_set, test_set, lr,
                   total_steps, device, hp)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')
