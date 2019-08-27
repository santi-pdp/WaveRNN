
# CONFIG -----------------------------------------------------------------------------------------------------------#

#LIBRITTS_ROOT='LibriTTS_calcula'
LIBRITTS_ROOT='LibriTTS'

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
#wav_path = 'data/vctk/all_trimmed_wav16'
#wav_path = 'data/LibriTTS/all_train_11s-sel_16k'
wav_path = 'data/{}/all_train_clean_60s-sel_16k'.format(LIBRITTS_ROOT)
#data_path = 'data/vctk/all_trimmed_paseQRNN'
#data_path = 'data/LibriTTS/all_train_11s-sel_paseQRNN'
data_path = 'data/{}/all_train_clean_60s-sel_paseQRNN'.format(LIBRITTS_ROOT)
#data_path = 'data/LibriTTS_calcula/all_train_11s-sel_paseQRNN'
num_workers = 30

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
#voc_model_id = 'pase-eval_enhancement_libritts_raw'
#voc_model_id = 'pase-eval-FT_enhancement_libritts_raw'
#voc_model_id = 'pase-eval_enhancement-addrev_libritts_raw'
voc_model_id = 'pase-eval_clean_libritts_raw'
tts_model_id = 'ljspeech_lsa_smooth_attention'

# set this to True if you are only interested in WaveRNN
ignore_tts = True

# distortions configuration file
distortions_cfg = None
#distortions_cfg = 'addrev.cfg'
# ALL distortions active
#distortions_cfg = 'distortions.cfg'
#distortions_cfg = 'distortions_calcula.cfg'

# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
#sample_rate = 22050
sample_rate = 16000
n_fft = 2048
fft_bins = n_fft // 2 + 1
#num_mels = 80
num_mels = 100
#pase_cfg = 'pase_models/QRNN/PASE_dense_QRNN.cfg'
#pase_ckpt = 'pase_models/QRNN/PASE_dense_QRNN.ckpt'
pase_cfg = 'pase_models/QRNN_LibriTTS_5reg/PASE_dense_QRNN.cfg'
pase_ckpt = 'pase_models/QRNN_LibriTTS_5reg/FE_e4.ckpt'
pase_id_ft = False
#pase_cntnt_ft = False
pase_cntnt_ft = True
cuda = True
hop_length = 160 
win_length = 640
#hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
#win_length = 1100                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
#voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
#voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_upsample_factors = (4, 4, 10)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 32
voc_lr = 1e-4
voc_write_every = 1_000
#voc_checkpoint_every = 100
voc_checkpoint_every = 10_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 1_000_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_r = 1                           # model predicts r frames per output step
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']

# Training


tts_schedule = [(7,  1e-3,  10_000,  32),   # progressive training schedule
                (5,  1e-4, 100_000,  32),   # (r, lr, step, batch_size)
                (2,  1e-4, 180_000,  16),
                (1,  1e-4, 350_000,  8)]

tts_max_mel_len = 1250              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes


conversion_mode = 2 # 0: arithmetic at input, 1: adaptnet, 2: none

# ------------------------------------------------------------------------------------------------------------------#

