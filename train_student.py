import argparse

import sys

import os
from os.path import dirname, join, expanduser
from tqdm import tqdm  # , trange
from datetime import datetime
import random

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavenet_vocoder import builder
import lrschedule

import torch
from torch.utils import data as data_utils
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import librosa.display

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from wavenet_vocoder.mixture import discretized_mix_logistic_loss,discretized_mix_gaussian_loss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic,sample_from_discretized_gaussian

import audio
from hparams import hparams, hparams_debug_string

fs = hparams.sample_rate

global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Trainining script for Student WaveNet vocoder")
    parser.add_argument('--data_root', type=str, default=None, help='Directory contains preprocessed features.')
    parser.add_argument('--checkpoint_dir', type=str, default='student_checkpoints',
                        help='Directory where to save model checkpoints')
    parser.add_argument('--hparams', type=str, default='', help='Hyper parameters')
    parser.add_argument('--preset', type=str, default='./presets/ljspeech_gaussian.json', help='Path of preset parameters (json)')
    parser.add_argument('--checkpoint_teacher', type=str, default='./checkpoints/checkpoint_step000405000_ema.pth',
                        help='Restore teacher model from checkpoint path must given.')
    parser.add_argument('--checkpoint_student', type=str,
                        #default='./student_checkpoints/student/checkpoint_step000003000.pth',
                        default=None,
                        help='Restore student model from checkpoint path if given.')
    parser.add_argument('--restore_parts', type=str, default=None, help='Restore part of the model.')
    parser.add_argument('--log_event_path', type=str, default='log/gaussian', help='Log event path.')
    parser.add_argument('--reset_optimizer', type=str, default=None, help='Reset optimizer.')
    parser.add_argument('--speaker_id', type=int, default=None,
                        help='Use specific speaker of data in case for multi-speaker datasets.')
    args = parser.parse_args()
    return args


def sanity_check(model, c, g):
    if model.has_speaker_embedding():
        if g is None:
            raise RuntimeError(
                "WaveNet expects speaker embedding, but speaker-id is not provided")
    else:
        if g is not None:
            raise RuntimeError(
                "WaveNet expects no speaker embedding, but speaker-id is provided")

    if model.local_conditioning_enabled():
        if c is None:
            raise RuntimeError("WaveNet expects conditional features, but not given")
    else:
        if c is not None:
            raise RuntimeError("WaveNet expects no conditional features, but given")


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def interest_indices(self, paths):
        indices = np.arange(len(paths))
        if self.test_size is None:
            test_size = self.test_num_samples / len(paths)
        else:
            test_size = self.test_size
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=self.random_state)
        return train_indices if self.train else test_indices

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # Filter by train/tset
                indices = self.interest_indices(paths)
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        lengths_np = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids)[indices])
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)

        return paths

    def collect_features(self, path):
        return np.load(path)


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 0, **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand.requires_grad = False
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(model, ema, name_,hparams):
    assert ema is not None
    averaged_model = build_model(hparams,name_)
    if use_cuda:
        averaged_model = averaged_model.cuda()
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def get_power_loss(y, y1, frame_length=1024, hop_length=256):
    batch = y.size(0)
    x = y.view(batch, -1)
    x1 = y1.view(batch, -1)
    window = torch.hann_window(frame_length, periodic=True)
    if use_cuda:
        window = window.cuda()
    s = torch.stft(x, frame_length=frame_length, hop=hop_length, window=window)
    s1 = torch.stft(x1, frame_length=frame_length, hop=hop_length, window=window)
    ss = torch.log(torch.sqrt(torch.sum(s ** 2, -1) + 1e-5)) - torch.log(torch.sqrt(torch.sum(s1 ** 2, -1) + 1e-5))
    return torch.sum(ss ** 2) / batch


def get_power_loss_v1(y, y1, frame_length=1024, hop_length=256):
    batch = y.size(0)
    x = y.view(batch, -1)
    x1 = y1.view(batch, -1)
    window = torch.hann_window(frame_length, periodic=True)
    if use_cuda:
        window = window.cuda()
    s = torch.stft(x, frame_length=frame_length, hop=hop_length, fft_size=1024, window=window)
    s1 = torch.stft(x1, frame_length=frame_length, hop=hop_length, fft_size=1024, window=window)

    s_sqrt = torch.sqrt(torch.sum(s ** 2, -1))
    s1_sqrt = torch.sqrt(torch.sum(s1 ** 2, -1))
    ss = s_sqrt -s1_sqrt
    return torch.sum(ss ** 2) / (batch*(frame_length/2+1))


class KLDivLoss(nn.Module):
    def __init__(self,lambda_=4):
        super(KLDivLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, y_hat, mu_q, scale_q, mask, sample_T=32):
        if hparams.output_type == 'Gaussian':
            # teacher p,student q
            mu_p, scale_p = y_hat[:, :1, :], torch.exp(y_hat[:, 1:, :])
            loss = torch.log(scale_p / scale_q) + (scale_q ** 2 - scale_p ** 2 + (mu_q - mu_p) ** 2) / (2 * scale_p ** 2)
            # loss += torch.log(scale_q / scale_p) + (scale_p ** 2 - scale_q ** 2 + (mu_q - mu_p) ** 2) / (2 * scale_q ** 2)
            # loss /= 2
            loss += self.lambda_*(torch.log(scale_p)-torch.log(scale_q))**2
            kl_loss = torch.sum(loss[:,:,:-1] * mask.permute(0,2,1)) / mask.sum()
            return kl_loss
        elif hparams.output_type == "MOL":
            h_pt_ps = 0
            for i in range(sample_T):
                u = torch.zeros(mu_q.size()).uniform_(1e-5, 1 - 1e-5)
                if use_cuda:
                    u = u.cuda()
                z = torch.log(u) - torch.log(1 - u)
                student_predict = mu_q + z * scale_q
                assert student_predict.requires_grad is True

                student_predict = student_predict.permute(0, 2, 1)
                teacher_log_p = discretized_mix_logistic_loss(y_hat[:, :, :-1], student_predict[:, 1:, :], reduce=False)
                h_pt_ps += torch.sum(teacher_log_p * mask) / mask.sum()

            # compute h_ps
            a = scale_q.permute(0, 2, 1)
            h_ps = torch.sum((torch.log(a[:, 1:, :]) + 2) * mask) / (mask.sum())

            # compute kl loss
            cross_entropy = h_pt_ps / sample_T
            kl_loss = cross_entropy - h_ps
            return kl_loss


class PowerLoss(nn.Module):
    def __init__(self, power_loss_fn=get_power_loss_v1):
        super(PowerLoss, self).__init__()
        self.loss_fn = power_loss_fn

    def forward(self, x, predict):
        power_loss_tot = 0
        #power_loss_tot += self.loss_fn(predict, x, frame_length=128, hop_length=32)
        #power_loss_tot += self.loss_fn(predict, x, frame_length=256, hop_length=64)
        #power_loss_tot += self.loss_fn(predict, x, frame_length=512, hop_length=128)
        #power_loss_tot += self.loss_fn(predict, x, frame_length=1024, hop_length=256)
        power_loss_tot += self.loss_fn(predict, x, frame_length=2048, hop_length=512)
        return power_loss_tot 


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.gin_channels > 0

    # To save GPU memory... I don't want to do this though
    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None

    # Time resolution adjustment
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            if hparams.upsample_conditional_features:
                assert_ready_for_upsampling(x, c)
                if max_time_steps is not None:
                    max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                    if len(x) > max_steps:
                        max_time_frames = max_steps // audio.get_hop_size()
                        s = np.random.randint(0, len(c) - max_time_frames)
                        # print("Size of file=%6d, t_offset=%6d"  % (len(c), s,))
                        ts = s * audio.get_hop_size()
                        x = x[ts:ts + audio.get_hop_size() * max_time_frames]
                        c = c[s:s + max_time_frames, :]
                        assert_ready_for_upsampling(x, c)
            else:
                x, c = audio.adjust_time_resolution(x, c)
                if max_time_steps is not None and len(x) > max_time_steps:
                    s = np.random.randint(0, len(x) - max_time_steps)
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                assert len(x) == len(c)
            new_batch.append((x, c, g))
        batch = new_batch
    else:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            x = audio.trim(x)
            if max_time_steps is not None and len(x) > max_time_steps:
                s = np.random.randint(0, len(x) - max_time_steps)
                if local_conditioning:
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                else:
                    x = x[s:s + max_time_steps]
            new_batch.append((x, c, g))
        batch = new_batch

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        x_batch = np.array([_pad_2d(np_utils.to_categorical(
            x[0], num_classes=hparams.quantize_channels),
            max_input_len) for x in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, c_batch, g_batch, input_lengths


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_waveplot(path, y_teacher, y_target, y_student,writer):

    sr = hparams.sample_rate
    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.title('target')
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(3, 1, 2)
    plt.title('teacher')
    librosa.display.waveplot(y_teacher, sr=sr)
    plt.subplot(3, 1, 3)
    plt.title('student')
    librosa.display.waveplot(y_student, sr=sr)
    plt.tight_layout()
    plt.savefig(path, format="png")
    if writer:
        import io
        from PIL import Image
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        plt.close()
        buff.seek(0)
        im = np.array(Image.open(buff))
        writer.add_image('image', im)
    plt.close()


def eval_model(global_step, writer, teacher_model, student_model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        student_model = clone_as_averaged_model(student_model, ema, name_=hparams.name,hparams=hparams)

    student_model.eval()
    teacher_model.eval()
    idx = np.random.randint(0, len(y))

    length = input_lengths[idx].data.cpu().numpy()
    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        initial_value = P.mulaw(0.0, hparams.quantize_channels)
    else:
        initial_value = 0.0
    print("Intial value:", initial_value)

    # (C,)
    if is_mulaw_quantize(hparams.input_type):
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
    initial_input = initial_input.cuda() if use_cuda else initial_input

    # Run the model in fast eval mode
    with torch.no_grad():
        y_hat = teacher_model.incremental_forward(
            initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y_target = P.inv_mulaw_quantize(y_target, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
        y_target = P.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    z = np.random.logistic(0, 1, y_target.shape)
    z = torch.from_numpy(z).view(1, 1, -1).float()
    if use_cuda:
        z = z.cuda()
    with torch.no_grad():
        predict_list,y_student, _, _ = student_model(z, c=c, g=g, softmax=False, use_scale=hparams.use_scale)
    y_student = y_student.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = join(eval_dir, "step{:09d}_teacher_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(eval_dir, "step{:09d}_student_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_student, sr=hparams.sample_rate)
    path = join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)

    # save figure
    path = join(eval_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_student=y_student, y_target=y_target, y_teacher=y_hat,writer=writer)


def save_states(global_step, writer, y_hat, y, y_student,scale_tot, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().numpy()

    # (B, C, T)
    if y_hat.dim() == 4:
        y_hat = y_hat.squeeze(-1)

    if is_mulaw_quantize(hparams.input_type):
        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(1)[1]

        # (T,)
        y_hat = y_hat[idx].data.cpu().long().numpy()
        y = y[idx].view(-1).data.cpu().long().numpy()

        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y = P.inv_mulaw_quantize(y, hparams.quantize_channels)
    else:
        # (B, T)
        scale = y_hat[:,1:,:]
        teacher_log_scale = scale.data.cpu().numpy()
        student_log_scale = torch.log(scale_tot).data.cpu().numpy()
        writer.add_histogram('log_teacher_scale', teacher_log_scale, global_step)
        writer.add_histogram('log_student_scale', student_log_scale, global_step)
        y_hat = sample_from_discretized_gaussian(
            y_hat, log_scale_min=hparams.log_scale_min)

        # (T,)
        y_hat = y_hat[idx].view(-1).data.cpu().numpy()
        y = y[idx].view(-1).data.cpu().numpy()

        if is_mulaw(hparams.input_type):
            y_hat = P.inv_mulaw(y_hat, hparams.quantize_channels)
            y = P.inv_mulaw(y, hparams.quantize_channels)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0

    y_student = y_student[idx].view(-1).data.cpu().numpy()
    y_student[length:] = 0

    # Save audio
    audio_dir = join(checkpoint_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_teacher_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_student_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_student, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}.jpg".format(global_step))
    save_waveplot(path,y_teacher=y_hat,y_student=y_student,y_target=y,writer=writer)

def __train_step(phase, epoch, global_step, global_test_step,
                 teacher_model, student_model, kl_criterion, pl_criterion, optimizer, writer,
                 x, y, c, g, input_lengths,
                 checkpoint_dir, eval_dir=None, do_eval=False, ema=None):
    sanity_check(teacher_model, c, g)
    sanity_check(student_model, c, g)

    # x : (B, C, T)
    # y : (B, T, 1)
    # c : (B, C, T)
    # g : (B,)
    train = (phase == "train")
    clip_thresh = hparams.clip_thresh
    teacher_model.eval()
    if train:
        student_model.train()
        student_model.upsample_conv.eval()
        step = global_step
    else:
        student_model.eval()
        step = global_test_step

    # Learning rate schedule
    current_lr = hparams.initial_learning_rate
    if train and hparams.lr_schedule is not None:
        lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
        current_lr = lr_schedule_f(
            hparams.initial_learning_rate, step, **hparams.lr_schedule_kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    optimizer.zero_grad()

    # Prepare data
    c = c if c is not None else None
    g = g if g is not None else None
    if use_cuda:
        x, y = x.cuda(), y.cuda()
        input_lengths = input_lengths.cuda()
        c = c.cuda() if c is not None else None
        g = g.cuda() if g is not None else None



    # Apply model: Run the model in regular eval mode
    # NOTE: softmax is handled in F.cross_entrypy_loss
    # y_hat: (B x C x T)

    # get mu and scale from student model
    if hparams.output_type=="MOL":
        u = torch.zeros(x.size()).uniform_(1e-5, 1 - 1e-5)
        if use_cuda:
            u = u.cuda()
        z = torch.log(u) - torch.log(1 - u)
    else:
        z = torch.randn(x.size())
    predict_list, predict, mu_tot, scale_tot = torch.nn.parallel.data_parallel(student_model, (
        z, c, g, False, True, hparams.use_scale))

    y_hat = torch.nn.parallel.data_parallel(teacher_model, (predict, c, g, False))

    # (B, T, 1)
    mask = sequence_mask(input_lengths, max_len=x.size(-1)).unsqueeze(-1)
    if hparams.iaf_shift:
        iaf_length = len(student_model.iaf_layers_size)
        mask = mask[:, 1 + iaf_length:, :]
        x = x[:, :, iaf_length + 1:]
    else:
        mask = mask[:, 1:, :]
        x = x[:, :, 1:]
    kl_loss = kl_criterion(y_hat, mu_tot,scale_tot, mask)
    power_loss = pl_criterion(x, predict[:,:,:-1])
    loss = kl_loss + power_loss

    if train and step > 0 and step % hparams.checkpoint_interval == 0:
        save_states(step, writer, y_hat, y, predict,scale_tot, input_lengths, checkpoint_dir)
        save_checkpoint(student_model, optimizer, step, checkpoint_dir, epoch, ema)
    if train and step > 0 and step % 200 == 0:
        save_states(step, writer, y_hat, y,predict,scale_tot, input_lengths, checkpoint_dir)

    if do_eval:
        # NOTE: use train step (i.e., global_step) for filename
        eval_model(global_step, writer, teacher_model, student_model, y, c, g, input_lengths, eval_dir, ema)

    # Update
    if train:
        loss.backward()
        if clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm(student_model.parameters(), clip_thresh)
        optimizer.step()
        # update moving average
        if ema is not None:
            for name, param in student_model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

    # Logs
    writer.add_scalar("{} loss".format(phase), float(loss.data), step)

    writer.add_scalar('{} kl loss'.format(phase), float(kl_loss.data), step)
    writer.add_scalar('{} power loss'.format(phase), float(power_loss.data), step)
    if train:
        if clip_thresh > 0:
            writer.add_scalar("gradient norm", grad_norm, step)
        writer.add_scalar("learning rate", current_lr, step)
    # print(type(loss.data), loss.data)
    return float(loss.data), float(kl_loss.data), float(power_loss.data)


def train_loop(teacher_model, student_model, data_loaders, optimizer, writer, checkpoint_dir=None):
    if use_cuda:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    if hparams.exponential_moving_average:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None
    kl_criterion = KLDivLoss()
    pl_criterion = PowerLoss()
    global global_step, global_epoch, global_test_step
    while global_epoch < hparams.nepochs:
        for phase, data_loader in data_loaders.items():
            train = (phase == "train")
            running_loss = 0.
            running_kl_loss = 0.
            running_power_loss = 0.

            test_evaluated = False
            for step, (x, y, c, g, input_lengths) in tqdm(enumerate(data_loader)):
                # Whether to save eval (i.e., online decoding) result
                do_eval = False
                eval_dir = join(checkpoint_dir, "{}_eval".format(phase))
                # Do eval per eval_interval for train
                if train and global_step > 0 \
                        and global_step % hparams.train_eval_interval == 0:
                    do_eval = True
                # Do eval for test
                # NOTE: Decoding WaveNet is quite time consuming, so
                # do only once in a single epoch for testset
                if not train and not test_evaluated \
                        and global_epoch % hparams.test_eval_epoch_interval == 0:
                    do_eval = True
                    test_evaluated = True
                if do_eval:
                    print("[{}] Eval at train step {}".format(phase, global_step))

                # Do step
                loss, kl_loss, power_loss = __train_step(
                    phase, global_epoch, global_step, global_test_step, teacher_model, student_model,
                    kl_criterion, pl_criterion,
                    optimizer, writer, x, y, c, g, input_lengths,
                    checkpoint_dir, eval_dir, do_eval, ema)

                running_loss += loss
                running_kl_loss += kl_loss
                running_power_loss += power_loss

                # update global state
                if train:
                    global_step += 1
                else:
                    global_test_step += 1

            # log per epoch
            averaged_loss = running_loss / len(data_loader)
            averaged_kl_loss = running_kl_loss / len(data_loader)
            averaged_power_loss = running_power_loss / len(data_loader)

            writer.add_scalar("{} loss (per epoch)".format(phase), averaged_loss, global_epoch)
            writer.add_scalar("{} kl loss (per epoch)".format(phase), averaged_kl_loss, global_epoch)
            writer.add_scalar("{} power loss (per epoch)".format(phase), averaged_power_loss, global_epoch)

            print("Step {} [{}] Loss: {} KL Loss: {} Power Loss: {}".format(global_step, phase, averaged_loss,
                                                                            averaged_kl_loss, averaged_power_loss))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema, name_=hparams.name,hparams=hparams)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)


def build_model(hparams,name=None):
    assert name is not None

    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)
    if name == "teacher":
        model = getattr(builder, "wavenet")(
            out_channels=hparams.out_channels,
            layers=hparams.layers,
            stacks=hparams.stacks,
            residual_channels=hparams.residual_channels,
            gate_channels=hparams.gate_channels,
            skip_out_channels=hparams.skip_out_channels,
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            weight_normalization=hparams.weight_normalization,
            n_speakers=hparams.n_speakers,
            dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
            freq_axis_kernel_size=hparams.freq_axis_kernel_size,
            scalar_input=is_scalar_input(hparams.input_type),
        )
    elif name == "parallel":
        model = getattr(builder, "student_wavenet")(
            out_channels=hparams.student_out_channels,
            layers=hparams.student_layers,
            stacks=hparams.student_stacks,
            residual_channels=hparams.student_residual_channels,
            iaf_layer_sizes=hparams.iaf_layer_sizes,
            gate_channels=hparams.student_gate_channels,
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            weight_normalization=hparams.weight_normalization,
            n_speakers=hparams.n_speakers,
            dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
            freq_axis_kernel_size=hparams.freq_axis_kernel_size,
            scalar_input=is_scalar_input(hparams.input_type),
        )
    elif name == "clari":
        model = getattr(builder, "clari_wavenet")(
            out_channels=hparams.student_out_channels,
            layers=hparams.student_layers,
            stacks=hparams.student_stacks,
            residual_channels=hparams.student_residual_channels,
            iaf_layer_sizes=hparams.iaf_layer_sizes,
            gate_channels=hparams.student_gate_channels,
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            weight_normalization=hparams.weight_normalization,
            n_speakers=hparams.n_speakers,
            dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
            freq_axis_kernel_size=hparams.freq_axis_kernel_size,
            scalar_input=is_scalar_input(hparams.input_type),
            use_skip=hparams.use_skip,
            iaf_shift=hparams.iaf_shift
        )
    else:
        raise Exception("No such model")
    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))


def get_data_loaders(data_root, speaker_id, test_shuffle=True):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0
    for phase in ["train", "test"]:
        train = phase == "train"
        X = FileSourceDataset(RawAudioDataSource(data_root, speaker_id=speaker_id,
                                                 train=train,
                                                 test_size=hparams.test_size,
                                                 test_num_samples=hparams.test_num_samples,
                                                 random_state=hparams.random_state))
        if local_conditioning:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size,
                                                      test_num_samples=hparams.test_num_samples,
                                                      random_state=hparams.random_state))
            assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchDataset(X, Mel)
        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size,
            num_workers=hparams.num_workers, sampler=sampler, shuffle=shuffle,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

        speaker_ids = {}
        for idx, (x, c, g) in enumerate(dataset):
            if g is not None:
                try:
                    speaker_ids[g] += 1
                except KeyError:
                    speaker_ids[g] = 1
        if len(speaker_ids) > 0:
            print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders


if __name__ == "__main__":
    args = get_args()
    print("Command line args:\n", args)
    checkpoint_dir = join(args.checkpoint_dir, "student")
    checkpoint_teacher_path = args.checkpoint_teacher
    checkpoint_student_path = args.checkpoint_student
    checkpoint_restore_parts = args.restore_parts
    speaker_id = args.speaker_id
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args.preset

    data_root = args.data_root
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = join(args.log_event_path, "student")
    reset_optimizer = args.reset_optimizer

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    assert hparams.name == "clari"
    print(hparams_debug_string())

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader setup
    data_loaders = get_data_loaders(data_root, speaker_id, test_shuffle=True)

    # Model
    teacher_model = build_model(hparams,name="teacher")
    student_model = build_model(hparams,name='clari')
    if hparams.share_condition_net:
        student_model.load_teacher_upsample_conv(teacher_model)
    print("*" * 50, "==> This is Teacher Model <==", "*" * 50)
    print(teacher_model)
    print("*" * 50, "==> This is Student Model <==", "*" * 50)
    print(student_model)
    if use_cuda:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    receptive_field = teacher_model.receptive_field
    print("Receptive field (samples / ms): {} / {}".format(
        receptive_field, receptive_field / fs * 1000))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()),
                           lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, hparams.adam_beta2),
                           eps=hparams.adam_eps, weight_decay=hparams.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10 * 1000, gamma=0.5)
    # restore teacher
    assert checkpoint_teacher_path is not None
    if checkpoint_teacher_path is not None:
        restore_parts(checkpoint_teacher_path, teacher_model)

    for param in teacher_model.parameters():
        param.requires_grad = False

    # restore student
    if checkpoint_student_path is not None:
        load_checkpoint(checkpoint_student_path, student_model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train_loop(teacher_model, student_model, data_loaders, optimizer, writer, checkpoint_dir=checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        save_checkpoint(
            student_model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")

    sys.exit(0)
