"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<params>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
from torch.autograd import Variable
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams

torch.set_num_threads(1)
# use_cuda = torch.cuda.is_available()
use_cuda = False


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    if use_cuda:
        model = model.cuda()
    model.eval()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
        assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = Variable(torch.FloatTensor(c.T).unsqueeze(0))

    # if initial_value is None:
    #     if is_mulaw_quantize(hparams.input_type):
    #         initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
    #     else:
    #         initial_value = 0.0
    #
    # if is_mulaw_quantize(hparams.input_type):
    #     assert 0 <= initial_value < hparams.quantize_channels
    #     initial_input = np_utils.to_categorical(
    #         initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
    #     initial_input = Variable(torch.from_numpy(initial_input)).view(
    #         1, 1, hparams.quantize_channels)
    # else:
    #     initial_input = Variable(torch.zeros(1, 1, 1)).fill_(initial_value)
    #
    # g = None if g is None else Variable(torch.LongTensor([g]))
    # if use_cuda:
    #     initial_input = initial_input.cuda()
    #     g = None if g is None else g.cuda()
    #     c = None if c is None else c.cuda()

    # y_hat = model.incremental_forward(
    #     initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
    #     log_scale_min=hparams.log_scale_min)

    with torch.no_grad():
        y_student, _, _ = model(None, c, g, softmax=False, use_cuda=use_cuda)
    y_student = y_student.view(-1).cpu().data.numpy()
    return y_student

    # if is_mulaw_quantize(hparams.input_type):
    #     y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
    #     y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    # elif is_mulaw(hparams.input_type):
    #     y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    # else:
    #     y_hat = y_hat.view(-1).cpu().data.numpy()


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
    conditional_path = args["--conditional"]
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker-id"]
    speaker_id = None if speaker_id is None else int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
    else:
        c = None

    from train_student import build_model

    # Model
    model = build_model(name="student")
    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    checkpoint_name = splitext(basename(checkpoint_path))[0]

    wav_id = conditional_path.split("/")[-1].split(".")[0].split("-")[-1]
    dataset_name = conditional_path.split("/")[-1].split(".")[0].split("-")[0]
    save_dir = join(dst_dir, checkpoint_name + "_student", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    dst_wav_path = join(save_dir, "{}{}.wav".format(wav_id, file_name_suffix))

    # DO generate
    waveform = wavegen(model, length, c=c, g=speaker_id)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
