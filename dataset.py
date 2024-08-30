import random
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from speechbrain.inference.speaker import EncoderClassifier
from tqdm.auto import tqdm

MAX_WAV_VALUE = 32768.0


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    if len(audio.shape) > 1:
        audio = audio.squeeze()

    if len(audio.shape) != 1:
        raise ValueError("Audio should be a one-dimensional array, but got shape: {}".format(audio.shape))

    f0s = []
    audio = audio.astype(np.float64)
    y_pad = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)
    signal = basic.SignalObj(y_pad, rate)
    pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                    'tda_frame_length': 25.0})
    if interp:
        f0s.append(pitch.samp_interp[None, None, :])
    else:
        f0s.append(pitch.samp_values[None, None, :])

    f0 = np.vstack(f0s)
    return f0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='float64')
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def parse_manifest(manifest):
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                elif 'hubert' in sample:
                    k = 'hubert'
                else:
                    k = 'codes'

                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes


def get_dataset_filelist(h):
    training_files, training_codes = parse_manifest(h.input_training_file)
    validation_files, validation_codes = parse_manifest(h.input_validation_file)

    return (training_files, training_codes), (validation_files, validation_codes)


def parse_speaker(path, method):
    if type(path) == str:
        path = Path(path)

    if method == 'parent_name':
        return path.parent.name
    elif method == 'parent_parent_name':
        return path.parent.parent.name
    elif method == '_':
        return path.name.split('_')[0]
    elif method == 'single':
        return 'A'
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, f0=None, multispkr=False, pad=None,
                 f0_stats=None, f0_normalize=False, f0_feats=False, f0_median=False,
                 f0_interp=False, vqvae=False):
        self.audio_files, self.codes = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0 = f0
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = f0_stats
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        self.multispkr = multispkr
        self.pad = pad

        if f0_stats is None:
            self.f0_stats = self._compute_f0_stats()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if multispkr:
            self.spek_emb = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                           savedir="pretrained_models/spkrec-xvect-voxceleb",
                                                           run_opts={"device": device})

    def _compute_f0_stats(self):
        """Compute global f0 mean and std from the dataset."""
        f0_values = []
        for filename in tqdm(self.audio_files):
            audio, _ = load_audio(filename)
            f0 = get_yaapt_f0(audio, rate=self.sampling_rate, interp=self.f0_interp)
            f0_values.append(f0[f0 > 0])

        f0_values = np.concatenate(f0_values)
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)

        return {'f0_mean': f0_mean, 'f0_std': f0_std}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        if self.vqvae:
            code_length = audio.shape[0] // self.code_hop_size
        else:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
            code = self.codes[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]
        assert self.vqvae or audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            if not self.vqvae:
                code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        if self.vqvae:
            audio = self._sample_interval([audio])[0]
        else:
            audio, code = self._sample_interval([audio, code])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        if self.vqvae:
            feats = {
                "code": audio.view(1, -1).numpy()
            }
        else:
            feats = {"code": code.squeeze()}

        if self.f0:
            try:
                f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
            except:
                f0 = np.zeros((1, 1, audio.shape[-1] // 80))
            f0 = f0.astype(np.float32)
            feats['f0'] = f0.squeeze(0)

        if self.multispkr:
            feats['spkr'] = self._get_spkr_embedding(audio)

        if self.f0_normalize:
            mean = self.f0_stats['f0_mean']
            std = self.f0_stats['f0_std']
            ii = feats['f0'] != 0

            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
                feats['f0'][~ii] = (feats['f0'][~ii] - mean) / std

            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spkr_embedding(self, audio):
        embeddings = self.spek_emb.encode_batch(audio).squeeze()
        return embeddings.detach().cpu().numpy()

    def __len__(self):
        return len(self.audio_files)


class F0Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate,
                 split=True, n_cache_reuse=1, device=None, multispkr=False,
                 pad=None, f0_stats=None, f0_normalize=False, f0_feats=False,
                 f0_median=False, f0_interp=False, vqvae=False):
        self.audio_files, _ = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = None
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        self.pad = pad
        self.multispkr = multispkr

        if f0_stats is not None:
            self.f0_stats = torch.load(f0_stats)
        else:
            self.f0_stats = self._compute_f0_stats()

    def _compute_f0_stats(self):
        """Compute global f0 mean and std from the dataset."""
        f0_values = []
        for filename in tqdm(self.audio_files):
            audio, _ = load_audio(filename)
            f0 = get_yaapt_f0(audio, rate=self.sampling_rate, interp=self.f0_interp)
            f0_values.append(f0[f0 > 0])

        f0_values = np.concatenate(f0_values)
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)

        return {'f0_mean': f0_mean, 'f0_std': f0_std}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        audio = self._sample_interval([audio])[0]

        feats = {}
        try:
            f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
        except:
            f0 = np.zeros((1, 1, audio.shape[-1] // 80))
        f0 = f0.astype(np.float32)
        feats['f0'] = f0.squeeze(0)

        if self.f0_normalize:
            mean = self.f0_stats['f0_mean']
            std = self.f0_stats['f0_std']
            ii = feats['f0'] != 0

            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
                feats['f0'][~ii] = (feats['f0'][~ii] - mean) / std

            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()

        return feats, feats['f0'], str(filename)

    def __len__(self):
        return len(self.audio_files)
