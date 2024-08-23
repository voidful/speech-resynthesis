import argparse
import logging
import os
import random
import shutil
import gc
from typing import List, Tuple

import numpy as np
import joblib
import torch
import tqdm
import soundfile as sf
import torch.nn.functional as F
import fairseq


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000, use_cuda=True):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = sf.read(path)
        if channel_id is not None:
            assert wav.ndim == 2, f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id - 1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.use_cuda:
                x = x.cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def get_feature_reader(feature_type):
    feature_readers = {
        "hubert": HubertFeatureReader,
    }
    if feature_type not in feature_readers:
        raise NotImplementedError(f"{feature_type} is not supported.")
    return feature_readers[feature_type]


def get_feature_iterator(feature_type, checkpoint_path, layer, manifest_path, sample_pct, channel_id):
    feature_reader_cls = get_feature_reader(feature_type)
    with open(manifest_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        file_path_list = [os.path.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        if sample_pct < 1.0:
            file_path_list = random.sample(file_path_list, int(sample_pct * len(file_path_list)))
        num_files = len(file_path_list)
        reader = feature_reader_cls(checkpoint_path=checkpoint_path, layer=layer)

        def iterate():
            for file_path in file_path_list:
                feats = reader.get_feats(file_path, channel_id=channel_id)
                yield feats.cpu().numpy()

    return iterate, num_files


def get_features(feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten, channel_id):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        channel_id=channel_id
    )
    iterator = generator()

    features_list = []
    for features in tqdm.tqdm(iterator, total=num_files):
        features_list.append(features)

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if flatten:
        return np.concatenate(features_list)

    return features_list


def get_and_dump_features(feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten, out_features_path):
    # Feature extraction
    features_batch = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        flatten=flatten,
    )

    # Save features
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(manifest_path, os.path.join(out_dir_path, os.path.basename(manifest_path)))
    np.save(out_features_path, features_batch)

    return features_batch


def get_audio_files(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert len(items) == 2, f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Quantize using K-means clustering over acoustic features.")
    parser.add_argument("--feature_type", type=str, choices=["logmel", "hubert", "w2v2", "cpc"], required=True,
                        help="Acoustic feature type")
    parser.add_argument("--acoustic_model_path", type=str, help="Pretrained acoustic model checkpoint")
    parser.add_argument("--layer", type=int, default=-1, help="The layer of the pretrained model to extract features from")
    parser.add_argument("--kmeans_model_path", type=str, required=True, help="K-means model file path to use for inference")
    parser.add_argument("--features_path", type=str, help="Features file path. Not needed if you have dumped features")
    parser.add_argument("--manifest_path", type=str, help="Manifest file containing the root dir and file names")
    parser.add_argument("--out_quantized_file_path", required=True, type=str, help="File path of quantized output.")
    parser.add_argument("--extension", type=str, default=".flac", help="File extension")
    parser.add_argument("--channel_id", choices=['1', '2'], help="Audio channel to extract the units in case of stereo file")
    parser.add_argument("--hide-fname", action='store_true', help="Hide file names in the output file.")
    return parser


def main(args, logger):
    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        features_batch = get_features(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            sample_pct=1.0,
            flatten=False,
            channel_id=int(args.channel_id) if args.channel_id else None,
        )
        logger.info(f"Features extracted for {len(features_batch)} utterances.\n")
        logger.info(f"Dimensionality of representation = {features_batch[0].shape[1]}")

    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    _, fnames, _ = get_audio_files(args.manifest_path)

    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    logger.info(f"Writing quantized predictions to {args.out_quantized_file_path}")
    with open(args.out_quantized_file_path, "w") as fout:
        for i, feats in enumerate(features_batch):
            pred = kmeans_model.predict(feats)
            pred_str = " ".join(str(p) for p in pred)
            base_fname = os.path.basename(fnames[i]).rstrip('.'+args.extension.lstrip('.'))
            if args.channel_id is not None:
                base_fname = f"{base_fname}-channel{args.channel_id}"
            if not args.hide_fname:
                fout.write(f"{base_fname}|{pred_str}\n")
            else:
                fout.write(f"{pred_str}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)