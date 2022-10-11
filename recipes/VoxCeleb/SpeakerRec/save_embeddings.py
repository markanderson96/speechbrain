#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
import numpy as np
import pandas as pd
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            if torch.any(torch.isnan(emb)):
                raise ValueError('NaN in embedding')
            if torch.any(torch.isinf(emb)):
                raise ValueError('Inf in embedding')
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    from voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev", "test"],
        split_ratio=[90, 10],
        seg_dur=3.0,
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])
    params["compute_features"].to(params["device"])

    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    enrol_dict = compute_embedding_loop(enrol_dataloader)
    #test_dict = compute_embedding_loop(test_dataloader)

    for key in enrol_dict:
        enrol_dict[key] = enrol_dict[key].detach().cpu().numpy()

    keys = list(enrol_dict.keys())
    for idx, key in enumerate(keys):
        key = key.split('/')[0]
        keys[idx] = key

    def unique_list(input_list):
        output_list = []
        for word in input_list:
            if word not in output_list:
                output_list.append(word)
        return output_list

    unique_keys = unique_list(keys)
    emb_dict = dict.fromkeys(unique_keys)
    for unique_key in unique_keys:
        emb_list = []
        for key in enrol_dict:
            if key.__contains__(unique_key):
                emb_list.append(enrol_dict[key])
        emb_dict[unique_key] = emb_list

    df = pd.DataFrame(columns=['id', 'emb'])
    for key in emb_dict.keys():
        arrays = emb_dict[key]
        for value in arrays:
            df_dict = {'id': key, 'emb': [value]}
            df_temp = pd.DataFrame(df_dict)
            df = pd.concat([df, df_temp])

    init = params['init']
    df.to_pickle(f'emb_{init}.pkl')
