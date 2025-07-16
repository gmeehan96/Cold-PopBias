import argparse
import torch
import numpy as np
import pickle
from data_utils import DataLoader
import os
import copy
import yaml
from GAR import GAR
from Heater import Heater
from GoRec import GoRec
from magnitude_scaling import compress_shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clothing")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_gpu", default=True, help="Whether to use CUDA")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--feat_dir", type=str, default="feats", help="Feat file location"
    )
    parser.add_argument("--backbone", type=str, default="FREEDOM")
    parser.add_argument("--model", default="Heater", type=str)
    parser.add_argument("--topN", default="20")

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset

    if model == "GAR" or model == "Heater":
        if args.dataset == "electronics":
            args.patience = 15
    if model == "GoRec":
        args.bs = 256
    elif model in ["GAR", "Heater"]:
        args.bs = 1024

    args.emb_size = 64
    args.cold_object = "item"
    args.save_output = False
    args.save_emb = False
    args.eval_freq = 1
    print(args)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu"
    )
    # data loader
    training_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_train.csv"
    )
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/overall_val.csv"
    )
    warm_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_val.csv"
    )
    cold_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/cold_item_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/overall_test.csv"
    )
    warm_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/warm_test.csv"
    )
    cold_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_item/cold_item_test.csv"
    )

    data_info_dict = pickle.load(
        open(f"./data/{dataset}/cold_item/info_dict.pkl", "rb")
    )
    user_num = data_info_dict["user_num"]
    item_num = data_info_dict["item_num"]
    warm_user_idx = data_info_dict["warm_user"]
    warm_item_idx = data_info_dict["warm_item"]
    cold_user_idx = data_info_dict["cold_user"]
    cold_item_idx = data_info_dict["cold_item"]
    print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

    # content obtaining
    feat_filenames = sorted(os.listdir(f"./data/{dataset}/feats"))
    feat_files = [f"./data/{dataset}/feats/{f}" for f in feat_filenames]
    item_content = torch.from_numpy(
        [np.load(f).astype(np.float32) for f in feat_files][0]
    )

    with open("./hyperparams.yml", "r") as f:
        config = yaml.safe_load(f)[model][dataset]

    inp_args = copy.deepcopy(args)
    for k, v in config.items():
        vars(inp_args)[k] = v

    model_obj = eval(args.model)(
        inp_args,
        training_data,
        warm_valid_data,
        cold_valid_data,
        all_valid_data,
        warm_test_data,
        cold_test_data,
        all_test_data,
        user_num,
        item_num,
        warm_user_idx,
        warm_item_idx,
        cold_user_idx,
        cold_item_idx,
        device,
        user_content=None,
        item_content=item_content,
    )
    model_obj.train()
    
    print("Pre-Scaling:")
    model_obj.eval_test()
    print("Post-Scaling:")
    model_obj.item_emb = compress_shift(
        model_obj.item_emb,
        alpha=config["scaling_alpha"],
        warm_item_idx=range(len(model_obj.data.mapped_warm_item_idx)),
    )
    model_obj.eval_test()


if __name__ == "__main__":
    main()
