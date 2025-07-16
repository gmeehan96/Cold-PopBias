import numpy as np
import argparse
import os

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="clothing", help="Dataset to use.")
parser.add_argument(
    "--rawdir", type=str, default="feats_raw", help="Directory of the raw features."
)
args = parser.parse_args()

raw_feat_dir = './data/%s/%s'%(args.dataset,args.rawdir)
raw_feat_files = os.listdir(raw_feat_dir)

raw_feat_arrs = [np.load("%s/%s"%(raw_feat_dir,f)) for f in raw_feat_files]

concat = np.concatenate([normalized(arr) for arr in raw_feat_arrs], axis=1)

output_dir = './data/%s/feats'%args.dataset
os.makedirs(output_dir, exist_ok=True)

with open(
    "%s/item_content_concat_norm.npy"%output_dir, "wb"
) as f:
    np.save(f, concat)