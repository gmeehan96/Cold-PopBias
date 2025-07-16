# On Inherited Popularity Bias in Cold-Start Item Recommendation
This repository contains code and resources for the above ACM RecSys 2025 short paper. The model implementations and setup were based on the [ColdRec](https://github.com/YuanchenBei/ColdRec) repository, with adjustments made to model code to improve performance and align more closely with the original implementations by the authors. We also include the [supplementary material document](https://github.com/gmeehan96/Cold-PopBias/blob/main/ColdPopBias_SupplementaryMaterialFinal.pdf) mentioned in the paper.

## Setup
Below we outline how to setup the environment and datasets to run the training.

#### Environment
Code was run with Python 3.9.2, and the necessary packages are in the [requirements](https://github.com/gmeehan96/Cold-PopBias/blob/main/requirements.txt) file. 

#### Data
All split interaction data and pretrained FREEDOM embeddings can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1I2Uh7bF98oPVbnZqaE-Na7gRg9ER5FC9?usp=sharing). Folders for each dataset should be unzipped into the `./src/data` folder, and the embeddings should go into a `./src/emb` folder. 

Item content features for each dataset can be downloaded from the Google Drives ([Clothing/Electronics](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG) and [Microlens](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w)) for the [MMRec](https://github.com/enoche/MMRec) toolbox (e.g. `text_feat.npy`). These should be placed into the `./src/data/[dataset_name]/feats_raw` folders. To perform the feature L2 normalization and concatenation, run 
`python data/feats_concat.py --dataset [dataset_name]`
This will create the concatenated feature file in the corresponding `feats` folder.

## Training
To run training for a specific model/dataset, use the command

`python main.py --dataset [dataset_name] --model [model_name]`

This will run training using the [optimal hyperparameters](https://github.com/gmeehan96/Cold-PopBias/blob/main/src/hyperparams.yml) and output the user acc., item acc., and exposure metrics before and after magnitude scaling is applied. 

