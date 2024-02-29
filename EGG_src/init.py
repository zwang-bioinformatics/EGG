###################################

import os
import json

###################################

ROOT = "/home/EGG/"

GROUP_MAPPINGS = "./group_mappings.json"
GLOBAL_DF = "./global_df.csv"
EMA_TARGETS = "./ema_targets.json"

BLIND_TEST_DATABASE = "./blind_test/"
TRAINING_DATABASE = "./training/"
VALIDATION_DATABASE = "./validation/"

###################################

assert os.path.exists(ROOT)

if BLIND_TEST_DATABASE is not None: assert os.path.exists(BLIND_TEST_DATABASE)
if TRAINING_DATABASE is not None: assert os.path.exists(TRAINING_DATABASE)
if VALIDATION_DATABASE is not None: assert os.path.exists(VALIDATION_DATABASE)

assert os.path.exists(GROUP_MAPPINGS)
assert os.path.exists(GLOBAL_DF)
assert os.path.exists(EMA_TARGETS)

###################################

TM_models = set([
    "TM_EBM_TRANSFORMER.json",
    "TM_EBM_METALAYER.json",
    "TM_REGRESSION_TRANSFORMER.json",
    "TM_REGRESSION_METALAYER.json"
])

QS_models = set([
    "QS_EBM_TRANSFORMER.json",
    "QS_EBM_METALAYER.json",
    "QS_REGRESSION_TRANSFORMER.json",
    "QS_REGRESSION_METALAYER.json"
])

backbone_map = {
    # EBM Config: (BackBone Config Name, Epoch)
    
    "TM_EBM_TRANSFORMER.json" : ("TM_REGRESSION_TRANSFORMER","default"),
    "TM_EBM_METALAYER.json" : ("TM_REGRESSION_METALAYER","default"),
    "QS_EBM_TRANSFORMER.json" : ("QS_REGRESSION_TRANSFORMER","default"),
    "QS_EBM_METALAYER.json" : ("QS_REGRESSION_METALAYER","default")
}

overall_fold_models = set([
    "TM_REGRESSION_TRANSFORMER.json",
    "TM_EBM_TRANSFORMER.json",
    "TM_REGRESSION_METALAYER.json",
    "TM_EBM_METALAYER.json"
])

interface_models = set([
    "QS_REGRESSION_TRANSFORMER.json",
    "QS_EBM_TRANSFORMER.json",
    "QS_REGRESSION_METALAYER.json",
    "QS_EBM_METALAYER.json"
])

###################################

single_model_groups_TM = set([
    "GuijunLab-RocketX",
    "GuijunLab-Threader",
    "GuijunLab-Human",
    "GuijunLab-Assembly",
    "FoldEver",
    "MULTICOM_deep",
    "MULTICOM_egnn",
    "ChaePred",
    "APOLLO",
    "LAW",
    "MASS"
])

single_model_groups_QS = set([
    "FoldEver",
    "MULTICOM_deep",
    "DLA-Ranker",
    "ChaePred",
    "Bhattacharya",
    "Manifold",
    "APOLLO",
    "LAW",
    "MASS"
])

###################################
