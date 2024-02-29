###################################

# Author: Andrew Jordan Siciliano

# Name: Run Eval (Blind Test)

# Reference: https://git.scicore.unibas.ch/schwede/casp15_ema

###################################

from termcolor import colored
import pprint

from init import *
from utils import *

###################################

reproduce_figures = ROOT + "reproduced_figures/"

if not os.path.exists(reproduce_figures + "sub_figs/"): os.makedirs(reproduce_figures + "sub_figs/")

###################################

TM_inits = [(model,"default") for model in TM_models]
TM_results = general_results(TM_inits,is_overall_fold=True)

print("\nTM-L1:\n")
pprint.pprint(TM_results["L1"])
print("\nTM-MSE:\n")
pprint.pprint(TM_results["MSE"])

line_break()

###################################

QS_inits = [(model,"default") for model in QS_models]
QS_results = general_results(QS_inits,is_overall_fold=False)

print("QS-L1:\n")
pprint.pprint(QS_results["L1"])
print("\nQS-MSE:\n")
pprint.pprint(QS_results["MSE"])

model_categories = {
    "TM": TM_results,
    "QS": QS_results
}

line_break()

###################################

with tqdm(total=len(model_categories)*4 + 3,desc="Generating Figures") as pbar:

    for score_type in model_categories: 

        casp15_loss_figure(
            model_categories[score_type]["CASP15_loss"], 
            reproduce_figures + "sub_figs/" + score_type+"_CASP15_loss.png", 
            score_type, 40
        )
        pbar.update(1) 
        
        casp15_loss_figure(
            model_categories[score_type]["CASP15_two_chains"], 
            reproduce_figures + "sub_figs/" + score_type+"_CASP15_two_chains_loss.png", 
            score_type, 23
        ) 

        pbar.update(1)
        
        casp15_loss_figure(
            model_categories[score_type]["CASP15_more_than_two_chains"], 
            reproduce_figures + "sub_figs/" + score_type+"_CASP15_more_than_two_chains_loss.png", 
            score_type, 17
        ) 
        pbar.update(1)
        
        ndcg_figure(
            model_categories[score_type]["NDCG"], 
            reproduce_figures + "sub_figs/" + score_type + "_NDCG.png", 
            score_type, 40
        
        ) 
        pbar.update(1)
        
    merge_images([
        ("TM-score ranking loss",reproduce_figures + "sub_figs/" + "TM_CASP15_loss.png"),
        ("QS-score ranking loss",reproduce_figures + "sub_figs/" + "QS_CASP15_loss.png")
    ], reproduce_figures + "combined_CASP15_loss.png")

    pbar.update(1)

    merge_images([
        ("Ranking loss (TM-score) of two-chain models",reproduce_figures + "sub_figs/" + "TM_CASP15_two_chains_loss.png"),
        ("Ranking loss (TM-score) of models with more than two-chains",reproduce_figures + "sub_figs/" + "TM_CASP15_more_than_two_chains_loss.png"),
        ("Ranking loss (QS-score) of two-chain models",reproduce_figures + "sub_figs/" + "QS_CASP15_two_chains_loss.png"),
        ("Ranking loss (QS-score) of models with more than two-chains",reproduce_figures + "sub_figs/" + "QS_CASP15_more_than_two_chains_loss.png")
    ], reproduce_figures + "combined_multichain_CASP15_loss.png")

    pbar.update(1)

    merge_images([
        ("TM-score ranking loss",reproduce_figures + "sub_figs/TM_NDCG.png"),
        ("QS-score ranking loss",reproduce_figures + "sub_figs/QS_NDCG.png")
    ], reproduce_figures + "combined_NDCG.png")

    pbar.update(1)

###################################

line_break()

print("Done!")

line_break()

###################################