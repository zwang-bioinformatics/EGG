###################################

import torch
import torch.nn as nn

import math
import string
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm

from init import *

###################################

import cv2

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = "DeJavu Serif"
matplotlib.rcParams['font.sans-serif'] = ["Arial"]

###################################

def line_break(n=15): print("\n"+"*"*n+"\n")

###################################
# clow = (53/255, 229/255, 163/255) #(128.0/255,0.0,0.0)
clow = (52/255, 223/255, 158/255) #(128.0/255,0.0,0.0)

chigh = (90/255, 177/255, 236/255)#(102.0/255,153.0/255,204.0/255)

def casp15_loss_figure(metrics, file_path, score_name, total): 

    labels = []
    high_values = []
    low_values = []

    for group_id in sorted(list(metrics.keys()), key = lambda x: (metrics[x][0], metrics[x][1])):

        if group_id + ".json" in TM_models or group_id + ".json" in QS_models: labels += [group_id[3:].replace("_","-")]
        else:  labels += [group_id]

        high_values += [metrics[group_id][0]]
        low_values += [metrics[group_id][1]]

    y_idx = list(range(len(labels)))

    plt.figure(figsize=(25,15))

    plt.barh(y_idx, high_values, color=chigh, linewidth=1.0, edgecolor='k', label="$N\ Loss("+score_name+"\mathrm{-}score)<0.1$")
    plt.yticks(y_idx, labels, fontsize = 35)

    for i, v in enumerate(high_values): 
        if v - low_values[i] > 0: 
            plt.text(low_values[i] + (v - low_values[i])/2, i, str(v - low_values[i]), color="black", fontsize=30, ha='center', va='center')

    plt.barh(y_idx, low_values, color=clow, linewidth=1.0, edgecolor='k', label="$N\ Loss("+score_name+"\mathrm{-}score)<0.05$")

    for i, v in enumerate(low_values): 
        if v > 0: plt.text(v/2, i, str(v), color="black", fontsize=30, ha='center', va='center')
    
    plt.xticks(fontsize = 35)

    for label in plt.gca().get_yticklabels():
        if label.get_text() in ["REGRESSION-METALAYER","EBM-METALAYER", "EBM-TRANSFORMER", "REGRESSION-TRANSFORMER"]: label.set_fontweight('bold')

    plt.legend(frameon=False,fontsize=35)

    plt.xlabel("$N$ (Total = " + str(total) + ")", fontsize=35)

    plt.tight_layout()

    plt.savefig(file_path)

    plt.clf()

    plt.close()

    return

def ndcg_figure(metrics, file_path, score_name, total): 

    plt.figure(figsize=(25,15))

    labels = []
    values = []

    for group_id in sorted(list(metrics.keys()), key = lambda x: metrics[x]):
        if group_id + ".json" in TM_models or group_id + ".json" in QS_models: labels += [group_id[3:].replace("_","-")]
        else:  labels += [group_id]
        values += [metrics[group_id]]

    y_idx = list(range(len(labels)))

    plt.barh(y_idx, values, color=chigh, linewidth=1.0, edgecolor='k')

    for i, v in enumerate(values): 
        if round(v,1) > 0: plt.text(v/2, i, str(round(v,2)), color="black", fontsize=30, ha='center', va='center')

    plt.yticks(y_idx, labels, fontsize = 35)
    plt.xticks(fontsize = 35)

    for label in plt.gca().get_yticklabels():
        if label.get_text() in ["REGRESSION-METALAYER","EBM-METALAYER", "EBM-TRANSFORMER", "REGRESSION-TRANSFORMER"]: label.set_fontweight('bold')

    plt.xlabel("NDCG@3 (" + str(total) + " targets: "+score_name+"-score)", fontsize=35)

    plt.tight_layout()

    plt.savefig(file_path)

    plt.clf()

    plt.close()

def merge_images(images, file_path): 

    ## Reference: https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/

    rows = None
    columns = None

    if len(images) == 2: 
        rows = 2
        columns = 1
    elif len(images) == 4: 
        rows = 2
        columns = 2
    
    assert rows is not None and columns is not None

    fig = plt.figure(figsize=(25 * columns, 15 * rows)) 

    for i,image_info in enumerate(images): 

        label, file = image_info

        fig.add_subplot(rows, columns, i + 1) 
        
        plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)) 
        plt.axis('off') 
        plt.title("("+string.ascii_uppercase[i]+") " + label, fontsize=50, pad=35, loc='left') 

    plt.tight_layout()

    plt.savefig(file_path)

    plt.clf()

    plt.close()

    return

###################################

def weighted_L1(pred, targ, n): 
    # divisor = 1 / n
    groups = {k:[0,0] for k in range(n)}

    crit = nn.L1Loss()

    for i, p in enumerate(pred): 
        if targ[i] == 1: groups[n-1] = [groups[n-1][0] + crit(p,targ[i]), groups[n-1][1] + 1]
        else: 
            groups[math.floor((targ[i]*n).item())][0] += crit(p,targ[i])
            groups[math.floor((targ[i]*n).item())][1] += 1

    sm = 0
    n = 0

    for g in groups: 
        if groups[g][1] > 0: 
            sm += groups[g][0]/groups[g][1]
            n += 1

    return sm / n

###################################

def general_results(model_inits, is_overall_fold): 

    ###################################

    group_mappings = json.load(open(GROUP_MAPPINGS,'r'))
    truth_df = pd.read_csv(GLOBAL_DF)

    ###################################

    eval_groups = None
    single_model_groups = None
    score_type = None
    score_key = None

    if is_overall_fold: 
        single_model_groups = single_model_groups_TM
        score_type = "SCORE"
        score_key = "tm"    
    else: 
        single_model_groups = single_model_groups_QS
        score_type = "QSCORE"
        score_key = "qs_best"

    eval_groups = []
    group_mappings_new = {}

    for id in group_mappings:
        if group_mappings[id] in single_model_groups:
            group_mappings_new[id] = group_mappings[id]
            eval_groups += [id]

    group_mappings = group_mappings_new 

    model_names = []

    for model in model_inits: 

        csv = ROOT + "results/" + model[0][:-5] + "/" + model[1] + "/predictions.csv"

        if not os.path.exists(csv): continue

        model_names += [model[0][:-5]]

        eval_groups += [model[0][:-5]]
    
        group_mappings[model[0][:-5]] = model[0][:-5]

        truth_df = truth_df.merge(pd.read_csv(csv),on="mdl")

    ###################################

    aggregated_metrics = {}

    target_map = {}
    target_results = {}

    for index, row in tqdm(truth_df.iterrows(),total=len(truth_df),desc="Aggregating Metrics"):
        target = row["trg"]
        model = row["mdl"]

        if target not in target_map: target_map[target] = {}

        target_map[target][model] = {
            "values":{
                "tm": row["tm_score"],
                "gdtts": row["gdtts"],
                "qs_best": row["qs_best"],
                "dockq_wave": row["dockq_wave"]
            },
            "info": {
                "n_mdl_chains": row["n_mdl_chains"],
                "n_trg_chains": row["n_trg_chains"]
            }
        }

        if target not in target_results: target_results[target] = {}

        for group_id in eval_groups:
            
            if group_id + "_" + score_type not in row: continue

            if not math.isnan(row[group_id + "_" + score_type]): 
                if group_id not in aggregated_metrics: aggregated_metrics[group_id] = {}

                if target not in aggregated_metrics[group_id]: 
                    aggregated_metrics[group_id][target] = {}
                
                if model not in aggregated_metrics[group_id][target]: 
                    aggregated_metrics[group_id][target][model] = row[group_id + "_" + score_type]

    ###################################

    line_break()

    MSE = {}
    L1 = {}

    for model_name in model_names: 

        mse = 0
        l1 = 0
        n = 0

        for target in tqdm(aggregated_metrics[model_name],desc="Computing MSE & L1 ("+model_name+")"): 

            for model in aggregated_metrics[model_name][target]: 
                prediction = aggregated_metrics[model_name][target][model]
                ground_truth = target_map[target][model]["values"][score_key]

                l1 += abs(prediction - ground_truth)
                mse += (prediction - ground_truth)**2
                n += 1

        mse /= n
        l1 /= n

        MSE[model_name] = mse
        L1[model_name] = l1

    ###################################

    is_hetero = {}

    CASP15_loss = {}
    CASP15_two_chains = {}
    CASP15_more_than_two_chains = {}

    for target in tqdm(target_map,desc="Computing CASP Loss"):
        top_reference = {}

        for model in target_map[target]:

            if score_key not in top_reference: 
                top_reference[score_key] = (target_map[target][model]["values"][score_key],target,model)
            elif top_reference[score_key][0] < target_map[target][model]["values"][score_key]: 
                top_reference[score_key] = (target_map[target][model]["values"][score_key],target,model)

        for group_id in aggregated_metrics:    
            if group_id not in CASP15_loss: CASP15_loss[group_id] = [0,0]
            if group_id not in CASP15_two_chains: CASP15_two_chains[group_id] = [0,0]
            if group_id not in CASP15_more_than_two_chains: CASP15_more_than_two_chains[group_id] = [0,0]
            if group_id not in target_results[target]: target_results[target][group_id] = {}

            preds = set()

            top_assigned = None

            diff = None
            r_diff = None

            try: 
                
                for model in aggregated_metrics[group_id][target]:
                    
                    curr_p = aggregated_metrics[group_id][target][model] # actual predicted quality scores

                    if top_assigned is None: top_assigned = (curr_p,[model])
                    elif curr_p > top_assigned[0]: top_assigned = (curr_p,[model])
                    elif curr_p == top_assigned[0]: top_assigned = (curr_p,top_assigned[1] + [model])

                if len(top_assigned[1]) == 1: 
                    pred_met = target_map[target][top_assigned[1][0]]["values"][score_key]
                    diff = abs(top_reference[score_key][0] - pred_met)    
                    r_diff = diff
                else: r_diff = "DUP"
                    #if target == "T1123o.pdb": 
                        # print(target,"|",diff,"|",top_assigned,"|",group_mappings[group_id],">",pred_met)

            except: None

            if diff is not None: 

                check_set = set(map(int,[
                    target_map[target][top_reference[score_key][2]]["info"]["n_mdl_chains"],
                    target_map[target][top_reference[score_key][2]]["info"]["n_trg_chains"],
                    target_map[target][top_assigned[1][0]]["info"]["n_mdl_chains"],
                    target_map[target][top_assigned[1][0]]["info"]["n_trg_chains"]
                ]))

                check_set = list(check_set)
                
                if target not in is_hetero and len(check_set) > 1: is_hetero[target] = check_set[0] > 2

                if diff < 0.1: 
                    CASP15_loss[group_id][0] += 1
                    if len(check_set) != 1: continue
                    elif check_set[0] == 2: CASP15_two_chains[group_id][0] += 1
                    elif check_set[0] > 2: CASP15_more_than_two_chains[group_id][0] += 1

                if diff < 0.05: 
                    CASP15_loss[group_id][1] += 1
                    if len(check_set) != 1: continue
                    elif check_set[0] == 2: CASP15_two_chains[group_id][1] += 1
                    elif check_set[0] > 2: CASP15_more_than_two_chains[group_id][1] += 1
                
            target_results[target][group_id]["CASP Loss"] = r_diff

    ###################################

    def compute_DCG(target, top_model, group_id, ranking):

        ## Reference: https://www.evidentlyai.com/ranking-metrics/ndcg-metric#discounted-gain-dcg

        score = 0

        reference_value = target_map[target][top_model]["values"][score_key]

        duplicate_values = None

        if group_id is not None:

            values = [aggregated_metrics[group_id][target][m] for m in ranking]

            duplicate_values = set(item for item, count in collections.Counter(values).items() if count > 1)

        for r_i, model in enumerate(ranking[:3]):

            position = r_i + 1

            model_value = target_map[target][model]["values"][score_key]

            rel_i = 1 - (reference_value - model_value)**2

            if group_id is not None: 
                if aggregated_metrics[group_id][target][model] in duplicate_values: continue

            score += rel_i / math.log2(1 + position)

        return score

    NDCG = {}

    for target in tqdm(target_map,desc="Computing NDCG"):

        reference_order = sorted(list(target_map[target].keys()),key=lambda model: target_map[target][model]["values"][score_key],reverse=True)

        true_ranking_score = compute_DCG(target, reference_order[0], None, reference_order)

        for group_id in aggregated_metrics:    
            if group_id not in NDCG: NDCG[group_id] = 0

            try: 

                group_order = sorted(
                    list(aggregated_metrics[group_id][target].keys()), 
                    key = lambda model: aggregated_metrics[group_id][target][model],
                    reverse = True
                )
            
                group_ranking_score = compute_DCG(target, reference_order[0], group_id, group_order)

                NDCG[group_id] += group_ranking_score/true_ranking_score

                target_results[target][group_id]["NDCG@3"] = group_ranking_score/true_ranking_score

            except: target_results[target][group_id]["NDCG@3"] = None

    ###################################

    line_break()

    CASP15_loss = {group_mappings[id]:CASP15_loss[id] for id in CASP15_loss}
    CASP15_two_chains = {group_mappings[id]:CASP15_two_chains[id] for id in CASP15_two_chains}
    CASP15_more_than_two_chains = {group_mappings[id]:CASP15_more_than_two_chains[id] for id in CASP15_more_than_two_chains}
    NDCG = {group_mappings[id]:NDCG[id] for id in NDCG}

    return {
        "MSE": MSE,
        "L1": L1,
        "CASP15_loss": CASP15_loss,
        "CASP15_two_chains": CASP15_two_chains,
        "CASP15_more_than_two_chains": CASP15_more_than_two_chains,
        "NDCG": NDCG
    }, target_results

###################################

def generate_target_xlsx(target_outputs, output_fl): 

    import xlsxwriter

    list_name = ["first sheet", "second sheet", "third sheet"]

    workbook = xlsxwriter.Workbook(output_fl)
    col_format = workbook.add_format({'bold': True})

    for category in target_outputs: 

        model_inits, results = target_outputs[category]

        single_model_groups = None
        
        if "TM" in category: single_model_groups = single_model_groups_TM
        elif "QS" in category: single_model_groups = single_model_groups_QS
        else: assert False

        group_mappings = json.load(open(GROUP_MAPPINGS,'r'))
        group_mappings_new = {}

        for id in group_mappings:
            if group_mappings[id] in single_model_groups: group_mappings_new[id] = group_mappings[id]
        
        group_mappings = group_mappings_new 

        group_order = list(k for k in group_mappings.keys())

        for model in model_inits: 
            group_mappings[model[0][:-5]] = model[0][:-5].replace("_","-")
            group_order = [model[0][:-5]] + group_order

        for metric in ["CASP Loss", "NDCG@3"]: 

            worksheet = workbook.add_worksheet(category + " - " + metric)

            for i, group in enumerate(group_order): worksheet.write(0, i + 1, group_mappings[group], col_format)

            for i, target in enumerate(results): 

                target_id = target[:-4]

                worksheet.write(i + 1, 0, target_id, col_format)
                
                for k, id in enumerate(group_order): 
                    group_metric = results[target][id][metric]

                    if group_metric is not None: worksheet.write(i + 1, k + 1, group_metric)
                    else: worksheet.write(i + 1, k + 1, "N/A")

            # print(results)

    workbook.close()
    
###################################