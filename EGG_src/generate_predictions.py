###################################

# Author: Andrew Jordan Siciliano

# Name: Generate Predictions (Blind Test)

###################################

import os
import csv
import math
import json
import pprint
import collections
import pandas as pd
from tqdm import tqdm
from termcolor import colored

import torch
from torch.utils import data
from torch_geometric.data import Data

from init import *
from utils import *
from model import *

###################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m","--model_config",action="store",type=str,
    default="TM_REGRESSION_TRANSFORMER.json", 
    help="model config (.json)... must be located @:ROOT/configs/",
)
parser.add_argument("-e","--model_epoch",action="store",type=str,
    default="default",
    help="epoch to run evaluations",
)

parser.add_argument("-w","--overwrite",action="store_true",help="overwrite predictions")

parser.add_argument("-d","--device",action="store",type=str,default="cpu",help="device id for generating predictions")

args = parser.parse_args()

assert os.path.exists(ROOT + "/configs/" + args.model_config), "ERROR: config file does not exist!"

assert os.path.exists(ROOT + "/models/" + args.model_config[:-5] + "/" + args.model_epoch + "/model.pt"), "ERROR: Model file does not exist!"

OUT_DIR = ROOT + "/results/"

save_dir = OUT_DIR + args.model_config[:-5] + "/" + args.model_epoch + "/"

if not os.path.exists(save_dir): os.makedirs(save_dir)

###################################

line_break()

print(colored("Starting!","blue"))

line_break()

print("Config:",colored(args.model_config,"green"),"\n")
print("Epoch:",colored(args.model_epoch,"green"),"\n")
print("Output Directory:",colored(OUT_DIR,"green"),"\n")
print("Overwrite Predictions:",colored(args.overwrite,"red"),"\n")
print("Device:",colored(args.device,"green"))

line_break()

###################################

score_type = None
score_key = None

if args.model_config in overall_fold_models:
    score_type = "SCORE"
    score_key = "tm"
elif args.model_config in interface_models:
    score_type = "QSCORE"
    score_key = "qs_best"

assert score_type is not None and score_key is not None

###################################

truth_df = pd.read_csv(GLOBAL_DF)
group_mappings = json.load(open(GROUP_MAPPINGS,'r'))
target_data = json.load(open(EMA_TARGETS,'r'))
target_data = {k: v[0] for k,v in target_data.items()}

###################################

torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

if not os.path.exists(save_dir + "predictions/"): os.makedirs(save_dir + "predictions/")

config = json.load(open(ROOT + "/configs/" + args.model_config,'r'))

model = None

if config["model_type"] == "global_graph": model = GraphModel(setup = config["init"])
elif config["model_type"] == "ebm": model = EBM(setup=config["init"],in_channels=config["in_channels"])	

assert model is not None

state_dict = torch.load(ROOT + "/models/" + args.model_config[:-5] + "/" + args.model_epoch + "/model.pt", map_location='cpu')

model.load_state_dict(state_dict)

model = model.to(args.device)

model.eval()

###################################

with torch.no_grad():

    for file in tqdm(os.listdir(BLIND_TEST_DATABASE + "/processed/"),desc="Generating Predictions"): 
        if not args.overwrite and os.path.exists(save_dir + "predictions/" + file): continue

        data_point = torch.load(BLIND_TEST_DATABASE + "/processed/" + file)
        multimer = data_point.multimer

        content = truth_df.loc[truth_df['mdl'] == multimer]

        if len(content) != 1: continue

        if config["model_type"] == "global_graph": 

            example = Data(
                edge_attr=data_point.edge_attr.float().to(args.device),
                node_features=data_point.node_features.float().to(args.device),
                edge_index=data_point.edge_i.long().to(args.device),
                esm=data_point.esm_features.float().to(args.device),
                esm_stats=data_point.esm_stat_features.float().to(args.device)
            )

            _, global_embeddings, final_out = model([example],torch.tensor([0]).to(args.device))

            global_embeddings = global_embeddings.to('cpu')
            final_out = final_out.to('cpu')
            
            protein_target = content['trg'].item()

            final_out = torch.sigmoid(final_out)

            embedded_data = Data(
                protein_id = protein_target,
                model_id = multimer,
                prediction =final_out[0].item(),
                global_embeddings = global_embeddings[0]
            )

            torch.save(embedded_data,save_dir + "predictions/" + file)

        elif config["model_type"] == "ebm":

            linear_space = torch.from_numpy(np.linspace(0.05, 0.95, num=1000)).float()
			
            y_i = torch.unsqueeze(torch.logit(linear_space),dim=1).to(args.device)

            backbone_save_dir = backbone_map[args.model_config]
            backbone_save_dir = OUT_DIR + "/" + backbone_save_dir[0] + "/" + backbone_save_dir[1] + "/predictions/"

            assert os.path.exists(backbone_save_dir), "Please run this script with the backbone first before running the EBM!"

            x_i = torch.load(backbone_save_dir + file).global_embeddings.to(args.device)
            x_i = x_i.repeat(1000, 1)

            out_i = model(torch.cat((x_i,y_i),dim=1))
            out_i = torch.squeeze(out_i,dim=1).cpu()

            prediction_i = linear_space[torch.argmax(out_i)].cpu()

            protein_target = content['trg'].item()

            embedded_data = Data(
                protein_id = protein_target,
                model_id = multimer,
                prediction = prediction_i.item()
            )

            torch.save(embedded_data,save_dir + "predictions/" + file)

###################################

with open(save_dir + "predictions.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    
    field = ["mdl",args.model_config[:-5]+"_" + score_type]
    writer.writerow(field)

    for prediction_file in tqdm(os.listdir(save_dir + "predictions/"), "Writing to CSV"): 

        pred_out = torch.load(save_dir + "predictions/" + prediction_file)

        writer.writerow([pred_out.model_id,pred_out.prediction])

print("\nSaved Predictions to CSV @:",save_dir + "predictions.csv")

###################################

line_break()

print("Done!")

line_break()

###################################
