###################################

# Author: Andrew Jordan Siciliano

# Name: EGG Training Script

###################################

import os

import json
import pprint
import random
from numpy import random as nrand
from tqdm import tqdm
from scipy.stats import norm
from termcolor import colored
from more_itertools import chunked

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

parser.add_argument("-e","--epochs",action="store",type=int,
    default=10,
    help="number of epochs to train"
)

parser.add_argument("-b","--batch_size",action="store",type=int,
    default=5,
    help="batch_size for training"
)

parser.add_argument("-d","--device",action="store",type=str,default="cpu",help="device id for generating predictions")

args = parser.parse_args()

assert os.path.exists(ROOT + "configs/" + args.model_config), "ERROR: config file does not exist!"

###################################

line_break()

print(colored("Starting!","blue"))

line_break()

print("Config:",colored(args.model_config,"green"),"\n")
print("Epochs:",colored(args.epochs,"green"),"\n")
print("Batch Size:",colored(args.batch_size,"green"),"\n")
print("Device:",colored(args.device,"green"))

###################################

config = json.load(open(ROOT + "configs/" + args.model_config,'r'))

assert os.path.exists(TRAINING_DATABASE + "processed/")
training_database = list(os.listdir(TRAINING_DATABASE + "processed/"))

assert os.path.exists(VALIDATION_DATABASE + "processed/")
validation_database = list(os.listdir(VALIDATION_DATABASE + "processed/"))

###################################

torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

model = None
backbone_model = None

if config["model_type"] == "global_graph": 

    model = GraphModel(setup = config["init"]).to(args.device)

elif config["model_type"] == "ebm": 

    backbone_model = GraphModel(setup = json.load(open(ROOT + "configs/" + backbone_map[args.model_config][0] + ".json",'r'))["init"])

    backbone_state_dict = torch.load(
        ROOT + "/models/" + backbone_map[args.model_config][0] + "/" + backbone_map[args.model_config][1] + "/model.pt", 
        map_location='cpu'
    )
    backbone_model.load_state_dict(backbone_state_dict)

    backbone_model = backbone_model.to(args.device)
    backbone_model.eval()

    model = EBM(setup=config["init"],in_channels=config["in_channels"]).to(args.device)

optimizer = None

if config["optimizer"][0] == "adamw": optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["optimizer"][1])
elif config["optimizer"][0] == "SGD": optimizer = torch.optim.SGD(model.parameters(),lr=config["learning_rate"],momentum=config["optimizer"][1])

assert optimizer is not None
assert model is not None

target_type = None

if args.model_config in TM_models: target_type = "tm"
elif args.model_config in QS_models: target_type = "qs_best"

###################################

assert target_type is not None

metrics = {}

for epoch in range(1,args.epochs+1): 

    line_break()

    metrics[epoch] = {
        "Epoch": epoch,
        "Training": {
            "Loss": 0,
            "Learning Rate": optimizer.param_groups[-1]['lr']
        },
        "Validation": {
            "L1": {
                0: [0,0],
                1: [0,0],
                2: [0,0],
                3: [0,0]
            }
        }
    }

    if config["model_type"] == "ebm": 
        metrics[epoch]["Validation"]["MSE"] = {
            0: [0,0],
            1: [0,0],
            2: [0,0],
            3: [0,0]
        }

    if config["shuffle"] == True: random.shuffle(training_database)

    ###################################

    model.train()
    
    n = 0

    for batch in tqdm(list(chunked(training_database[:10], args.batch_size)),desc="Training (Epoch "+str(epoch)+")"): 

        if config["model_type"] == "global_graph": 

            batched_examples = []

            batch_id = []
            targets = []

            for e_i,example_fl in enumerate(batch):

                example = torch.load(TRAINING_DATABASE + "processed/" + example_fl)

                batch_id += [e_i]

                batched_examples += [Data(
                    edge_attr=example.edge_attr.float().to(args.device),
                    node_features=example.node_features.float().to(args.device),
                    edge_index=example.edge_i.long().to(args.device),
                    esm=example.esm_features.float().to(args.device),
                    esm_stats=example.esm_stat_features.float().to(args.device)
                )]
                if target_type == "tm": targets += [example.global_score.float().to(args.device)]
                elif target_type == "qs_best":targets += [example.interface_score.float().to(args.device)]

            batch_id = torch.tensor(batch_id).to(args.device)
            targets = torch.tensor(targets).float().to(args.device)
            sig = nn.Sigmoid()

            _, _, final_out = model(batched_examples,batch_id)
            final_out = sig(torch.squeeze(final_out,dim=1))

            loss = weighted_L1(final_out,targets, config["weight_distribution"]) # remember to subtract one in the config...

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics[epoch]["Training"]["Loss"] += loss.item()
            n += 1

        elif config["model_type"] == "ebm": 

            curr_in = None
            logged_densities = []

            with torch.no_grad():

                for e_i,example_fl in enumerate(batch):

                    example = torch.load(TRAINING_DATABASE + "processed/" + example_fl)

                    batched_example = [Data(
                        edge_attr=example.edge_attr.float().to(args.device),
                        node_features=example.node_features.float().to(args.device),
                        edge_index=example.edge_i.long().to(args.device),
                        esm=example.esm_features.float().to(args.device),
                        esm_stats=example.esm_stat_features.float().to(args.device)
                    )]

                    _, x_i, _ = backbone_model(batched_example,torch.tensor([0]).to(args.device))

                    del batched_example

                    target = None

                    if target_type == "tm": target = example.global_score.float()
                    elif target_type == "qs_best": target = example.interface_score.float()

                    y_i = target.item()

                    if y_i == 1: target = torch.tensor(0.99)
                    if y_i == 0: target = torch.tensor(0.01)
                    
                    y_i = torch.logit(target).item()
                    
                    if target + config["quartile_spread"] < 1: logit_stretch = torch.logit(torch.tensor(target + config["quartile_spread"]))-y_i
                    else: logit_stretch = y_i - torch.logit(torch.tensor(target - config["quartile_spread"]))
					
                    sigma_k = logit_stretch / .675

                    #**** Truth Examples ****
                    
                    variance_beta = config["beta"]*(sigma_k**2)
                    v_i = nrand.normal(loc=0,scale=variance_beta**0.5,size=1)
                    
                    y_i_0 = torch.tensor(y_i + v_i)
                    truth_probability_density = torch.tensor(norm.pdf(y_i_0, loc=y_i, scale=sigma_k))

                    #**** Adversarial Examples ****
                    
                    y_i_m = torch.tensor(nrand.normal(loc=y_i_0,scale=sigma_k,size=config["adversarial_examples"]))
                    adversarial_probability_density = torch.tensor(norm.pdf(y_i_m, loc=y_i_0, scale=sigma_k))

                    curr_x = x_i.repeat(config["adversarial_examples"]+1, 1)
                    curr_y = torch.unsqueeze(torch.cat((y_i_0,y_i_m),dim=0),dim=1)

                    batch_i_in = torch.cat((curr_x,curr_y),dim=1)

                    if curr_in is None: curr_in = batch_i_in
                    else: curr_in = torch.cat((curr_in,batch_i_in),dim=0)

                    logged_densities += [torch.log(torch.cat((truth_probability_density,adversarial_probability_density),dim=0)).to(args.device)]

            curr_in = curr_in.float().to(args.device)
            curr_out = model(curr_in)

            loss = None

            for e_i,densities in enumerate(logged_densities):
                f_out = curr_out[e_i*(config["adversarial_examples"]+1):e_i*(config["adversarial_examples"]+1)+config["adversarial_examples"]+1]
                f_out = torch.squeeze(f_out,dim=1)

                difference = f_out - densities

                loss_i = torch.unsqueeze(f_out[0] - densities[0] - torch.logsumexp(difference,0),dim=0)
                    
                if loss is None: loss = loss_i
                else: loss = torch.cat((loss,loss_i))

            loss = -torch.mean(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics[epoch]["Training"]["Loss"] += loss.item()
            n += 1

    metrics[epoch]["Training"]["Loss"] /= n

    ###################################

    model.eval()

    crit_L1 = nn.L1Loss()
    crit_MSE = nn.MSELoss()

    with torch.no_grad():

        for example_fl in tqdm(list(validation_database)[:25],desc="Validation"): 

            example = torch.load(TRAINING_DATABASE + "processed/" + example_fl)

            batch_id = [0]

            batched_examples = [Data(
                edge_attr=example.edge_attr.float().to(args.device),
                node_features=example.node_features.float().to(args.device),
                edge_index=example.edge_i.long().to(args.device),
                esm=example.esm_features.float().to(args.device),
                esm_stats=example.esm_stat_features.float().to(args.device)
            )]

            if target_type == "tm": targets = [example.global_score.float().to(args.device)]
            elif target_type == "qs_best": targets = [example.interface_score.float().to(args.device)]

            batch_id = torch.tensor(batch_id).to(args.device)
            targets = torch.tensor(targets).float().to(args.device)
            sig = nn.Sigmoid()

            if config["model_type"] == "global_graph": 

                _, _, final_out = model(batched_examples,batch_id)
                final_out = sig(torch.squeeze(final_out,dim=1))

                target_item = targets.item()

                if target_item == 1: 
                    metrics[epoch]["Validation"]["L1"][3][0] += crit_L1(final_out,targets).item()
                    metrics[epoch]["Validation"]["L1"][3][1] += 1
                else: 
                    metrics[epoch]["Validation"]["L1"][math.floor(target_item*4)][0] += crit_L1(final_out,targets).item()
                    metrics[epoch]["Validation"]["L1"][math.floor(target_item*4)][1] += 1

            elif config["model_type"] == "ebm": 
                linear_space = torch.from_numpy(np.linspace(0.05, 0.95, num=1000)).float()

                y_i = torch.unsqueeze(torch.logit(linear_space),dim=1).to(args.device)
                _, x_i, _ = backbone_model(batched_examples,torch.tensor([0]).to(args.device))
                x_i = x_i.repeat(1000, 1)

                out_i = model(torch.cat((x_i,y_i),dim=1))
                out_i = torch.squeeze(out_i,dim=1).cpu()

                prediction_i = linear_space[torch.argmax(out_i)].cpu()

                target_item = targets.item()

                if target_item == 1: 
                    metrics[epoch]["Validation"]["L1"][3][0] += crit_L1(prediction_i,targets).item()
                    metrics[epoch]["Validation"]["L1"][3][1] += 1
                else: 
                    metrics[epoch]["Validation"]["L1"][math.floor(target_item*4)][0] += crit_L1(prediction_i,targets).item()
                    metrics[epoch]["Validation"]["L1"][math.floor(target_item*4)][1] += 1

                if target_item == 1: 
                    metrics[epoch]["Validation"]["MSE"][3][0] += crit_MSE(prediction_i,targets).item()
                    metrics[epoch]["Validation"]["MSE"][3][1] += 1
                else: 
                    metrics[epoch]["Validation"]["MSE"][math.floor(target_item*4)][0] += crit_MSE(prediction_i,targets).item()
                    metrics[epoch]["Validation"]["MSE"][math.floor(target_item*4)][1] += 1
                
    objective = 0
    n = 0

    for metric in metrics[epoch]["Validation"]: 
        for group in metrics[epoch]["Validation"][metric]:
            metrics[epoch]["Validation"][metric][group] = metrics[epoch]["Validation"][metric][group][0] / metrics[epoch]["Validation"][metric][group][1]
            objective += metrics[epoch]["Validation"][metric][group]
            n += 1

    objective /= n

    metrics[epoch]["Validation"]["objective"] = objective
    
    epath = ROOT + "/models/" + args.model_config[:-5] + "/epoch_" + str(epoch) + "/"
    if not os.path.exists(epath): os.makedirs(epath)

    with open(epath+"metrics.json", 'w') as outfile: json.dump(metrics[epoch], outfile)

    torch.save(model.state_dict(),epath + "model.pt")

    pprint.pprint(metrics[epoch])

###################################

line_break()

print("Done!")

line_break()

###################################
