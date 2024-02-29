import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer,Sequential,TransformerConv,ResGatedGraphConv,TopKPooling
from torch.nn import Sequential as Seq
from torch_scatter import scatter_mean
from torch_geometric.nn import DataParallel
import torch_geometric.nn as gnn
import time
import numpy as np

class EBM(torch.nn.Module):

	def __init__(self, setup,in_channels):
		super(EBM, self).__init__()

		self.pipe = nn.ModuleList()
		self.layers = setup

		prev = {"0":in_channels}

		for i,layer in enumerate(setup):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.pipe.append(nn.Linear(up_stream + int(params["adversarial_input"] and 0 not in shelf), params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "ReLU": 
				self.pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "batchNorm1D":
				self.pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream	
			elif name == "layerNorm":
				self.pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
			elif name == "GraphNorm":
				self.pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream
			elif name == "Sigmoid":
				self.pipe.append(nn.Sigmoid())
				prev[str(i+1)] = up_stream

	def forward(self, x):

		out = [x]

		hit_counter = {}

		for layer in self.layers:
			name,shelf = layer[0]
			for k in shelf:
				if str(k) in hit_counter:
					temp = hit_counter[str(k)]
					hit_counter[str(k)] = [0,temp[1] + 1]
				else: hit_counter[str(k)] = [0,1]

		for i,m in enumerate(self.pipe):
			kill = []

			name,shelf = self.layers[i][0]
			into = out[shelf[0]]

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			for res in shelf[1:]:
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			if name in ["Linear"]:
				params_linear = self.layers[i][1]

				if params_linear['adversarial_input'] and 0 not in self.layers[i][0][1]: 
					into = torch.cat((into,torch.unsqueeze(x[:,-1],dim=1)),dim=1)

				out+=[m(into)]
				
			elif name in ["ReLU","dropout","batchNorm1D","layerNorm","Sigmoid"]: out+=[m(into)]
			elif name == "conv1d":
				into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
				out+=[torch.permute(torch.squeeze(m(into),0), (1,0))]
				
			for r in kill: out[r] = None

		return out[-1]

class NeuralNetwork(torch.nn.Module):

	def __init__(self, setup,in_channels):
		super(NeuralNetwork, self).__init__()

		self.pipe = nn.ModuleList()
		self.layers = setup

		prev = {"0":in_channels}

		for i,layer in enumerate(setup):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "ReLU": 
				self.pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "batchNorm1D":
				self.pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream	
			elif name == "layerNorm":
				self.pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
			elif name == "GraphNorm":
				self.pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream

	def forward(self, x):

		out = [x]

		hit_counter = {}

		for layer in self.layers:
			name,shelf = layer[0]
			for k in shelf:
				if str(k) in hit_counter:
					temp = hit_counter[str(k)]
					hit_counter[str(k)] = [0,temp[1] + 1]
				else: hit_counter[str(k)] = [0,1]

		for i,m in enumerate(self.pipe):
			kill = []

			name,shelf = self.layers[i][0]
			into = out[shelf[0]]

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			for res in shelf[1:]:
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			if name in ["Linear","ReLU","dropout","batchNorm1D","layerNorm"]: out+=[m(into)]
			elif name == "GraphNorm": out += [m(into,batch=eb)] 
			elif name == "conv1d":
				into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
				out+=[torch.permute(torch.squeeze(m(into),0), (1,0))]
				
			for r in kill: out[r] = None

		return out[-1]


class EnergyModel(torch.nn.Module):

	def __init__(self, setup,in_channels):
		super(EnergyModel, self).__init__()

		self.pipe = nn.ModuleList()
		self.layers = setup

		prev = {"0":in_channels+1}

		for i,layer in enumerate(setup):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "ReLU": 
				self.pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "batchNorm1D":
				self.pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream	
			elif name == "layerNorm":
				self.pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
			elif name == "GraphNorm":
				self.pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream

	def forward(self, x, truth_scores,adversarial_scores):

		output_logits = []
		trials = []

		if truth_scores is not None: trials += [torch.cat([x,torch.unsqueeze(truth_scores,dim=1)],dim=1)]
		
		for score in adversarial_scores:
			trials += [torch.cat([x,torch.unsqueeze(score,dim=1)],dim=1)]

		for in_x in trials:

			out = [in_x]

			hit_counter = {}

			for layer in self.layers:
				name,shelf = layer[0]
				for k in shelf:
					if str(k) in hit_counter:
						temp = hit_counter[str(k)]
						hit_counter[str(k)] = [0,temp[1] + 1]
					else: hit_counter[str(k)] = [0,1]

			for i,m in enumerate(self.pipe):
				kill = []

				name,shelf = self.layers[i][0]
				into = out[shelf[0]]

				hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
				if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

				for res in shelf[1:]:
					into = torch.cat((into,out[res]),dim=1)

					hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
					if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

				if name in ["Linear","ReLU","dropout","batchNorm1D","layerNorm"]: out+=[m(into)]
				elif name == "GraphNorm": out += [m(into,batch=eb)] 
				elif name == "conv1d":
					into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
					out+=[torch.permute(torch.squeeze(m(into),0), (1,0))]
					
				for r in kill: out[r] = None

			output_logits += [out[-1]]

			#print("Out:",out[-1].shape)

		#print(output_logits)

		return torch.stack(output_logits)
	

class MergeModel(torch.nn.Module):

	def __init__(self,setup,f_x,f_e):
		super(MergeModel, self).__init__()

		self.pipe = nn.ModuleList()
		self.layers = setup

		prev = {"0":f_x + f_e}

		for i,layer in enumerate(setup):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "ReLU": 
				self.pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "batchNorm1D":
				self.pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream	
			elif name == "layerNorm":
				self.pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
			elif name == "GraphNorm":
				self.pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream
			elif name == "GlobalAttention":
				gate_nn = NeuralNetwork(params["gate_nn"],up_stream)
				map_nn = NeuralNetwork(params["map_nn"],up_stream)
				self.pipe.append(gnn.glob.GlobalAttention(gate_nn=gate_nn,nn=map_nn))
				
				prev[str(i+1)] = params["map_nn"][-1][1]["out_channels"]


	def forward(self, x, edge_index, edge_attr, u):

		node_embeddings = None
		global_embeddings = None

		u,nb,eb = u

		hit_counter = {}

		for layer in self.layers:
			name,shelf = layer[0]
			for k in shelf:
				if str(k) in hit_counter:
					temp = hit_counter[str(k)]
					hit_counter[str(k)] = [0,temp[1] + 1]
				else: hit_counter[str(k)] = [0,1]
		
		row, col = edge_index

		merge_in = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))

		merge_in = torch.cat([merge_in,x], dim=1)

		out = [merge_in]

		for i,m in enumerate(self.pipe):
			kill = []

			name,shelf = self.layers[i][0]
			into = out[shelf[0]]

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			for res in shelf[1:]:
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			if name in ["Linear","ReLU","dropout","batchNorm1D","layerNorm"]: out+=[m(into)]
			elif name == "GraphNorm": out += [m(into,batch=nb)] 
			elif name == "GlobalAttention": 
				
				global_embeddings = m(into,batch=nb)
				node_embeddings = into
				
				out += [global_embeddings]

			elif name == "conv1d":

				into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
				
				into = torch.split(into,torch.flatten(u).tolist(), dim=2)

				out_temp = torch.permute(torch.squeeze(m(into[0]),0), (1,0))

				for b in into[1:]: out_temp = torch.cat((out_temp,torch.permute(torch.squeeze(m(b),0), (1,0))),dim=0)

				out+=[out_temp]

		return node_embeddings,global_embeddings,out[-1]
	

class EdgeModel(torch.nn.Module):
	def __init__(self, setup,f_x,f_e):
		super(EdgeModel, self).__init__()

		self.pipe = nn.ModuleList()
		self.layers = setup

		prev = {"0":2*f_x + f_e}

		for i,layer in enumerate(setup):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "ReLU": 
				self.pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
			elif name == "batchNorm1D":
				self.pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream	
			elif name == "layerNorm":
				self.pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
			elif name == "GraphNorm":
				self.pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream

	def forward(self, src, dest, edge_attr, u, batch):

		out = [torch.cat([src, dest, edge_attr], 1)]

		u,nb,eb = u

		hit_counter = {}

		start_E = time.time()

		for layer in self.layers:
			name,shelf = layer[0]
			for k in shelf:
				if str(k) in hit_counter:
					temp = hit_counter[str(k)]
					hit_counter[str(k)] = [0,temp[1] + 1]
				else: hit_counter[str(k)] = [0,1]

		for i,m in enumerate(self.pipe):
			kill = []

			name,shelf = self.layers[i][0]
			into = out[shelf[0]]

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			for res in shelf[1:]:
				#if out[res].requires_grad: out[res].retain_grad()	
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			if name in ["Linear","ReLU","dropout","batchNorm1D","layerNorm"]: out+=[m(into)]
			elif name == "GraphNorm": out += [m(into,batch=eb)] 
			elif name == "conv1d":
				into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
				out+=[torch.permute(torch.squeeze(m(into),0), (1,0))]
				
			for r in kill: out[r] = None

		return out[-1]

class NodeModel(torch.nn.Module):
	def __init__(self, setup, f_x, f_e):
		super(NodeModel, self).__init__()

		prev = {"0": f_x + f_e}
	
		self.top_pipe = nn.ModuleList()
		self.bottom_pipe = nn.ModuleList()
		
		self.layers = setup

		#top half
		for i,layer in enumerate(setup["top"]):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.top_pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1)] = params["out_channels"]
				if params["midway"] == True: prev[str(i+1)] += f_x
			elif name == "ReLU": 
				self.top_pipe.append(nn.ReLU())
				prev[str(i+1)] = up_stream
			elif name == "dropout":
				self.top_pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1)] = up_stream
			elif name == "conv1d":
				self.top_pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1)] = params["out_channels"]
				if params["midway"] == True: prev[str(i+1)] += f_x
			elif name == "batchNorm1D":
				self.top_pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1)] = up_stream
				if params["midway"] == True: prev[str(i+1)] += f_x
			elif name == "layerNorm":
				self.top_pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1)] = up_stream
				if params["midway"] == True: prev[str(i+1)] += f_x
			elif name == "GraphNorm":
				self.top_pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1)] = up_stream
				if params["midway"] == True: prev[str(i+1)] += f_x		

		#bottom half
		for i,layer in enumerate(setup["bottom"]):
			name,shelf = layer[0]			
			params = layer[1]

			up_stream = sum(prev[str(res)] for res in shelf)

			if name == "Linear":
				self.bottom_pipe.append(nn.Linear(up_stream, params["out_channels"]))
				prev[str(i+1+len(setup["top"]))] = params["out_channels"]
			elif name == "ReLU": 
				self.bottom_pipe.append(nn.ReLU())
				prev[str(i+1+len(setup["top"]))] = up_stream
			elif name == "dropout":
				self.bottom_pipe.append(nn.Dropout(p=params["p"]))
				prev[str(i+1+len(setup["top"]))] = up_stream
			elif name == "conv1d":
				self.bottom_pipe.append(nn.Conv1d(up_stream, params["out_channels"], params["kernel"], padding=params["padding"], dilation=params["dilation"],bias=params["bias"]))
				prev[str(i+1+len(setup["top"]))] = params["out_channels"]
			elif name == "batchNorm1D":
				self.bottom_pipe.append(nn.BatchNorm1d(up_stream,affine=params["affine"]))
				prev[str(i+1+len(setup["top"]))] = up_stream
			elif name == "layerNorm":
				self.bottom_pipe.append(nn.LayerNorm(up_stream,elementwise_affine=params["affine"]))
				prev[str(i+1+len(setup["top"]))] = up_stream
			elif name == "GraphNorm":
				self.bottom_pipe.append(gnn.GraphNorm(up_stream))
				prev[str(i+1+len(setup["top"]))] = up_stream


	def forward(self, x, edge_index, edge_attr, u, batch):

		u,nb,eb = u

		start_N = time.time()

		row, col = edge_index

		out = [torch.cat([x[row], edge_attr], dim=1)]	

		hit_counter = {}

		for part in ["top","bottom"]:
			for layer in self.layers[part]:
				name,shelf = layer[0]
				
				for k in shelf:
				
					if str(k) in hit_counter:
						temp = hit_counter[str(k)]
						hit_counter[str(k)] = [0,temp[1] + 1]
					else: hit_counter[str(k)] = [0,1]

		for i,m in enumerate(self.top_pipe):

			name,shelf = self.layers["top"][i][0]

			kill = []

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			into = out[shelf[0]]

			for res in shelf[1:]:
				#if out[res].requires_grad: out[res].retain_grad()	
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			start = time.time()

			if name in ["Linear","ReLU","dropout","batchNorm1D","layerNorm"]: out+=[m(into)]
			elif name in "GraphNorm": out += [m(into,batch=eb)]

			for r in kill: out[r] = None

		out[-1] = scatter_mean(out[-1], col, dim=0, dim_size=x.size(0))

		out[-1] = torch.cat([x, out[-1]], dim=1)

		for i,m in enumerate(self.bottom_pipe):

			name,shelf = self.layers["bottom"][i][0]

			kill = []

			hit_counter[str(shelf[0])] = [hit_counter[str(shelf[0])][0]+1,hit_counter[str(shelf[0])][1]]
			if hit_counter[str(shelf[0])][0] == hit_counter[str(shelf[0])][1]: kill += [shelf[0]]

			into = out[shelf[0]]

			for res in shelf[1:]:
				if out[res].requires_grad: out[res].retain_grad()	
				into = torch.cat((into,out[res]),dim=1)

				hit_counter[str(res)] = [hit_counter[str(res)][0]+1,hit_counter[str(res)][1]]
				if hit_counter[str(res)][0] == hit_counter[str(res)][1]: kill += [res]

			start = time.time()

			if name in ["Linear","ReLU","dropout","layerNorm"]: 
				out+=[m(into)]
			elif name in "GraphNorm": out += [m(into,batch=nb)]
			elif name == "conv1d" or "batchNorm1D":
				
				into = torch.permute(torch.unsqueeze(into,0), (0,2,1))
				
				if self.layers["bottom"][i][1]["batched"]:
					into = torch.split(into,torch.flatten(u).tolist(), dim=2)

					out_temp = torch.permute(torch.squeeze(m(into[0]),0), (1,0))

					for b in into[1:]: out_temp = torch.cat((out_temp,torch.permute(torch.squeeze(m(b),0), (1,0))),dim=0)

					out+=[out_temp]
				else:
					out+=[torch.permute(torch.squeeze(m(into),0), (1,0))]

			for r in kill: out[r] = None

		return out[-1]

class GraphModel(torch.nn.Module):

	def __init__(self,setup):
		super(GraphModel, self).__init__()

		size,layers = setup

		self.pipe = nn.ModuleList()

		node_size,edge_size = size

		prev_n = {"0":node_size}
		prev_e = {"0":edge_size}

		up_stream_final = 0

		self.layers = layers

		self.input_shape = size

		for i,layer in enumerate(layers):
			name,shelf = layer[0]

			up_stream_n = sum(prev_n[str(res)] for res in shelf[0])
			up_stream_e = sum(prev_e[str(res)] for res in shelf[1])

			if name == "transformer":				
				#out,heads,cat,beta,drop,Attention_Params

				l = TransformerConv(in_channels=up_stream_n,out_channels=layer[1], heads=layer[2], concat=layer[3], beta=layer[4], dropout=layer[5], edge_dim=up_stream_e)

				if layer[3]==True:prev_n[str(i+1)] = layer[1]*layer[2] #new node size is out_channels*heads
				else: prev_n[str(i+1)] = layer[1]

				
				if layer[6] == 'multiply': prev_e[str(i+1)] = up_stream_e*layer[2]#*attention_heads
				elif layer[6] == 'replace': prev_e[str(i+1)] = layer[2] #attention_heads	
				elif layer[6] == 'cat':prev_e[str(i+1)] = up_stream_e+layer[2] #+attention_heads
				elif layer[6] == 'none': prev_e[str(i+1)] = up_stream_e


				self.pipe.append(l)

			elif name == "MetaLayer":

				prev_e[str(i+1)] = layer[1][-1][1]["out_channels"]

				l = None
				if layer[2]["None"] == False:
					l = MetaLayer(edge_model=EdgeModel(layer[1],up_stream_n,up_stream_e), node_model=NodeModel(layer[2],up_stream_n,prev_e[str(i+1)]), global_model=None)
					prev_n[str(i+1)] = layer[2]["bottom"][-1][1]["out_channels"]
				else:
					l = MetaLayer(edge_model=EdgeModel(layer[1],up_stream_n,up_stream_e), node_model=None, global_model=None)
					prev_n[str(i+1)] = up_stream_n

				self.pipe.append(l)

			elif name == "TopKPool":
				l = TopKPooling(up_stream_n,ratio=layer[1])

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

				self.pipe.append(l)

			elif name == "MergeModel":

				self.pipe.append(MergeModel(layer[1],up_stream_n,up_stream_e))
				
				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

				up_stream_final = layer[1][-1][1]["out_channels"]

			elif name == "EnergyModel":

				self.pipe.append(EnergyModel(layer[1],up_stream_final))



			elif name == "linearN":

				self.pipe.append(nn.Linear(up_stream_n, layer[1]))
				
				prev_n[str(i+1)] = layer[1]
				prev_e[str(i+1)] = up_stream_e

			elif name == "linearE":

				self.pipe.append(nn.Linear(up_stream_e, layer[1]))

				prev_n[str(i+1)] = layer[1]
				prev_e[str(i+1)] = params["out_channels"]
	
			elif name == "LayerNormN":

				self.pipe.append(gnn.LayerNorm(up_stream_n,affine=layer[1]))
				
				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e


			elif name == "LayerNormE":

				self.pipe.append(gnn.LayerNorm(up_stream_e,affine=layer[1]))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			
			elif name == "BatchNorm1dE":

				self.pipe.append(nn.BatchNorm1d(up_stream_e,affine=layer[1]))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "BatchNorm1dN":

				self.pipe.append(nn.BatchNorm1d(up_stream_n,affine=layer[1]))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "GraphNormN":

				self.pipe.append(gnn.GraphNorm(up_stream_n))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "GraphNormE":

				self.pipe.append(gnn.GraphNorm(up_stream_e))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e
			
			elif name == "PairNormN":
				self.pipe.append(gnn.PairNorm(scale=layer[1],scale_individually=layer[2]))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "PairNormE":

				self.pipe.append(gnn.PairNorm(scale=layer[1],scale_individually=layer[2]))

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "ReLU":
				self.pipe.append(nn.ReLU())

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

			elif name == "tanh":
				self.pipe.append(nn.Tanh())

				prev_n[str(i+1)] = up_stream_n
				prev_e[str(i+1)] = up_stream_e

	def forward(self,batched_ex,batch_id,truth_scores=None,adversarial_scores=None):

		node_embeddings = None
		global_embeddings = None

		#construct large graph

		esm_flag = False
		esm_stats_flag = False

		if self.input_shape[0] == 1371:
			esm_flag = True
			esm_stats_flag = True
		elif self.input_shape[0] == 1367: esm_flag = True
		elif self.input_shape[0] == 87: esm_stats_flag = True

		curr_node_i = 0

		node_features = None
		edge_features = None
		edge_i = None

		node_batch = None
		edge_batch = None

		nl = []
		el = []

		for example in batched_ex:

			if example.edge_index.device==batch_id.device: 

				node_len = 0 if node_features is None else len(node_features)

				curr_node_features = example.node_features

				if esm_flag: curr_node_features = torch.cat((curr_node_features,example.esm),dim=1)
				if esm_stats_flag: curr_node_features = torch.cat((curr_node_features,example.esm_stats),dim=1)

				if node_features is None: node_features = curr_node_features
				else: node_features = torch.cat((node_features,curr_node_features))

				nl += [len(example.node_features)]

				if edge_features is None: edge_features = example.edge_attr
				else: edge_features = torch.cat((edge_features,example.edge_attr))

				el += [len(example.edge_attr)]
				
				if edge_i is None: edge_i = example.edge_index
				else: edge_i = torch.cat((edge_i,torch.add(example.edge_index,node_len)))

		edge_i = torch.permute(edge_i,(1,0))

		batch = torch.tensor(list(range(len(nl)))) #len is number of elements in this model
		el = torch.tensor(el)
		nl = torch.tensor(nl)

		eb = torch.repeat_interleave(batch,el).to(batch_id.device)
		nb = torch.repeat_interleave(batch,nl).to(batch_id.device)

		#generate dictionary that indicates number of uses within a model
		#helps with memory by removing unnecessary data in wire memory
		hit_counter_n = {}
		hit_counter_e = {}
		
		for component in self.layers:
			name,shelfs = component[0]
			n_shelf,e_shelf = shelfs
			
			for k_n in n_shelf:
				if str(k_n) in hit_counter_n: hit_counter_n[str(k_n)] = [0,hit_counter_n[str(k_n)][1]+1]
				else: hit_counter_n[str(k_n)] = [0,1]

			for k_e in e_shelf:
				if str(k_e) in hit_counter_e: hit_counter_e[str(k_e)] = [0,hit_counter_e[str(k_e)][1]+1]
				else: hit_counter_e[str(k_e)] = [0,1]
		
		#propogate singal through the pipeline

		out = [[node_features,edge_features]]

		#batch = batch.to(batch_id.device)

		final_out = None

		for i,m in enumerate(self.pipe):

			kill_n = []
			kill_e = []

			name = self.layers[i][0][0]
			shelf = self.layers[i][0][1]

			into_n = out[shelf[0][0]][0]
			into_e = out[shelf[1][0]][1]
		
			hit_counter_n[str(shelf[0][0])] = [hit_counter_n[str(shelf[0][0])][0]+1,hit_counter_n[str(shelf[0][0])][1]]
			if hit_counter_n[str(shelf[0][0])][0] == hit_counter_n[str(shelf[0][0])][1]: kill_n += [shelf[0][0]]

			hit_counter_e[str(shelf[1][0])] = [hit_counter_e[str(shelf[1][0])][0]+1,hit_counter_e[str(shelf[1][0])][1]]
			if hit_counter_e[str(shelf[1][0])][0] == hit_counter_e[str(shelf[1][0])][1]: kill_e += [shelf[1][0]]

			#node residual input wires	
			for res in shelf[0][1:]:	
				into_n = torch.cat((into_n,out[res][0]),dim=1)

				hit_counter_n[str(res)] = [hit_counter_n[str(res)][0]+1,hit_counter_n[str(res)][1]]
				if hit_counter_n[str(res)][0] == hit_counter_n[str(res)][1]: kill_n += [res]

			#edge residual input wires
			for res in shelf[1][1:]:
				into_e = torch.cat((into_e,out[res][1]),dim=1)

				hit_counter_e[str(res)] = [hit_counter_e[str(res)][0]+1,hit_counter_e[str(res)][1]]
				if hit_counter_e[str(res)][0] == hit_counter_e[str(res)][1]: kill_e += [res]
		
			start = time.time()
			
			#data passage
			if name == "transformer":

				node_update,edge_weights = m(x=into_n,edge_index=edge_i,edge_attr=into_e,return_attention_weights=True)
				ei_update, edge_weights = edge_weights

				edge_weight_config = self.layers[i][6]

				if edge_weight_config == 'replace': into_e = edge_weights
				elif edge_weight_config == 'cat': into_e = torch.cat((edge_weights,into_e),dim=1)
				elif edge_weight_config == 'multiply': 
					into_e = torch.einsum("bi,bj->bij",edge_weights,into_e)
					into_e = torch.flatten(into_e, start_dim=1, end_dim=- 1)

				out += [[node_update,into_e]]
		
			elif name == "TopKPool":

				node_update,edge_index_update,edge_attr_update,batch_index_update,_,_ = m(x=into_n,edge_index=edge_i,edge_attr=into_e,batch=nb)

				#print("\n",node_update.shape,edge_index_update.shape,edge_attr_update.shape,batch_index_update.shape)
				#print("\n",into_n.shape,edge_i.shape,into_e.shape,nb.shape,eb.shape)
				#print("\n",torch.bincount(batch_index_update))
				#print(node_update,edge_attr_update,into_e)

				#Update the graph to the pruned graph
				out += [[node_update,edge_attr_update]]
				edge_i = edge_index_update
				nb = batch_index_update

				#NEED TO UPDATE NL/EB HERE BUT NOT NECCESARY FOR NOW...

			elif name == "MetaLayer":

				node,edge,glob = m(x=into_n,edge_index=edge_i,edge_attr=into_e,u=(nl,nb,eb))

				out += [[node,edge]]
		
			elif name == "MergeModel":
				
				#print("\n",nb)

				#print("\nYAYAYYAYAYA")

				node_embeddings,global_embeddings,final_out = m(x=into_n,edge_index=edge_i,edge_attr=into_e,u=(nl,nb,eb))

				out += [[into_n,into_e]]

			elif name == "EnergyModel":
				
				final_out = m(final_out,truth_scores,adversarial_scores)

			elif name == "LayerNormN": out += [[m(into_n,batch=nb),into_e]]
			elif name == "LayerNormE": out += [[into_n,m(into_e,batch=eb)]]
			
			elif name == "linearN": out += [[m(into_n),into_e]]	
			elif name == "linearE": out += [[into_n,m(into_e)]]
			
			elif name == "BatchNorm1dE": out += [[into_n,m(into_e)]] 
			elif name == "BatchNorm1dN": out += [[m(into_n),into_e]]
			
			elif name == "GraphNormE": out += [[into_n,m(into_e,batch=eb)]]
			elif name == "GraphNormN": out += [[m(into_n,batch=nb),into_e]]
			
			elif name == "PairNormE": out += [[into_n,m(into_e,batch=eb)]]
			elif name == "PairNormN": out += [[m(into_n,batch=nb),into_e]]
			elif name == "ReLU":
				n = into_n
				e = into_e

				if "n" in self.layers[i][1]: n = m(n)
				if "e" in self.layers[i][1]: e = m(e)

				out += [[n,e]]

			elif name == "tanh":
				n = into_n
				e = into_e

				if "n" in self.layers[i][1]: n = m(n)
				if "e" in self.layers[i][1]: e = m(e)

				out += [[n,e]]

			for r in kill_n: out[r][0] = None
			for r in kill_e: out[r][1] = None

		return node_embeddings,global_embeddings,final_out #final_out #,index_order #out[-1]












