# EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks
Our code was implemented using Python version ~ 3.9, PyTorch version 2.1.0, and PyTorch Geometric version 2.4.0. 
Our source code can be downloaded using either of the following commands:
```
wget http://dna.cs.miami.edu/EGG/EGG.gzip
gzip -d EGG.gzip
```
```
git clone https://github.com/zwang-bioinformatics/EGG/
```
For evaluation, please download the CASP15 blind test data (~100GB) using the following commands: 
```
wget http://dna.cs.miami.edu/EGG/EGG_blind_test_database.tar.gz
tar -xvzf EGG_blind_test_database.tar.gz
```
For training & validation please download the generated databases (~307GB) using the following commands:
```
wget http://dna.cs.miami.edu/EGG/EGG_training_database.tar.gz
tar -xvzf EGG_training_database.tar.gz
wget http://dna.cs.miami.edu/EGG/EGG_validation_database.tar.gz
tar -xvzf EGG_validation_database.tar.gz
```
Please download the CASP15 group mappings and predictions using the following commands: 
```
wget https://git.scicore.unibas.ch/schwede/casp15_ema/-/raw/main/group_mappings.json
wget https://git.scicore.unibas.ch/schwede/casp15_ema/-/raw/main/custom_analysis/global_df.csv
wget https://git.scicore.unibas.ch/schwede/casp15_ema/-/raw/main/ema_targets.json
```
Update the `init.py` file to reflect the root directory of the project, the unzipped databases, the `global_df.csv`, `group_mappings.json`, and `ema_targets.json` files. 

## Evaluations
Predictions for our models reported in the original EGG paper are pre-saved in the CSV files for convenience (see `./results/`).
If you want to reproduce our blind-test (CASP15 Targets) predictions, run the following command for each of the configs in the `./configs/` directory:
```
python generate_predictions.py -m CONFIG.json 
```
Note: this will overwrite the existing pre-saved CSV files, minor differences can occur, and the default device is cpu but that can be changed using the `-d` flag. 
Run the following command to evaluate & generate figures for the blind-test (CASP15 Targets) predictions stored in the CSV files (Note: the previous step is not nessecary to run this command): 
```
python run_eval.py
```
This will save all figures in `ROOT + "reproduced_figures/"` and print L1 and MSE losses for each of the CSV files associated with a model architecture and score type (TM or QS). 
## Training
A mock training script is provided to re-train the model architectures (EBM or Regression Based Methods), reported in the original EGG paper. Note that this script does not utilize all of the features reported in the original paper and is a simpler version for usability. To train run the following command: 
```
python train.py -m CONFIG.json -e EPOCHS -b BATCH_SIZE -d DEVICE
```
Trained models and metrics will be saved in `./models/CONFIG/epoch_XXXX/` under `model.pt` & `metrics.json`. 
The parameters associated with the models reported in the original EGG paper are under `./models/CONFIG/default/model.pt`.
## Citation
Andrew Jordan Siciliano, Chenguang Zhao, Tong Liu, and Zheng Wang.
EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

