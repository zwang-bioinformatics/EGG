# EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks
Our code was implemented using Python version ~ 3.9. Please install the required packages using the following command:
```
pip install -r requirements.txt
```
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
```
Update the `init.py` file to reflect the root directory of the project, the unzipped databases, and both the `global_df.csv` and `group_mappings.json` files. 
## Evaluations
Run the following command to generate and evaluate blind-test (CASP15 Targets) predictions reported in the original EGG paper. 
Note: Minor differences can occur.
* model config (-m)
  * Regression-Transformer: ./configs/
  * Regression-MetaLayer: ./configs/
  * EBM-Transformer: ./configs/
  * EBM-MetaLayer: ./configs/
* epoch (-e)
 * setting this to `default` will revert to the weights of the models evaluated in the original EGG paper.
* output directory (-o)
```
python run_blindtest.py -m ./configs/config.json -e -1 -o ./results/
```
## Training
Run the following commands to re-train the model architectures (EBM or Regression Based Methods), reported in the original EGG paper. 

Train Regression Backbone Models:
```
python
```
Train EBMs:
```
python
```
To evaluate the newly trained models run the following command: 
```
python
```

## Citation
Andrew Jordan Siciliano, Chenguang Zhao, Tong Liu, and Zheng Wang.
EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

