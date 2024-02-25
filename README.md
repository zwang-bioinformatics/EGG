# EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

## Evaluations
Run the following command to generate and evaluate blind-test predictions reported in the original EGG paper. 
* model config (-m)
  * Regression-Transformer: ./configs/
  * Regression-MetaLayer: ./configs/
  * EBM-Transformer: ./configs/
  * EBM-MetaLayer: ./configs/
* output directory (-o)
```
python run_blindtest.py -m ./configs/config.json -o ./results/
```
## Training
Run the following commands to re-train the model architectures (EBM or Regression Based Methods), reported in the original EGG paper. 

## Citation
Andrew Jordan Siciliano, Chenguang Zhao, Tong Liu, and Zheng Wang.
EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

