# EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

## Reproducing Blind Test Evaluations

```
python run_blindtest.py -m ./configs/config.json -t EBM -s TM -o ./results/
```

## Training 

```
python train_model.py -m "./configs/config.json" -t "EBM" -s TM -o "./output_directory/"
```
