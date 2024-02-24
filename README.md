# EGG: Accuracy Estimation of Individual Multimeric Protein Models using Deep Energy-Based Models and Graph Neural Networks

## Evaluations
Run the following command to run the evaluations performed and reported in the original EGG paper. 

* model config (-m)
  * Regression-Transformer: ./configs/
  * Regression-MetaLayer: ./configs/
  * EBM-Transformer: ./configs/
  * EBM-MetaLayer: ./configs/
```
python run_blindtest.py -m ./configs/config.json -t EBM -s TM -o ./results/
```

## Training 

```
python train_model.py -m "./configs/config.json" -t "EBM" -s TM -o "./output_directory/"
```
