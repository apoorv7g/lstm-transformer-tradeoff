# results

Output of the experiments. Empty until training starts.

Each training run gets its own folder, named after the model, frame count, seed and fold:

```
results/lstm_base_T20_s0_f0/
    metrics.json          AUROC, F1, sensitivity, specificity and others on the test fold
    history.csv           loss and validation AUROC per epoch
    oof_predictions.csv   predicted probability for every test video
    model.keras           trained weights (not committed)
```

Speed and memory measurements and the final summary tables and plots will also be written here.

Metrics and predictions are committed (they hold no medical content). Model weights are not.
