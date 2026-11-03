# src/training

**Status: to do.** This is the next step.

Will hold `train.py`, which trains one model on every fold of a split and saves the scores.

Planned command:

```bash
python -m src.training.train --preset lstm_base --T 20 --seed 0
```

What it will do for each of the 5 folds:
1. Load the features (`load_features(T)`) and the split (`load_split(seed)`).
2. Build the model by name and train it on the train part, using the recipe in
   `configs/lstm.yaml` or `configs/transformer.yaml`.
3. Stop training when validation AUROC stops improving.
4. Score it once on the test part and write `results/<preset>_T<T>_s<seed>_f<fold>/`.
