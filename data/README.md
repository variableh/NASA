# Data Directory

- `raw/`: Raw C-MAPSS dataset files. This project uses the **FD001** subset (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) plus the challenge paper (`Damage Propagation Modeling.pdf`). Download separately from the [NASA Prognostics Data Repository](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) — files are gitignored.
- `processed/`: Cleaned, feature-engineered, and preprocessed datasets ready for modeling, exported by `01_analysis.ipynb` and consumed by Notebooks 02 and 03.
