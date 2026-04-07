# Preprocessing Module (`preprocessing.py`)

The `preprocessing.py` module is responsible for loading, cleaning, encoding, and scaling the data (specifically tailored to handle the UNSW-NB15 dataset structure and oddities) before passing it to the machine learning algorithms.

### 1. Variables & Dependencies
The script relies on `pandas` and `numpy` for manipulation, and `sklearn.preprocessing` for algorithms. It defines constants defining the target categorical variables (`proto`, `service`, `state`) and standardizes where the trained AI artifacts get stored (`models/scaler.pkl` and `models/label_encoders.pkl`).

### 2. `load_data(filepath)`
Reads the raw data from a `.csv` path on disk into Pandas.
- **Feature Check**: Confirms the file exists.
- **Robust Encoding Logic**: Initially attempts to read the file with default UTF-8 settings. Standard UNSW-NB15 sets are sometimes historically saved in `latin-1` encodings, so the function catches any `UnicodeDecodeError` and forces a fallback interpretation automatically.
- **Validation**: Checks if parsing yielded an empty dataframe and raises an error if true.

### 3. `clean_data(df)`
Pulls out bad or useless data so algorithms don't throw numeric errors.
- **Column Dropping**: Deletes purely administrative variables like `id` and text-heavy unused labels like `attack_cat`.
- **Special Formatting (`ct_ftp_cmd`)**: The `ct_ftp_cmd` field in NB15 datasets often evaluates to empty strings/spaces. This block converts any regex space character `^\s*$` to true empty (`np.nan`), replaces it with a hard `0`, and ensures it scales back to a pure `int64` column preventing text-comparison bugs.
- **Numerical Imputation**: Fills all missing numeric arrays by finding the column median.
- **Categorical Imputation**: Replaces any empty string/missing data in `object`/string columns with a simple hardcoded dash (`"-"`).

### 4. `encode_categoricals(df, training=True)`
Machine learning matrices can only read numbers natively, so words (e.g., TCP, UDP, FTP) must be mapped to distinct integers safely using `LabelEncoder`.
- **When `training=True`**: For each categorical column, the algorithm adds `"-"` explicitly and fits the `LabelEncoder`. This trains the encoder to definitively map our previously imputed `"-"` string to an integer! These trained text-to-integer mappings are stored safely to `label_encoders.pkl`.
- **When `training=False`**: Instead of fitting random objects, it reads the saved `.pkl` dict from disk. It runs an intelligent string check map. If it encounters a completely weird IP protocol in live production that it has never seen (unseen category instance), it forcefully falls it back to the `"-"` classification mapping guaranteeing the API pipeline does not crash in inference mode.

### 5. `scale_features(df, training=True)`
Algorithms like Isolation Forest prefer datasets that operate on unified scales (Z-score distributions) rather than disparate ranges like `[0 - 1]` alongside `[1000 - 99000]`.
- Determines exactly which columns hold strictly numerical dtypes while actively ensuring `label` (if in the testing file) does not get improperly scaled causing information leakage.
- **When `training=True`**: Learns the means and variances from the training chunk, scales the values within the active `df`, and persists these exact constants to `scaler.pkl`.
- **When `training=False`**: Retries the `scaler.pkl` file, verifying it runs `.transform()` across identically matched numbers.

### 6. `prepare_features(df)`
The final separation wrapper immediately prior to feeding data into a model pipeline hook. 
- Searches for `label`. If present (supervised training data), copies the column to a $y$ vector entirely stripped from the $X$ core DataFrame. 
- If running under FastAPI inference (no `label` exists), it effectively returns the `(X, None)` format protecting prediction scripts.
