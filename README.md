# Online Shoppers Purchase Intention - Classification Models and Streamlit App

# BITS ID = 2026AJ05849
# Name = Girish Mahimaiah

## a. Problem statement
Predict whether an e-commerce website visit (session) ends in a purchase, using the session's browsing behaviour, page metrics, timing and visitor details. This is a **binary classification** problem (`Revenue`: 1 = purchase, 0 = no purchase). Five classifiers - Logistic Regression, Decision Tree, K-Nearest Neighbour, Gaussian Naive Bayes and Support Vector Machine - are trained on the same dataset, compared on six evaluation metrics, and served through a Streamlit web app.

## b. Dataset description
- **Name:** Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository (dataset 468), https://archive.ics.uci.edu/dataset/468/online-shoppers-purchasing-intention-dataset - donated by C. Sakar and Y. Kastro (CC BY 4.0).
- **Instances:** 12,330 sessions (minimum required: 500). No missing values.
- **Features:** 17 input features (minimum required: 12) plus the target `Revenue`.
  - Numeric (10): `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
  - Binary (1): `Weekend`
  - Categorical (6): `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`
- **Class balance:** 10,422 no-purchase (84.5%) / 1,908 purchase (15.5%) - an imbalanced problem.
- **Preprocessing (inside each saved pipeline):** numeric and `Weekend` columns are standardised with `StandardScaler`; the six categorical columns are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`. `OperatingSystems`, `Browser`, `Region` and `TrafficType` are integer codes with no numeric order, so they are treated as categories. `Weekend` and `Revenue` are converted from True/False to 1/0. No rows were removed (the file contains 125 exact duplicate rows, which were kept as supplied).
- **Split:** 80/20 stratified train/test split, `random_state=42` -> 9,864 training rows and 2,466 test rows (2,084 no-purchase, 382 purchase). The test rows are saved as `test_data.csv`.
- **Positive class** for Precision, Recall, F1 and MCC is `Revenue = 1` (purchase).

## c. GitHub repository link
  https://github.com/GirishMahimaiah03/online-shoppers-purchase-intent

## d. Models used

Hyperparameters: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), KNN (`n_neighbors=11`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `probability=True`). `random_state=42` is used where applicable. Metrics are on the 2,466-row held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8808 | 0.8830 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8990 | 0.9172 | 0.7111 | 0.5864 | 0.6428 | 0.5883 |
| KNN | 0.8735 | 0.8149 | 0.6944 | 0.3272 | 0.4448 | 0.4184 |
| Naive Bayes | 0.2733 | 0.7333 | 0.1727 | 0.9738 | 0.2934 | 0.1292 |
| Support Vector Machine | 0.8869 | 0.8829 | 0.7036 | 0.4660 | 0.5606 | 0.5127 |

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.8808 and AUC 0.8830 are good, and it has the highest precision (0.7366): when it predicts a purchase it is usually right (137 correct, 49 false alarms). Recall is low (0.3586) - it finds only 137 of the 382 buyers - so F1 (0.4824) and MCC (0.4592) are modest. A linear boundary captures the main signal but misses interactions. |
| Decision Tree | Best model on Accuracy (0.8990), AUC (0.9172), F1 (0.6428) and MCC (0.5883). It finds 224 of 382 buyers (recall 0.5864) with 91 false alarms. A depth-5 tree can capture non-linear rules and interactions between behaviour features such as page value and exit rate. |
| KNN | Accuracy (0.8735) is mostly the effect of the 84.5% majority class. It has the lowest AUC among the non-Naive-Bayes models (0.8149) and low recall (0.3272, only 125 of 382 buyers found), giving F1 0.4448 and MCC 0.4184. Distances over the many one-hot columns are likely less informative. |
| Naive Bayes | Very poor accuracy (0.2733) and precision (0.1727), but the highest recall (0.9738). It labels most sessions as purchases (2,154 of 2,466), so it catches 372 of 382 buyers at the cost of 1,782 false alarms. AUC (0.7333) and MCC (0.1292) show weak overall discrimination. The Gaussian and independence assumptions fit these skewed, correlated and one-hot features poorly. |
| Support Vector Machine | Second-best overall: Accuracy 0.8869, AUC 0.8829, F1 0.5606 and MCC 0.5127. It is more balanced than Logistic Regression and KNN (recall 0.4660, precision 0.7036), but it still misses 204 of 382 buyers and is behind the Decision Tree on every summary metric. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It has the highest Accuracy, AUC, F1 and MCC. MCC and F1 are the most reliable summaries on this imbalanced data. Logistic Regression has the highest precision and Naive Bayes the highest recall, but each is much weaker on the balanced metrics. |

## e. Streamlit app
Live app link: [<STREAMLIT_APP_LINK>](https://2026aj05849-online-shoppers-purchase-intent.streamlit.app/)

Features: (1) CSV upload of test data, (2) model selection dropdown, (3) evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), (4) confusion matrix and classification report, plus a table comparing all five models on the uploaded file. The app loads the saved models from `model/` and does not retrain. The uploaded CSV must contain the 17 feature columns and `Revenue`.

## Repository structure
```
shoppers-intention-ml-app/
|-- app.py
|-- data_utils.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- .streamlit/
    |-- config.toml
|-- assets/
    |-- style.css
|-- data/
    |-- online_shoppers_intention.csv
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
```

## How to run
```
pip install -r requirements.txt
python train_models.py      # retrains and re-saves models, writes test_data.csv
streamlit run app.py
```
