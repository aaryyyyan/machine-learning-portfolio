# Aryan Giri – Machine Learning Projects

Three end-to-end projects built with Python and scikit-learn. Every number below comes from running the scripts (random seed 42), so you can reproduce it.

| Project | Script | Key result |
|---|---|---|
| Breast Cancer Diagnosis Prediction | `breast_cancer_diagnosis.py` | SVM: 97.4% accuracy, 100% precision, 92.9% recall, 0.993 ROC-AUC on unseen data |
| Customer Segmentation (RFM + K-Means) | `customer_segmentation_kmeans.py` | k = 4, silhouette 0.435, ARI 0.995 across seeds; Champions (21.7% of customers) = 58.8% of revenue |
| Handwritten Digit Recognition | `digit_recognition_neural_network.py` | SVM 97.5% (chosen by CV); tuned MLP 98.1% on test |

## Run
```
pip install -r requirements.txt
python breast_cancer_diagnosis.py
python customer_segmentation_kmeans.py
python digit_recognition_neural_network.py
streamlit run app.py          # demo app for the breast cancer model
```
Each script saves figures, the trained model and `results.json` in an output folder.

## Method highlights
- Stratified train/test split; scaling inside Pipelines to prevent data leakage
- Model comparison with 5-fold cross-validation, tuning with GridSearchCV
- Final evaluation only on untouched test data; error analysis and feature importance

## Honest notes
- Customer segmentation uses **synthetic** transaction data (seeded generator); pass your own CSV (`customer_id,date,amount`) to use real data.
- Breast cancer and digits use datasets bundled with scikit-learn. This is a learning project, not a medical tool.
