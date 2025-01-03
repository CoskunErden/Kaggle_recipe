# Recipe Traffic Classifier

This project aims to classify recipes into "High Traffic" or "Low Traffic" categories based on their nutritional content and other features. It leverages machine learning techniques to predict the likelihood of high user engagement with the recipes.

---

## Project Overview

### Objectives
- Predict whether a recipe will have high traffic based on its features.
- Compare the performance of various machine learning models including Logistic Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks.
- Analyze feature importance to identify factors influencing recipe traffic.

### Dataset
The dataset contains nutritional and categorical information for various recipes. Key features include:
- **Calories, Carbohydrates, Sugar, Protein**: Nutritional values.
- **Category**: Type of dish (e.g., Breakfast, Dessert, Vegetables, etc.).
- **Servings**: Number of servings.

---

## Workflow and Methodology

### Data Preprocessing
1. **Handling Missing Values**: Imputed or removed missing data points.
2. **Scaling**: Standardized numerical features using Power Transformation and StandardScaler.
3. **Encoding**: Applied one-hot encoding to categorical features for machine learning compatibility.

### Model Building
Trained and evaluated six machine learning models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Support Vector Machine (SVM)**
6. **Neural Networks**

### Metrics Used
- Precision
- Accuracy
- Recall
- F1 Score
- ROC AUC Score

---

## Results

### Model Performance
| Model                  | Precision | Accuracy | Recall | F1 Score | ROC AUC Score |
|------------------------|-----------|----------|--------|----------|---------------|
| Logistic Regression    | 0.803419  | 0.766497 | 0.803419 | 0.803419 | 0.836432      |
| Random Forest          | 0.707143  | 0.700588 | 0.846154 | 0.770428 | 0.784669      |
| K-Nearest Neighbor     | 0.601351  | 0.558376 | 0.760684 | 0.671698 | 0.534028      |
| Support Vector Machine | 0.803419  | 0.766497 | 0.803419 | 0.803419 | 0.822222      |
| Gradient Boosting      | 0.725926  | 0.715736 | 0.837607 | 0.777778 | 0.823130      |
| Neural Network         | 0.803571  | 0.751269 | 0.769231 | 0.786026 | 0.791560      |

### Insights
- Logistic Regression and Gradient Boosting models demonstrate the best overall performance with high precision and ROC AUC scores.
- The KNN model underperformed compared to other classifiers, indicating it may not be suitable for this dataset.
- The feature "Protein" was consistently identified as the most important predictor across multiple models.

---

## Visualizations

1. **Feature Importance**: Visualized the key features contributing to model predictions.
2. **Confusion Matrices**: Displayed model performance in terms of true/false positives and negatives.
3. **ROC Curves**: Analyzed the trade-off between sensitivity and specificity.

---

## Conclusions

- **Model Selection**: Logistic Regression and Gradient Boosting are ideal for this classification problem.
- **Feature Importance**: High protein content is a strong indicator of high recipe traffic.
- **Future Work**: Experiment with additional features (e.g., preparation time, ingredients) and advanced deep learning techniques.

---

## Setup and Usage

### Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

### Instructions
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to explore the analysis and model performance.

---

## Acknowledgments
This project was inspired by the growing need to understand user preferences and improve recipe recommendations. Special thanks to Kaggle.com for providing the dataset.

