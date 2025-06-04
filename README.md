# 🧠 Data_Science Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)  
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)  

**Data_Science** is a curated collection of end-to-end machine learning and deep learning projects, primarily implemented in Python (Jupyter Notebooks). Each project tackles a real-world dataset—mostly sourced from Kaggle—and demonstrates data-loading, preprocessing, exploratory data analysis, model building, training, evaluation, and visualization. Key areas include:

- **Computer Vision:** Food classification, brain tumor detection/classification  
- **Healthcare & Epidemiology:** Medical cost prediction, Pima Indians diabetes, climate-driven disease forecasting  
- **Tabular & Regression Tasks:** Social media addiction prediction, loan status prediction, e-commerce customer analysis, advertisement click prediction  
- **Natural Language Processing:** Disaster tweet classification, IMDB movie-review sentiment  
- **Exploratory Data Analysis (EDA):** 911 calls, Netflix movies & TV shows  

---

## 📋 Table of Contents

1. [📁 Repository Structure](#-repository-structure)  
2. [📂 Projects Overview](#-project-descriptions--links)  
   - [💻 Computer Vision](#-computer-vision)  
   - [🏥 Healthcare & Epidemiology](#-healthcare--epidemiology)  
   - [📊 Tabular & Regression Tasks](#-tabular--regression-tasks)  
   - [🗣️ Natural Language Processing](#-natural-language-processing)  
   - [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)  
   - [📂 Other Folders](#-other-folders)  
3. [🤝 Contributing](#-contributing)  
4. [📜 License](#-license)  
5. [📬 Contact & Acknowledgments](#-contact--acknowledgments)  

---

## 📁 Repository Structure

```text
Data_Science/
├── LICENSE
├── README.md
├── machine_learning/
│   └── supervised_learning/
│       └── [Example notebooks illustrating core supervised learning algorithms]
│
├── projects/
│   ├── advertisement_click_on_ads/
│   ├── brain_tumor/
│   ├── brain_tumor_classification/
│   ├── disease_spread/
│   ├── disaster_tweets/
│   ├── ecommerce_customers/
│   ├── emergency_911_calls/
│   ├── food_classification/
│   ├── imdb_movie_review/
│   ├── lending_club_loan_status/
│   ├── medical_cost_personal/
│   ├── netflix_movies_and_tv_shows/
│   ├── pima_indians_diabetes/
│   └── social_media_addiction/
└── sql/
    └── mysql/
```
---

## 🔗 Project Descriptions & Links

Below are all projects organized by theme. Click on each link to navigate directly to that project’s folder on GitHub. Descriptions are pulled from the CV-provided explanations, preserving detail and clarity.

---

### 💻 Computer Vision

1. **[Food Classification (10 classes)](projects/food_classification/classification_food_10.ipynb)**  
   - **Source:** Inspired by [Mr. Bourke’s TensorFlow Deep Learning Course](https://github.com/mrdbourke/tensorflow-deep-learning)  
   - **Dataset:** 10-category food images (subset of Food-101)  
   - **Key Techniques & Workflow:**  
     - Leveraged a pre-trained **EfficientNetB0** feature extractor in TensorFlow 1.x.  
     - Built and compared against a custom CNN baseline.  
     - Data loading and augmentation pipelines: random flips, rotations, scaling.  
     - Training with callbacks (EarlyStopping, ReduceLROnPlateau).  
     - Achieved high accuracy by fine-tuning model hyperparameters and augmentations.  

2. **[Food Classification (11 classes)](projects/food_classification/classification_food_11.ipynb)**  
   - **Dataset:** [Food11 (Kaggle)](https://www.kaggle.com/datasets/imbikramsaha/food11)  
   - **Key Techniques & Workflow:**  
     - Transfer learning using **EfficientNetB0**, **B1**, and **B2** backbones in TensorFlow 2.x.  
     - Data augmentation (random zooms, brightness, horizontal flips, centro-cropping).  
     - TensorBoard for real-time visualization; EarlyStopping & ModelCheckpoint callbacks.  
     - Thorough evaluation: accuracy, precision/recall, confusion matrix heatmaps.  

3. **[Brain Tumor Detection by MRI Images](projects/brain_tumor/tumor_detection.ipynb)**  
   - **Dataset:** [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
   - **Key Techniques & Workflow:**  
     - Preprocessing: resized images to 224×224, pixel normalization.  
     - Built a custom CNN (Conv→Pool→Dense) in TensorFlow/Keras.  
     - Data augmentation: flips, rotations, shifts to increase robustness.  
     - Train/validation split with stratification to ensure balanced classes.  
     - Evaluated using accuracy, ROC AUC, and created Grad-CAM saliency maps for interpretability.  

4. **[Brain Tumor Classification (4 classes)](projects/brain_tumor_classification/tumor_classification.ipynb)**  
   - **Dataset:** [Brain Tumor Classification MRI (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
   - **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
   - **Key Techniques & Workflow:**  
     - Constructed a deeper **VGG-style** CNN architecture in TensorFlow/Keras.  
     - Employed stratified train/validation/test splits to avoid class imbalance.  
     - Applied extensive data augmentation (rotation, zoom, shear).  
     - Used callbacks (EarlyStopping, ModelCheckpoint) to prevent overfitting.  
     - Reported per-class precision, recall, F1-score, and displayed confusion matrices.  

---

### 🏥 Healthcare & Epidemiology

1. **[Medical Cost Prediction](projects/medical_cost_personal/medical_cost_prediction.ipynb)**  
   - **Dataset:** [Insurance Cost Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
   - **Key Techniques & Workflow:**  
     - Data cleaning: handled missing values, detected and removed outliers.  
     - Feature engineering: one-hot encoding of categorical features (`sex`, `smoker`, `region`), BMI bucketing.  
     - Built a deep neural network in TensorFlow (two hidden layers, ReLU activations).  
     - Monitored training/validation loss curves; reported final MSE and MAE on test set.  

2. **[Pima Indians Diabetes Prediction](projects/pima_indians_diabetes/pima_diabetes_prediction.ipynb)**  
   - **Dataset:** [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
   - **Key Techniques & Workflow:**  
     - Exploratory Data Analysis (EDA): histograms, boxplots, correlation heatmaps.  
     - Trained a **K-Nearest Neighbors (KNN)** classifier.  
     - Performed hyperparameter tuning with `GridSearchCV` (searching for optimal k).  
     - Evaluated performance using ROC curve, AUC score, and confusion matrix.  

3. **[Climate-Driven Disease Spread Prediction](projects/disease_spread/disease_forecast.ipynb)**  
   - **Dataset:** [Climate-Driven Disease Spread (Kaggle)](https://www.kaggle.com/datasets/hopeofchange/climate-driven-disease-spread)  
   - **Target:** Forecast malaria and dengue case counts from climate and socio-economic variables  
   - **Key Techniques & Workflow:**  
     - Data cleaning: imputed missing meteorological values, checked for outliers.  
     - Feature engineering: created lag features (monthly temperature/rainfall), rolling averages.  
     - Built a feed-forward neural network in TensorFlow (Dense layers with ReLU).  
     - Tracked training/validation RMSE and MAE; plotted residuals to check for bias.  

---

### 📊 Tabular & Regression Tasks

1. **[Students’ Social Media Addiction Prediction](projects/social_media_addiction/social_media_addiction_prediction.ipynb)**  
   - **Dataset:** [Social Media Addiction vs. Relationships (Kaggle)](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships/code)  
   - **Key Techniques & Workflow:**  
     - Data preprocessing: filled missing demographics, scaled numerical usage metrics (min–max scaling).  
     - Feature selection via correlation matrix and mutual information scores.  
     - Trained and compared **Random Forest** and **Bayesian Ridge** models to identify at-risk students.  
     - Visualized feature importance to highlight top predictors of addiction.  

2. **[Lending Club Loan Status Prediction](projects/lending_club_loan_status/loan_status_prediction.ipynb)**  
   - **Dataset:** [Lending Club Loan Defaulters (Kaggle)](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction)  
   - **Key Techniques & Workflow:**  
     - Data cleaning: encoded categorical variables (one-hot for loan purpose, grade), handled missing numeric values.  
     - Addressed class imbalance using class weights in model fitting.  
     - Trained and compared **Random Forest** vs. **Decision Tree** classifiers.  
     - Reported accuracy, precision, recall, and displayed ROC curves.  

3. **[E-commerce Customers Analysis](projects/ecommerce_customers/ecommerce_customers_analysis.ipynb)**  
   - **Dataset:** [E-commerce Customers (Kaggle)](https://www.kaggle.com/datasets/srolka/ecommerce-customers)  
   - **Key Techniques & Workflow:**  
     - EDA: scatterplots of yearly spending vs. account tenure, bar charts of gender distribution.  
     - Performed **Linear Regression** to predict yearly spending based on tenure, history, and average session length.  
     - Analyzed residuals, reported R² score, and visualized model fit.  

4. **[Advertisement Click Prediction](projects/advertisement_click_on_ads/advertisement_click.ipynb)**  
   - **Dataset:** [Advertisement Click on Ad (Kaggle)](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad)  
   - **Key Techniques & Workflow:**  
     - Preprocessing: one-hot encoding for ad topic and device type, scaled numerical features.  
     - Trained a **Logistic Regression** model for binary click/no-click classification.  
     - Evaluated using confusion matrix, precision/recall, and ROC AUC.

---

### 🗣 Natural Language Processing

1. **[Disaster Tweet Classification](projects/disaster_tweets/disaster_tweets_classification.ipynb)**  
   - **Dataset:** [NLP Getting Started Competition (Kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)  
   - **Key Techniques & Workflow:**  
     - Text preprocessing: removed punctuation, lowercased text, removed stopwords.  
     - Converted tweets into TF-IDF vectors using scikit-learn’s `TfidfVectorizer`.  
     - Trained and compared **Multinomial Naïve Bayes**, **Logistic Regression**, and **SVM** classifiers.  
     - Reported precision, recall, F1-score and visualized confusion matrix heatmaps.  

2. **[IMDB Movie Review Sentiment Analysis](projects/imdb_movie_review/imdb_review_classification.ipynb)**  
   - **Dataset:** [Bag of Words Meets Bags of Popcorn (Kaggle)](https://www.kaggle.com/competitions/word2vec-nlp-tutorial)  
   - **Key Techniques & Workflow:**  
     - Vectorized reviews using both **CountVectorizer** and **TF-IDF**.  
     - Built classification pipelines for **Logistic Regression**, **Naive Bayes**, and **Linear SVC**.  
     - Employed `GridSearchCV` for hyperparameter tuning.  
     - Generated submission-ready CSV for Kaggle leaderboard.

---

### 🔍 Exploratory Data Analysis (EDA)

1. **[Emergency 911 Calls Analysis](projects/emergency_911_calls/911_calls.ipynb)**  
   - **Dataset:** [MontcoAlert 911 Call Logs (Kaggle)](https://www.kaggle.com/datasets/mchirico/montcoalert)  
   - **Key Techniques & Workflow:**  
     - Parsed timestamps to extract day, hour, and month.  
     - Grouped calls by emergency type and plotted count trends over time.  
     - Created Seaborn visualizations: heatmaps of call density, line plots of monthly trends.  

2. **[Netflix Movies & TV Shows Analysis](projects/netflix_movies_and_tv_shows/netflix_shows_analysis.ipynb)**  
   - **Dataset:** [Netflix Shows (Kaggle)](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
   - **Key Techniques & Workflow:**  
     - Cleaned data: parsed release year, split cast and director columns.  
     - Identified top genres over time and analyzed content type ratios (Movie vs. TV Show).  
     - Visualized metrics: bar charts of most prolific directors, distribution of ratings.

---

### 📂 Other Folders

1. **[Supervised Learning Examples](machine_learning/supervised_learning/)**  
   - Contains example notebooks demonstrating core supervised learning algorithms (e.g., Linear Regression, Decision Trees, Support Vector Machines). These notebooks provide educational, from-scratch implementations and comparisons on synthetic or small public datasets.

2. **[SQL Layoff Data Analysis](sql/mysql/layoff_data/)**  
   - Includes SQL scripts (`queries.sql`) and sample CSVs showing how to import layoff-related datasets into MySQL, write analytical queries, and generate summary statistics on layoffs by industry, date, and location.

---

## 🤝 Contributing

1. Fork the repository  
2. Create a new branch: `git checkout -b feature/new-feature`  
3. Commit your changes  
4. Open a Pull Request  

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 📬 Contact

Maintained by **MNMMOoN**  
GitHub: [github.com/MNMMOoN](https://github.com/MNMMOoN)