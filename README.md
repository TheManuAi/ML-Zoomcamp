# ML-Zoomcamp

My journey through the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course - a comprehensive machine learning engineering bootcamp covering everything from fundamentals to production deployment.

## 📚 Course Overview

This repository contains my notes, code implementations, and homework solutions for the ML Zoomcamp course. The course provides hands-on experience with real-world datasets and teaches both theoretical concepts and practical implementations of machine learning algorithms.

## 📂 Repository Structure

```
ML-Zoomcamp/
├── Week1 - Intro/              # Introduction to NumPy, Pandas, and Linear Algebra
├── Week2 - Regression/         # Linear Regression and Machine Learning for Regression
├── Week3 - Classification/     # Logistic Regression and Classification Models
├── Week4 - Evaluation/         # Model Evaluation Metrics and Validation
├── Week5+/                     # Additional weeks covering advanced topics
└── Homework/                   # Weekly homework assignments and solutions
```

## 🎯 Topics Covered

### Week 1: Introduction to Machine Learning
- **NumPy fundamentals**: Array operations, broadcasting, and vectorization for efficient numerical computing
- **Pandas for data manipulation**: DataFrames, filtering, grouping, and data cleaning
- **Linear algebra basics**: Vectors, matrices, dot products, and matrix operations essential for ML algorithms

### Week 2: Regression
- **Linear Regression from scratch**: Understanding the mathematical foundation using the normal equation: `w = (XᵀX)⁻¹Xᵀy`
- **Regularized Linear Regression (Ridge)**: Adding L2 penalty `(XᵀX + rI)⁻¹Xᵀy` to prevent overfitting by penalizing large coefficients
- **Root Mean Square Error (RMSE)**: Standard metric for regression that measures prediction accuracy: `√(Σ(yᵢ - ŷᵢ)²/n)`
- **Train/Validation/Test Split**: Dividing data (typically 60/20/20) to train models, tune hyperparameters, and evaluate final performance
- **Feature Engineering**: Creating and selecting features to improve model performance

### Week 3: Classification
- **Logistic Regression**: Binary classification using sigmoid function `σ(z) = 1/(1 + e⁻ᶻ)` to predict probabilities
- **One-Hot Encoding**: Converting categorical variables into binary vectors (e.g., [red, blue, green] → [1,0,0], [0,1,0], [0,0,1])
- **Mutual Information**: Measures dependency between features and target - higher values indicate more useful features for prediction. Unlike correlation, it captures non-linear relationships
- **Correlation Coefficient**: Measures linear relationship between numerical variables (ranges from -1 to +1). Values close to ±1 indicate strong linear relationships
- **Feature Selection**: Identifying and removing features that don't contribute to model performance, reducing overfitting and computational cost

### Week 4: Evaluation Metrics
- **Confusion Matrix**: 2x2 table showing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) to understand model predictions
- **Precision**: `TP/(TP + FP)` - Of all positive predictions, how many were actually positive? Important when false positives are costly
- **Recall**: `TP/(TP + FN)` - Of all actual positives, how many did we catch? Important when false negatives are costly
- **F1 Score**: Harmonic mean of precision and recall `2PR/(P+R)` - balances both metrics, useful when dealing with imbalanced classes
- **ROC Curve & AUC**: ROC plots True Positive Rate vs False Positive Rate at different thresholds. AUC (Area Under Curve) measures overall model performance - 0.5 is random, 1.0 is perfect
- **K-Fold Cross-Validation**: Splits data into K equal parts (folds), trains K times using different folds as validation. Provides more reliable performance estimates than single train/test split and reduces variance
- **Hyperparameter Tuning**: Systematically searching for optimal model parameters (e.g., regularization strength C, learning rate) to maximize performance

### Week 5 and Beyond
*Additional topics will be added as the course progresses, including:*
- Decision Trees and Ensemble Methods
- Neural Networks and Deep Learning
- Model Deployment and Production ML
- And more...

## 🛠️ Technologies & Tools

- **Python 3.x** - Primary programming language
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization and plotting
- **Scikit-learn** - ML algorithms, preprocessing, and metrics
- **Jupyter Notebooks** - Interactive development and documentation

## 📊 Key Skills Developed

- ✅ Implementing ML algorithms from scratch (understanding the math behind the code)
- ✅ Feature engineering and selection strategies
- ✅ Model evaluation using multiple metrics appropriate for different problems
- ✅ Cross-validation techniques for robust model assessment
- ✅ Hyperparameter optimization for model tuning
- ✅ Working with real-world datasets including missing values and categorical features
- ✅ Understanding trade-offs between different models and metrics

## 💡 Practical Applications

Throughout this course, I've applied machine learning to real-world problems including:
- Predicting house prices using regression
- Customer churn prediction using classification
- Lead scoring for marketing campaigns
- Model performance evaluation and optimization

## 📖 Resources

- [ML Zoomcamp Official Repository](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Course lectures and materials

## 👤 Author

**Manu** | [@TheManuAi](https://github.com/TheManuAi)