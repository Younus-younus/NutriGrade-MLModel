# NutriGrade - Nutrition Grade Prediction Model

An NLP-based machine learning model that predicts nutrition grades (A-E) from product names and ingredient lists using LinearSVC and TF-IDF vectorization.

## Features

- **Text-based prediction**: Uses product names and ingredients to predict nutrition grades
- **Efficient pipeline**: Implements proper ML pipeline with preprocessing and feature engineering
- **Dual-mode operation**: Automatically handles training and inference phases
- **Model persistence**: Saves and loads trained models using joblib

## Download the .tsv dataset file
https://www.kaggle.com/datasets/openfoodfacts/world-food-facts

## Project Structure

```
NutriGrade/
├── main.py                      # Main script (training + inference)
├── OpenFoodFactsDataset.tsv     # Training dataset
├── test_input.csv               # Test input for predictions (created)
├── nutrigrade_model.joblib      # Saved model (created after training)
└── predictions.csv              # Prediction output (created after inference)
```

## 📋 Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## 🚀 How to Use

### First Run (Training)

Simply run the script with your dataset:

```bash
python main.py
```

**What happens:**
1. Loads data from `OpenFoodFactsDataset.tsv`
2. Preprocesses data (handles missing values, removes outliers)
3. Creates TF-IDF features from product name and ingredients
4. Trains Logistic Regression model
5. Evaluates performance (accuracy, classification report, confusion matrix)
6. Saves trained model and preprocessors to disk:
   - `nutrigrade_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`
   - `imputer.pkl`
   - `test_input.csv` (for testing)

### Subsequent Runs (Inference)

After training, running the script again will:
1. Load saved models
2. Look for `test_input.csv` for batch predictions
3. Or run example prediction with sample product
4. Save predictions to `predictions_output.csv`

## 📊 Input Data Format

Your `OpenFoodFactsDataset.tsv` should contain:

```
product_name    ingredients_text    energy_100g    fat_100g    saturated-fat_100g    carbohydrates_100g    sugars_100g    proteins_100g    salt_100g    nutrition_grade_fr
Milk           organic milk         64             3.5         2.3                   5.0                   5.0            3.3              0.1          b
```

### Required Columns:
- `product_name`: Name of the product
- `ingredients_text`: List of ingredients
- `energy_100g`: Energy per 100g
- `fat_100g`: Fat content per 100g
- `saturated-fat_100g`: Saturated fat per 100g
- `carbohydrates_100g`: Carbohydrates per 100g
- `sugars_100g`: Sugar content per 100g
- `proteins_100g`: Protein content per 100g
- `salt_100g`: Salt content per 100g
- `nutrition_grade_fr`: Target variable (grades: a, b, c, d, e)

## 🔄 Batch Predictions

Create a `test_input.csv` file:

```csv
product_name,ingredients_text,energy_100g,fat_100g,saturated-fat_100g,carbohydrates_100g,sugars_100g,proteins_100g,salt_100g
"Chocolate Bar","sugar cocoa butter milk",545,31,19,54,48,7,0.15
"Green Smoothie","spinach banana apple water",28,0.2,0.1,6,4,1,0.01
```

Then run:
```bash
python main.py
```

Results will be saved to `predictions_output.csv`

## 📈 Output Format

The predictions include:
- **Predicted Grade**: A, B, C, D, or E
- **Confidence**: Probability score (0-1)
- **Grade Description**: Human-readable description
- **Top 3 Predictions**: Alternative grades with probabilities

## 🎓 Nutrition Grades

- **A**: Excellent - Very healthy choice
- **B**: Good - Healthy option
- **C**: Fair - Moderately healthy
- **D**: Poor - Less healthy
- **E**: Bad - Unhealthy choice

## 🛠️ Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: 
  - TF-IDF features (5000 max) from product name + ingredients
  - 7 numerical nutritional values
- **Preprocessing**:
  - Median imputation for missing values
  - Outlier removal (values outside 0-100g range)
  - Text preprocessing with stopword removal
- **Validation**: 80-20 train-test split with stratification

## 📝 Example Output

```
🎯 NUTRITION GRADE PREDICTION
================================================================================

🏆 Predicted Grade: B
📊 Confidence: 87.45%
📝 Description: Good - Healthy option

Top 3 Possible Grades:
================================================================================

1. Grade B
   Probability: 87.45%
   Description: Good - Healthy option

2. Grade C
   Probability: 8.23%
   Description: Fair - Moderately healthy

3. Grade A
   Probability: 3.12%
   Description: Excellent - Very healthy choice
```

## 🔧 Customization

Modify these constants in `main.py`:

```python
# Adjust TF-IDF features
TfidfVectorizer(
    max_features=5000,  # Increase for more text features
    ngram_range=(1, 2)  # Adjust n-gram range
)

# Adjust model parameters
LogisticRegression(
    max_iter=1000,      # Increase if convergence fails
    solver='lbfgs',     # Try 'saga' for large datasets
    class_weight='balanced'  # Handles imbalanced classes
)
```

## 📊 Performance Tips

1. **More Data**: More training samples improve accuracy
2. **Feature Engineering**: Add more nutritional metrics if available
3. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
4. **Different Models**: Try RandomForest, XGBoost, or neural networks

## 🐛 Troubleshooting

**Model not training:**
- Ensure `OpenFoodFactsDataset.tsv` exists in the same directory
- Check data format matches requirements

**Low accuracy:**
- Check data quality (missing values, outliers)
- Try increasing `max_features` in TF-IDF
- Consider feature engineering

**Memory issues:**
- Reduce `max_features` in TF-IDF
- Sample your dataset for training

## 📁 File Structure

```
NutriGrade/
├── main.py                      # Main script
├── OpenFoodFactsDataset.tsv     # Training data
├── nutrigrade_model.pkl         # Trained model (generated)
├── tfidf_vectorizer.pkl        # Text vectorizer (generated)
├── label_encoder.pkl           # Target encoder (generated)
├── imputer.pkl                 # Data imputer (generated)
├── test_input.csv              # Test data (generated)
└── predictions_output.csv      # Results (generated)
```

## 🚀 Quick Start

```bash
# Clone or navigate to project directory
cd NutriGrade

# Ensure data file exists
ls OpenFoodFactsDataset.tsv

# Run training (first time)
python main.py

# Run inference (subsequent times)
python main.py
```

## 📧 Support

For issues or questions, check:
- Data format matches requirements
- All dependencies are installed
- Python version 3.7+
