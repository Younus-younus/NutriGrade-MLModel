# NutriGrade - Nutrition Grade Prediction Model

An NLP-based machine learning model that predicts nutrition grades (A-D) from product names and ingredient lists using LinearSVC and TF-IDF vectorization.

## Features

- **Text-based prediction**: Uses product names and ingredients to predict nutrition grades
- **Efficient pipeline**: Implements proper ML pipeline with preprocessing and feature engineering
- **Dual-mode operation**: Automatically handles training and inference phases
- **Model persistence**: Saves and loads trained models using joblib

## Project Structure

```
NutriGrade/
├── main.py                      # Main script (training + inference)
├── OpenFoodFactsDataset.tsv     # Training dataset
├── test_input.csv               # Test input for predictions (created)
├── nutrigrade_model.joblib      # Saved model (created after training)
└── predictions.csv              # Prediction output (created after inference)
```

## Usage

### First Run - Training Phase

When you run the script for the first time (no model file exists):

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Handle missing values and outliers
3. Create TF-IDF features from text
4. Train a LinearSVC model
5. Evaluate performance on test set
6. Save the model to `nutrigrade_model.joblib`

### Subsequent Runs - Inference Phase

After the model is trained and saved, running the script again:

```bash
python main.py
```

This will:
1. Load the saved model
2. Look for `test_input.csv` containing products to predict
3. Make predictions and save results to `predictions.csv`
4. If no test file exists, run an example prediction

### Test Input Format

Create a CSV file named `test_input.csv` with the following columns:

```csv
product_name,ingredients_text
Organic Whole Wheat Bread,whole wheat flour water yeast salt olive oil
Dark Chocolate Bar,cocoa mass cocoa butter sugar vanilla extract
Fruit Yogurt,yogurt fruit sugar pectin
```

## Model Details

- **Algorithm**: LinearSVC (Linear Support Vector Classification)
- **Text Features**: TF-IDF with bigrams (5000 max features)
- **Preprocessing**: 
  - Median imputation for numerical features
  - Outlier removal (0-100 range for nutritional values)
  - Text combination (product name + ingredients)
- **Output**: Nutrition grades (a, b, c, d)

## Files

- **MODEL_FILE**: `nutrigrade_model.joblib` - Contains model, vectorizer, encoder, and imputer
- **DATA_FILE**: `OpenFoodFactsDataset.tsv` - Training dataset
- **TEST_INPUT_FILE**: `test_input.csv` - Input for batch predictions
- **OUTPUT_FILE**: `predictions.csv` - Prediction results

## Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## Retraining the Model

To retrain the model from scratch:

1. Delete the existing model file:
   ```bash
   del nutrigrade_model.joblib  # Windows
   rm nutrigrade_model.joblib   # Linux/Mac
   ```

2. Run the script again:
   ```bash
   python main.py
   ```

## Example Output

### Training Phase
```
============================================================
NUTRIGRADE MODEL TRAINING
============================================================

Loading data...
Initial data shape: (50000, 10)
After dropping missing targets: (45000, 10)
...
Training Accuracy: 0.9234
Test Accuracy: 0.8956

Model saved to: nutrigrade_model.joblib
```

### Inference Phase
```
============================================================
NUTRIGRADE MODEL INFERENCE
============================================================

Loading saved model...
Reading test data from test_input.csv...

Inference complete!
Results saved to: predictions.csv
Number of predictions: 10
```

## Notes

- The model filters out nutrition grade 'e' during training
- Uses stratified train-test split (80/20) to maintain class distribution
- Automatically handles missing values in numerical columns
- Combines product names and ingredients for richer text features
