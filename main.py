import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# File paths
MODEL_FILE = "nutrigrade_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
ENCODER_FILE = "label_encoder.pkl"
IMPUTER_FILE = "imputer.pkl"
DATA_FILE = "OpenFoodFactsDataset.tsv"

# Selected columns for the model
SELECTED_COLUMNS = [
    'product_name',
    'ingredients_text',
    'main_category_en',
    'energy_100g',
    'fat_100g',
    'saturated-fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
    'fruits-vegetables-nuts_100g',
    'nutrition_grade_fr'
]

# Columns to drop during preprocessing
DROP_COLUMNS = ['fiber_100g', 'fruits-vegetables-nuts_100g', 'main_category_en']

# Nutritional columns (per 100g)
NUTRITION_COLUMNS = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 
    'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g'
]

# Nutrition grade descriptions
GRADE_DESCRIPTIONS = {
    'a': 'Excellent - Very healthy choice',
    'b': 'Good - Healthy option',
    'c': 'Fair - Moderately healthy',
    'd': 'Poor - Less healthy',
    'e': 'Bad - Unhealthy choice'
}


def load_and_prepare_data(file_path):
    """Load and prepare nutrition data"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    
    print(f"Original dataset: {len(df)} records, {len(df.columns)} columns")
    
    # Select relevant columns
    print(f"Selecting {len(SELECTED_COLUMNS)} relevant columns...")
    data = df[SELECTED_COLUMNS].copy()
    
    # Drop unwanted columns
    print(f"Dropping columns: {DROP_COLUMNS}")
    data = data.drop(columns=DROP_COLUMNS)
    
    # Remove rows where target variable is missing
    print("Removing rows with missing target variable (nutrition_grade_fr)...")
    initial_size = len(data)
    data = data.dropna(subset=['nutrition_grade_fr'])
    print(f"Removed {initial_size - len(data)} rows with missing target")
    
    print(f"Data prepared: {len(data)} records")
    return data


def preprocess_data(data, imputer=None, fit=True):
    """Preprocess nutrition data - handle missing values and outliers"""
    
    # Handle missing values in numerical columns
    print("\nHandling missing values in numerical columns...")
    data_num = data.select_dtypes(include=[np.number])
    
    if fit:
        imputer = SimpleImputer(strategy="median")
        imputed_values = imputer.fit_transform(data_num)
    else:
        imputed_values = imputer.transform(data_num)
    
    data_num_clean = pd.DataFrame(imputed_values, columns=data_num.columns, index=data.index)
    data[data_num.columns] = data_num_clean.values
    
    # Handle missing values in text columns
    print("Handling missing values in text columns...")
    data = data.dropna(subset=['product_name', 'ingredients_text'])
    
    if fit:
        # Remove outliers (values should be between 0-100 for per 100g measurements)
        print("Removing outliers (values outside 0-100g range)...")
        initial_size = len(data)
        data = data[
            (data[NUTRITION_COLUMNS] <= 100).all(axis=1) & 
            (data[NUTRITION_COLUMNS] >= 0).all(axis=1)
        ]
        print(f"Removed {initial_size - len(data)} outlier records")
    
    return data, imputer


def create_text_features(data, vectorizer=None, fit=True):
    """Combine text columns and create TF-IDF features"""
    print("\nCreating text features...")
    
    # Combine product name and ingredients
    data['combined_text'] = data['product_name'] + ' ' + data['ingredients_text']
    
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced for efficiency
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,  # Ignore terms appearing in >80% of docs
            min_df=3     # Ignore terms appearing in <3 docs
        )
        text_features = vectorizer.fit_transform(data['combined_text'])
    else:
        text_features = vectorizer.transform(data['combined_text'])
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        text_features.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=data.index
    )
    
    # Add TF-IDF features to data
    data = data.drop(['product_name', 'ingredients_text', 'combined_text'], axis=1)
    data = pd.concat([data, tfidf_df], axis=1)
    
    print(f"Created {len(tfidf_df.columns)} text features")
    return data, vectorizer


def predict_nutrition_grade(product_name, ingredients, nutrition_values, 
                           model, vectorizer, encoder, imputer):
    """Predict nutrition grade for a single product"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'product_name': [product_name],
        'ingredients_text': [ingredients],
        'energy_100g': [nutrition_values.get('energy', 0)],
        'fat_100g': [nutrition_values.get('fat', 0)],
        'saturated-fat_100g': [nutrition_values.get('saturated_fat', 0)],
        'carbohydrates_100g': [nutrition_values.get('carbohydrates', 0)],
        'sugars_100g': [nutrition_values.get('sugars', 0)],
        'proteins_100g': [nutrition_values.get('proteins', 0)],
        'salt_100g': [nutrition_values.get('salt', 0)]
    })
    
    # Preprocess numerical features
    numeric_cols = input_data.select_dtypes(include=[np.number]).columns
    input_data[numeric_cols] = imputer.transform(input_data[numeric_cols])
    
    # Create text features
    combined_text = input_data['product_name'] + ' ' + input_data['ingredients_text']
    text_features = vectorizer.transform(combined_text)
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        text_features.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    
    # Drop text columns and combine with TF-IDF features
    input_data = input_data.drop(['product_name', 'ingredients_text'], axis=1)
    input_data = pd.concat([input_data.reset_index(drop=True), tfidf_df], axis=1)
    
    # Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Get results
    predicted_grade = encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_grades = encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[top_3_idx]
    
    return predicted_grade, confidence, list(zip(top_3_grades, top_3_probs))


if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    print("=" * 80)
    print("TRAINING PHASE - NUTRITION GRADE PREDICTOR")
    print("=" * 80)
    
    # Load and prepare data
    data = load_and_prepare_data(DATA_FILE)
    
    # Preprocess data
    data, imputer = preprocess_data(data, fit=True)
    
    # Create text features
    data, vectorizer = create_text_features(data, fit=True)
    
    # Encode target variable
    print("\nEncoding target variable (nutrition_grade_fr)...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['nutrition_grade_fr'])
    X = data.drop('nutrition_grade_fr', axis=1)
    
    print(f"Number of nutrition grades: {len(label_encoder.classes_)}")
    print(f"Grades: {list(label_encoder.classes_)}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    print("This may take a few minutes...")
    model = LogisticRegression(
        max_iter=500,
        solver='saga',  # More efficient for large sparse datasets
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1  # Show progress
    )
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(
        y_test, y_test_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Save models and preprocessors
    print("\n" + "=" * 80)
    print("SAVING MODELS AND PREPROCESSORS")
    print("=" * 80)
    
    joblib.dump(model, MODEL_FILE)
    print(f"✓ Model saved to: {MODEL_FILE}")
    
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"✓ Vectorizer saved to: {VECTORIZER_FILE}")
    
    joblib.dump(label_encoder, ENCODER_FILE)
    print(f"✓ Label Encoder saved to: {ENCODER_FILE}")
    
    joblib.dump(imputer, IMPUTER_FILE)
    print(f"✓ Imputer saved to: {IMPUTER_FILE}")
    
    # Save test data for inference testing
    test_data = data.iloc[y_test.index].copy()
    test_data['nutrition_grade_fr'] = label_encoder.inverse_transform(y_test)
    test_data.to_csv("test_input.csv", index=False)
    print(f"✓ Test data saved to: test_input.csv")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

else:
    # INFERENCE PHASE
    print("=" * 80)
    print("INFERENCE PHASE - NUTRITION GRADE PREDICTOR")
    print("=" * 80)
    
    print("\nLoading saved model, vectorizer, encoder, and imputer...")
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(ENCODER_FILE)
    imputer = joblib.load(IMPUTER_FILE)
    
    print(f"✓ Model loaded from: {MODEL_FILE}")
    print(f"✓ Vectorizer loaded from: {VECTORIZER_FILE}")
    print(f"✓ Label Encoder loaded from: {ENCODER_FILE}")
    print(f"✓ Imputer loaded from: {IMPUTER_FILE}")
    
    # Check if test input file exists
    if os.path.exists("test_input.csv"):
        print(f"\nLoading test data from: test_input.csv")
        input_data = pd.read_csv("test_input.csv")
        
        # Save actual grades if available
        if 'nutrition_grade_fr' in input_data.columns:
            actual_grades = input_data['nutrition_grade_fr'].copy()
            input_data_for_prediction = input_data.drop('nutrition_grade_fr', axis=1)
        else:
            actual_grades = None
            input_data_for_prediction = input_data.copy()
        
        # Preprocess
        print("Preprocessing data...")
        input_data_processed, _ = preprocess_data(input_data_for_prediction, imputer=imputer, fit=False)
        
        # Create text features
        input_data_processed, _ = create_text_features(input_data_processed, vectorizer=vectorizer, fit=False)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(input_data_processed)
        probabilities = model.predict_proba(input_data_processed)
        
        predicted_grades = label_encoder.inverse_transform(predictions)
        confidence_scores = probabilities.max(axis=1)
        
        # Create results dataframe
        results = pd.DataFrame({
            'product_name': input_data['product_name'].values[:len(predicted_grades)],
            'predicted_grade': predicted_grades,
            'confidence': confidence_scores,
            'grade_description': [GRADE_DESCRIPTIONS.get(g.lower(), 'Unknown') for g in predicted_grades]
        })
        
        if actual_grades is not None:
            results['actual_grade'] = actual_grades.values[:len(predicted_grades)]
            accuracy = accuracy_score(actual_grades[:len(predicted_grades)], predicted_grades)
            print(f"\nAccuracy on test set: {accuracy:.4f}")
        
        # Get top 3 predictions for each product
        top_3_predictions = []
        for prob_row in probabilities:
            top_3_idx = np.argsort(prob_row)[-3:][::-1]
            top_3_grades = label_encoder.inverse_transform(top_3_idx)
            top_3_probs = prob_row[top_3_idx]
            top_3_str = ' | '.join([f"{grade.upper()} ({prob:.2%})" for grade, prob in zip(top_3_grades, top_3_probs)])
            top_3_predictions.append(top_3_str)
        
        results['top_3_predictions'] = top_3_predictions
        
        # Display sample results
        print("\n" + "=" * 80)
        print("Sample Predictions:")
        print("=" * 80)
        display_cols = ['product_name', 'predicted_grade', 'confidence', 'grade_description']
        if actual_grades is not None:
            display_cols.insert(1, 'actual_grade')
        print(results[display_cols].head(10).to_string(index=False))
        
        # Save results
        results.to_csv("predictions_output.csv", index=False)
        print(f"\n✓ Predictions saved to: predictions_output.csv")
        print(f"✓ Total predictions: {len(predictions)}")
        
    else:
        # Example prediction with sample product
        print("\nNo test_input.csv found. Running example prediction...")
        
        sample_product = {
            'name': "Organic Whole Milk",
            'ingredients': "organic milk, vitamin d3",
            'nutrition': {
                'energy': 64,
                'fat': 3.5,
                'saturated_fat': 2.3,
                'carbohydrates': 5.0,
                'sugars': 5.0,
                'proteins': 3.3,
                'salt': 0.1
            }
        }
        
        print("\nSample Product:")
        print("-" * 60)
        print(f"Name: {sample_product['name']}")
        print(f"Ingredients: {sample_product['ingredients']}")
        print("Nutrition (per 100g):")
        for key, value in sample_product['nutrition'].items():
            print(f"  {key}: {value}g")
        print("-" * 60)
        
        # Predict
        predicted_grade, confidence, top_3 = predict_nutrition_grade(
            sample_product['name'],
            sample_product['ingredients'],
            sample_product['nutrition'],
            model, vectorizer, label_encoder, imputer
        )
        
        print("\n" + "=" * 80)
        print("🎯 NUTRITION GRADE PREDICTION")
        print("=" * 80)
        
        print(f"\n🏆 Predicted Grade: {predicted_grade.upper()}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"📝 Description: {GRADE_DESCRIPTIONS.get(predicted_grade.lower(), 'Unknown')}")
        
        print("\n" + "=" * 80)
        print("Top 3 Possible Grades:")
        print("=" * 80)
        
        for i, (grade, prob) in enumerate(top_3, 1):
            print(f"\n{i}. Grade {grade.upper()}")
            print(f"   Probability: {prob:.2%}")
            print(f"   Description: {GRADE_DESCRIPTIONS.get(grade.lower(), 'Unknown')}")
        
        print("\n" + "=" * 80)
        print("💡 Lower grades (A/B) indicate healthier products")
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)
