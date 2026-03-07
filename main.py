import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

MODEL_FILE = "nutrigrade_model.joblib"
DATA_FILE = "OpenFoodFactsDataset.tsv"
TEST_INPUT_FILE = "test_input.csv"
OUTPUT_FILE = "predictions.csv"


class NutriGradeModel:
    """
    NLP-based ML model for predicting nutrition grades from product data
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = None
        self.tfidf_vectorizer = None
        self.numerical_imputer = None
        
    def load_and_preprocess_data(self, data_path=DATA_FILE):
        """Load data and perform preprocessing"""
        print("Loading data...")
        df = pd.read_csv(data_path, sep='\t')
        
        # Select relevant columns
        selected_columns = [
            'product_name',
            'ingredients_text',
            'energy_100g',
            'fat_100g',
            'saturated-fat_100g',
            'carbohydrates_100g',
            'sugars_100g',
            'proteins_100g',
            'salt_100g',
            'nutrition_grade_fr'
        ]
        
        data = df[selected_columns].copy()
        print(f"Initial data shape: {data.shape}")
        
        # Remove rows with missing target variable
        data = data.dropna(subset=['nutrition_grade_fr'])
        print(f"After dropping missing targets: {data.shape}")
        
        # Remove rows with missing text data
        data = data.dropna(subset=['product_name', 'ingredients_text'])
        print(f"After dropping missing text: {data.shape}")
        
        return data
    
    def handle_missing_values(self, data):
        """Impute missing numerical values with median"""
        print("Handling missing values...")
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.numerical_imputer = SimpleImputer(strategy='median')
        data[numerical_cols] = self.numerical_imputer.fit_transform(data[numerical_cols])
        
        return data
    
    def remove_outliers(self, data):
        """Remove outliers from nutritional columns"""
        print("Removing outliers...")
        nutrition_cols = [
            'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g'
        ]
        
        # Keep only valid values (0-100 for per 100g measurements)
        mask = (data[nutrition_cols] <= 100).all(axis=1) & (data[nutrition_cols] >= 0).all(axis=1)
        data = data[mask]
        print(f"After outlier removal: {data.shape}")
        
        return data
    
    def prepare_features(self, data):
        """Prepare text and numerical features"""
        print("Preparing features...")
        
        # Combine text features
        data['combined_text'] = data['product_name'].fillna('') + ' ' + data['ingredients_text'].fillna('')
        
        # Encode target variable
        data['nutrition_grade_encoded'] = self.label_encoder.fit_transform(data['nutrition_grade_fr'])
        
        # Filter out grade 'e' (encoded as 4 if present)
        data = data[data['nutrition_grade_encoded'] != 4]
        print(f"After filtering grade 'e': {data.shape}")
        
        return data
    
    def create_tfidf_features(self, data):
        """Create TF-IDF features from text"""
        print("Creating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        text_features = self.tfidf_vectorizer.fit_transform(data['combined_text'])
        tfidf_df = pd.DataFrame(
            text_features.toarray(),
            columns=self.tfidf_vectorizer.get_feature_names_out(),
            index=data.index
        )
        
        print(f"TF-IDF features shape: {tfidf_df.shape}")
        return tfidf_df
    
    def train(self):
        """Complete training pipeline"""
        # Load and preprocess data
        data = self.load_and_preprocess_data()
        data = self.handle_missing_values(data)
        data = self.remove_outliers(data)
        data = self.prepare_features(data)
        
        # Create features
        X = self.create_tfidf_features(data)
        y = data['nutrition_grade_encoded']
        
        # Train-test split
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train model
        print("\nTraining LinearSVC model...")
        self.model = LinearSVC(max_iter=2000, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        print(f"\nTraining Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=self.label_encoder.classes_[self.label_encoder.classes_ != 'e']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))
        
        return X_test, y_test, y_pred_test
    
    def save_model(self, model_path=MODEL_FILE):
        """Save the complete model pipeline"""
        print(f"\nSaving model to {model_path}...")
        
        model_artifacts = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'numerical_imputer': self.numerical_imputer
        }
        
        joblib.dump(model_artifacts, model_path)
        print(f"Model saved successfully!")
    
    def load_model(self, model_path=MODEL_FILE):
        """Load the complete model pipeline"""
        print(f"Loading model from {model_path}...")
        
        model_artifacts = joblib.load(model_path)
        self.model = model_artifacts['model']
        self.tfidf_vectorizer = model_artifacts['tfidf_vectorizer']
        self.label_encoder = model_artifacts['label_encoder']
        self.numerical_imputer = model_artifacts['numerical_imputer']
        
        print("Model loaded successfully!")
    
    def predict(self, product_name, ingredients_text):
        """Predict nutrition grade for a new product"""
        # Combine text
        combined_text = f"{product_name} {ingredients_text}"
        
        # Vectorize text
        text_features = self.tfidf_vectorizer.transform([combined_text])
        
        # Predict
        prediction = self.model.predict(text_features)
        grade = self.label_encoder.inverse_transform(prediction)[0]
        
        return grade


def main():
    """Main execution function"""
    
    if not os.path.exists(MODEL_FILE):
        # TRAINING PHASE
        print("="*60)
        print("NUTRIGRADE MODEL TRAINING")
        print("="*60 + "\n")
        print("Starting training phase...\n")
        
        # Initialize and train model
        nutrigrade = NutriGradeModel()
        nutrigrade.train()
        
        # Save model
        nutrigrade.save_model(MODEL_FILE)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Model saved to: {MODEL_FILE}")
        
    else:
        # INFERENCE PHASE
        print("="*60)
        print("NUTRIGRADE MODEL INFERENCE")
        print("="*60 + "\n")
        print("Loading saved model...\n")
        
        # Load model
        nutrigrade = NutriGradeModel()
        nutrigrade.load_model(MODEL_FILE)
        
        # Check if test input file exists
        if os.path.exists(TEST_INPUT_FILE):
            print(f"Reading test data from {TEST_INPUT_FILE}...")
            test_data = pd.read_csv(TEST_INPUT_FILE)
            
            # Ensure required columns exist
            if 'product_name' not in test_data.columns or 'ingredients_text' not in test_data.columns:
                print("Error: test_input.csv must contain 'product_name' and 'ingredients_text' columns")
                return
            
            # Make predictions
            predictions = []
            for idx, row in test_data.iterrows():
                product_name = str(row['product_name']) if pd.notna(row['product_name']) else ''
                ingredients = str(row['ingredients_text']) if pd.notna(row['ingredients_text']) else ''
                
                try:
                    grade = nutrigrade.predict(product_name, ingredients)
                    predictions.append(grade)
                except Exception as e:
                    print(f"Error predicting row {idx}: {e}")
                    predictions.append('unknown')
            
            # Add predictions to dataframe
            test_data['predicted_nutrition_grade'] = predictions
            
            # Save results
            test_data.to_csv(OUTPUT_FILE, index=False)
            
            print(f"\nInference complete!")
            print(f"Results saved to: {OUTPUT_FILE}")
            print(f"Number of predictions: {len(predictions)}")
            
            # Show sample predictions
            print("\nSample predictions:")
            print(test_data[['product_name', 'predicted_nutrition_grade']].head(10))
            
        else:
            # Example prediction if no test file
            print(f"No test input file found at: {TEST_INPUT_FILE}")
            print("\n" + "="*60)
            print("EXAMPLE PREDICTION")
            print("="*60)
            
            example_product = "Organic Whole Wheat Bread"
            example_ingredients = "whole wheat flour, water, yeast, salt, olive oil"
            
            predicted_grade = nutrigrade.predict(example_product, example_ingredients)
            print(f"\nProduct: {example_product}")
            print(f"Ingredients: {example_ingredients}")
            print(f"Predicted Nutrition Grade: {predicted_grade.upper()}")
            
            print(f"\nTo make batch predictions, create a file '{TEST_INPUT_FILE}' with columns:")
            print("  - product_name")
            print("  - ingredients_text")


if __name__ == "__main__":
    main()
