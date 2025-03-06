"""
Train resume classification model with improved scoring.
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocess import ResumePreprocessor


class ResumeClassifier:
    """Class to train and predict job categories for resumes"""

    def __init__(self):
        """Initialize the resume classifier"""
        self.preprocessor = ResumePreprocessor()

        # Create an improved ML pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=10000,  # Increase word features
                ngram_range=(1, 2),  # Use bigrams
                min_df=2,            # Remove very rare words
                max_df=0.85          # Remove very common words
            )),
            ('classifier', MultinomialNB())  # supports probabilities
        ])

    def load_dataset(self, dataset_path='dataset.csv'):
        """
        Load and preprocess dataset from CSV file.
        """
        df = pd.read_csv(dataset_path)

        # Clean text
        df['cleaned_text'] = df['Resume'].apply(self.preprocessor.clean_text)

        return df

    def train_model(self, dataset_path='dataset.csv'):
        """
        Train resume classification model.
        """
        df = self.load_dataset(dataset_path)

        # Split data with stratification
        x_train, x_test, y_train, y_test = train_test_split(
            df['cleaned_text'],
            df['Category'],
            test_size=0.2,
            random_state=42,
            stratify=df['Category']
        )

        # Train pipeline
        self.pipeline.fit(x_train, y_train)

        # Evaluate model
        y_pred = self.pipeline.predict(x_test)
        print(classification_report(y_test, y_pred))

        # Save model
        os.makedirs('model', exist_ok=True)
        with open('model/resume_classifier_pipeline.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

        print("Model saved successfully!")

    def predict_category(self, resume_text):
        """
        Predict job category for a resume.
        """
        try:
            # Load saved model
            with open('model/resume_classifier_pipeline.pkl', 'rb') as f:
                loaded_pipeline = pickle.load(f)

            # Clean text before prediction
            cleaned_resume = self.preprocessor.clean_text(resume_text)

            # Predict category
            prediction = loaded_pipeline.predict([cleaned_resume])[0]

            # Get probability confidence (only works with Naive Bayes)
            confidence = 0.0
            if hasattr(loaded_pipeline.named_steps
                       ['classifier'], 'predict_proba'):
                prediction_proba = (loaded_pipeline.predict_proba(
                    [cleaned_resume])[0])
                confidence = max(prediction_proba)

            print(f"Predicted: {prediction}, Confidence: {confidence:.2f}")
            return prediction, confidence

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Model loading error: {e}")
            return "Unknown", 0.0
        except ValueError as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0


# Train the model when this script is run
if __name__ == '__main__':
    classifier = ResumeClassifier()
    classifier.train_model()
