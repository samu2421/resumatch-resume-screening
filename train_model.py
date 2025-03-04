import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from preprocess import ResumePreprocessor


class ResumeClassifier:
    """Class to train and predict job categories for resumes"""
    def __init__(self):
        self.preprocessor = ResumePreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()

    def load_dataset(self, dataset_path):
        """
        Load and preprocess dataset
        """
        # Sample dataset structure
        data = {
            'text': [
                'python machine learning data science project',
                'web development javascript react nodejs',
                'software engineering java c++ algorithms',
                'human resources communication management',
                'mechanical engineering design cad'
            ],
            'category': [
                'Data Science',
                'Web Development', 
                'Software Engineer', 
                'HR',
                'Mechanical Engineer'
            ]
        }
        df = pd.DataFrame(data)
        
        # Preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        return df

    def train_model(self, dataset_path=None):
        """
        Train resume classification model
        """
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], 
            df['category'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_vectorized)
        print(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        os.makedirs('model', exist_ok=True)
        with open('model/resume_classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open('model/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model and vectorizer saved successfully!")

    def predict_category(self, resume_text):
        """
        Predict job category for a resume
        """
        # Load saved model and vectorizer
        with open('model/resume_classifier.pkl', 'rb') as f:
            loaded_classifier = pickle.load(f)
        
        with open('model/vectorizer.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        
        # Preprocess and vectorize text
        cleaned_text = self.preprocessor.clean_text(resume_text)
        vectorized_text = loaded_vectorizer.transform([cleaned_text])
        
        # Predict category
        prediction = loaded_classifier.predict(vectorized_text)[0]
        prediction_proba = loaded_classifier.predict_proba(vectorized_text)[0]
        
        return prediction, max(prediction_proba)

# Train the model when this script is run
if __name__ == '__main__':
    classifier = ResumeClassifier()
    classifier.train_model()
