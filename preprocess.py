import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ResumePreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Clean and preprocess text for analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

    def extract_skills(self, text):
        """
        Extract potential skills from resume text
        """
        skills_list = [
            'python', 'machine learning', 'data science', 'nlp', 
            'tensorflow', 'keras', 'sql', 'javascript', 'react', 
            'nodejs', 'flask', 'django', 'java', 'c++', 
            'data analysis', 'statistics', 'communication', 
            'project management'
        ]
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Find matched skills
        matched_skills = [
            skill for skill in skills_list 
            if skill in text_lower
        ]
        
        return matched_skills

def extract_resume_text(file_path):
    """
    Extract text from different file types
    """
    import PyPDF2
    import docx

    # Determine file type and extract text
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")