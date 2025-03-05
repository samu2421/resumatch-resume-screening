"""
This module contains helper functions for preprocessing resume text
"""
import os
import re
import nltk
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class ResumePreprocessor:
    """
    A class to preprocess resume text with cleaning and skill
    extraction capabilities.
    """

    def __init__(self):
        """
        Initialize the preprocessor with a lemmatizer and English stopwords.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.skills_list = [
            'python', 'machine learning', 'data science', 'nlp',
            'tensorflow', 'keras', 'sql', 'javascript', 'react',
            'nodejs', 'flask', 'django', 'java', 'c++',
            'data analysis', 'statistics', 'communication',
            'project management', 'git', 'agile', 'docker',
            'kubernetes', 'aws', 'azure', 'cloud computing'
        ]

    def clean_text(self, text):
        """
        Clean and preprocess text for analysis.
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
        Extract potential skills from resume text.
        """
        # Convert text to lowercase for matching
        text_lower = text.lower()

        # Find matched skills
        matched_skills = [
            skill for skill in self.skills_list
            if skill in text_lower
        ]

        return list(set(matched_skills))  # Remove duplicates


def extract_resume_text(file_path):
    """
    Extract text from different file types (PDF and DOCX).

    Args:
        file_path (str): Path to the resume file

    Returns:
        str: Extracted text from the resume

    Raises:
        ValueError: If an unsupported file type is provided
    """
    # Determine file type and extract text
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() for page in reader.pages)
        return text

    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        return text

    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
