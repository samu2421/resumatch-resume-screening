"""
Improved Resume Preprocessing Module
"""
import os
import re
import nltk
import PyPDF2
import docx
import spacy
from fuzzywuzzy import process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Load spaCy NLP model for better skill extraction
nlp = spacy.load("en_core_web_sm")


class ResumePreprocessor:
    """
    A class to preprocess resume text.
    """

    def __init__(self):
        """Initialize the preprocessor with a stemmer and custom stopwords."""
        self.stemmer = PorterStemmer()
        base_stopwords = set(stopwords.words('english'))

        # Keep domain-related words while removing generic ones
        keep_words = {"data", "analysis", "science", "machine", "learning"}
        self.stop_words = base_stopwords - keep_words

        self.skills_list = [
            'python', 'machine learning', 'data science', 'deep learning',
            'tensorflow', 'keras', 'sql', 'javascript', 'react', 'nodejs',
            'flask', 'django', 'java', 'c++', 'statistics', 'communication',
            'project management', 'git', 'agile', 'docker', 'kubernetes',
            'aws', 'azure', 'cloud computing', 'big data', 'nlp'
        ]

    def clean_text(self, text):
        """
        Clean and preprocess text for analysis.
        """
        # Convert to lowercase
        text = text.lower()

        # Retain alphanumeric words, hyphens, and underscores
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and apply stemming
        cleaned_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return ' '.join(cleaned_tokens)

    def extract_skills(self, text):
        """
        Extract potential skills from resume text using fuzzy matching.
        """
        text_lower = text.lower()

        # Find matched skills using fuzzy matching
        matched_skills = [
            process.extractOne(skill, self.skills_list, score_cutoff=70)
            for skill in text_lower.split()  # Split text into words
        ]

        return list(set([skill[0] for skill in matched_skills if skill]))

    def extract_skills_spacy(self, text):
        """
        Extract skills using spaCy NER.
        """
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "SKILL"]]


def extract_resume_text(file_path):
    """
    Extract text from PDF and DOCX resumes.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
        return text

    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        return text

    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
