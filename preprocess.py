"""
Resume preprocessing: NLTK setup, text cleaning, skill matching,
keyword overlap and text extraction from PDF/DOCX files.
"""
import os
import re
from pathlib import Path

import docx
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

# NLTK resources the code actually uses:
#   punkt_tab -> word_tokenize (NLTK >= 3.9)
#   stopwords -> stopwords.words('english')
NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}


def ensure_nltk_resources() -> None:
    """
    Make the NLTK resources available.

    ./nltk_data next to this file is searched first, so resources
    pre-downloaded at build time (e.g. on Render) are used directly.
    Anything still missing is downloaded once, quietly, into that folder.
    """
    nltk_dir = str(NLTK_DATA_DIR)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    for name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            NLTK_DATA_DIR.mkdir(exist_ok=True)
            nltk.download(name, download_dir=nltk_dir, quiet=True)
            nltk.data.find(resource_path)  # fail loudly if still missing


ensure_nltk_resources()


SKILLS_LIST = [
    'python', 'machine learning', 'data science', 'deep learning',
    'tensorflow', 'keras', 'sql', 'javascript', 'react', 'nodejs',
    'flask', 'django', 'java', 'c++', 'statistics', 'communication',
    'project management', 'git', 'agile', 'docker', 'kubernetes',
    'aws', 'azure', 'cloud computing', 'big data', 'nlp'
]


def _skill_pattern(skill: str) -> re.Pattern:
    """
    Whole-word / whole-phrase, case-insensitive pattern for a skill.

    Words of a multi-word skill may be separated by any whitespace.
    Boundaries also exclude '+' and '#' so that 'c' does not match 'c++'
    and 'java' does not match 'javascript'.
    """
    words = [re.escape(word) for word in skill.split()]
    return re.compile(
        r'(?<![\w+#])' + r'\s+'.join(words) + r'(?![\w+#])',
        re.IGNORECASE,
    )


SKILL_PATTERNS = {skill: _skill_pattern(skill) for skill in SKILLS_LIST}


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

        self.skills_list = list(SKILLS_LIST)

    def clean_text(self, text):
        """
        Clean and preprocess text for analysis.
        """
        # Convert to lowercase
        text = text.lower()

        # Keep letters and whitespace only
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
        Return the skills from the skills list that occur in the text as
        whole words/phrases (case-insensitive), in skills-list order.
        """
        return [
            skill for skill, pattern in SKILL_PATTERNS.items()
            if pattern.search(text)
        ]

    def keyword_overlap(self, resume_text, job_description):
        """
        Share (in %) of distinct job-description keywords, after cleaning,
        that also occur in the cleaned resume. Returns 0.0 if the job
        description has no keywords left after cleaning.
        """
        job_words = set(self.clean_text(job_description).split())
        if not job_words:
            return 0.0
        resume_words = set(self.clean_text(resume_text).split())
        common = job_words & resume_words
        return round(100 * len(common) / len(job_words), 1)


def extract_resume_text(file_path):
    """
    Extract text from PDF and DOCX resumes.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = (page.extract_text() for page in reader.pages)
            return ''.join(text for text in pages if text)

    if file_extension == '.docx':
        doc = docx.Document(file_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

    raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
