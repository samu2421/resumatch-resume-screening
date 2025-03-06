"""Unit tests for the ResumePreprocessor and ResumeClassifier classes"""
import unittest
from preprocess import ResumePreprocessor
from train_model import ResumeClassifier


class TestResumePreprocessor(unittest.TestCase):
    """Test the ResumePreprocessor class"""
    def setUp(self):
        """Initialize the ResumePreprocessor before tests"""
        self.preprocessor = ResumePreprocessor()

    def test_clean_text(self):
        """Test if clean_text correctly preprocesses input text"""
        text = "Python Developer with 5+ years of experience in AI!"
        expected = "python develop year experi ai"  # Expected cleaned output
        self.assertEqual(self.preprocessor.clean_text(text), expected)

    def test_extract_skills(self):
        """Test if extract_skills correctly identifies known skills"""
        text = "Experience with Python, Machine Learning, and SQL."
        extracted_skills = self.preprocessor.extract_skills(text)
        self.assertIn("python", extracted_skills)
        self.assertIn("machine learning", extracted_skills)
        self.assertIn("sql", extracted_skills)


class TestResumeClassifier(unittest.TestCase):
    """Test the ResumeClassifier class"""
    def setUp(self):
        """Initialize the classifier before tests"""
        self.classifier = ResumeClassifier()

    def test_predict_category(self):
        """Test if the model can predict a category for a sample resume"""
        resume_text = ("Experienced software engineer skilled in:"
                       "Python ML, and AI.")
        prediction, confidence = self.classifier.predict_category(resume_text)
        self.assertIsInstance(prediction, str)  # Should return a category name
        self.assertTrue(0 <= confidence <= 1)  # Confidence between 0 and 1


if __name__ == '__main__':
    unittest.main()
