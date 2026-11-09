"""Prediction with a model trained on the deduplicated dataset."""
import pytest

from conftest import SYNTHETIC_DS_RESUME
from train_model import ResumeClassifier


def test_predict_data_science(trained_classifier):
    category, confidence = trained_classifier.predict_category(
        SYNTHETIC_DS_RESUME)
    assert category == "Data Science"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("text", [
    "Experienced Java developer with Spring Boot, Hibernate and REST APIs.",
    "QA engineer with manual and automated testing using Selenium and JUnit.",
    "",
])
def test_predict_returns_known_category(trained_classifier, text):
    category, confidence = trained_classifier.predict_category(text)
    assert category in trained_classifier.pipeline.classes_
    assert 0.0 <= confidence <= 1.0


def test_predict_cleans_raw_text_like_training(trained_classifier):
    """Raw text and its cleaned form must give the same prediction."""
    raw = "PYTHON, Machine-Learning & SQL!!"
    cleaned = trained_classifier.preprocessor.clean_text(raw)
    assert trained_classifier.pipeline.predict([cleaned])[0] == \
        trained_classifier.predict_category(raw)[0]


def test_predict_top_three(trained_classifier):
    top = trained_classifier.predict_top(SYNTHETIC_DS_RESUME, k=3)
    assert len(top) == 3
    categories = [category for category, _ in top]
    probabilities = [probability for _, probability in top]
    assert categories[0] == "Data Science"
    assert len(set(categories)) == 3
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    # predicted category and confidence are the first entry
    assert trained_classifier.predict_category(SYNTHETIC_DS_RESUME) == top[0]


def test_predict_without_model_raises():
    with pytest.raises(RuntimeError, match="train_model.py"):
        ResumeClassifier().predict_category("python")


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        ResumeClassifier.load(tmp_path / "missing.pkl")
