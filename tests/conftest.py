"""Shared pytest fixtures (all test data is synthetic)."""
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess import ResumePreprocessor  # noqa: E402
import train_model  # noqa: E402

DATASET_PATH = Path(
    os.getenv('DATASET_PATH', train_model.DEFAULT_DATASET_PATH))

SYNTHETIC_DS_RESUME = (
    "Jordan Example - Data Scientist. Skills: Python, SQL, machine learning, "
    "deep learning, TensorFlow, Keras, statistics, pandas, numpy, "
    "scikit-learn, data visualization, NLP. Built predictive models with "
    "regression, random forest and neural networks; performed exploratory "
    "data analysis and feature engineering on large datasets."
)


@pytest.fixture(scope='session')
def preprocessor():
    return ResumePreprocessor()


@pytest.fixture(scope='session')
def trained_classifier(preprocessor):
    """
    A trained classifier for the tests.

    With the dataset present, the pipeline is trained on the deduplicated
    data. Without it, the shipped model/resume_classifier_pipeline.pkl is
    loaded instead. Tests are skipped only if neither is available.
    """
    if not DATASET_PATH.exists():
        if train_model.MODEL_PATH.exists():
            return train_model.ResumeClassifier.load(train_model.MODEL_PATH)
        pytest.skip(f"Neither the dataset ({DATASET_PATH}) nor the shipped "
                    f"model ({train_model.MODEL_PATH}) was found; download "
                    "the dataset to data/resume_dataset.csv and run "
                    "python train_model.py.")
    df = train_model.load_dataset(DATASET_PATH, preprocessor)
    df = df.drop_duplicates(subset='Resume')
    # Train the same classifier type as the saved (selected) model.
    classifier = None
    if train_model.MODEL_PATH.exists():
        from sklearn.base import clone
        saved = train_model.ResumeClassifier.load(train_model.MODEL_PATH)
        classifier = clone(saved.pipeline.named_steps['classifier'])
    pipeline = train_model.build_pipeline(classifier).fit(df['cleaned_text'],
                                                          df['Category'])
    return train_model.ResumeClassifier(pipeline)


def make_docx(paragraphs) -> bytes:
    """Build a small synthetic .docx in memory."""
    import docx
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def synthetic_docx() -> bytes:
    return make_docx([
        "Jordan Example",
        "Data Scientist",
        SYNTHETIC_DS_RESUME,
    ])
