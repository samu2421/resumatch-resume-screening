"""Preprocessing, skill matching and keyword overlap."""
import pytest

from preprocess import NLTK_RESOURCES, extract_resume_text


def test_clean_text(preprocessor):
    text = "Python Developer with 5+ years of experience in AI!"
    assert preprocessor.clean_text(text) == "python develop year experi ai"


def test_clean_text_keeps_domain_words(preprocessor):
    assert preprocessor.clean_text("data science and machine learning") == \
        "data scienc machin learn"


def test_nltk_resources_are_the_ones_used():
    assert set(NLTK_RESOURCES) == {"punkt_tab", "stopwords"}


def test_extract_skills_case_insensitive_and_multiword(preprocessor):
    text = "Experience with PYTHON, Machine   Learning,\nSQL and Big Data."
    skills = preprocessor.extract_skills(text)
    assert {"python", "machine learning", "sql", "big data"} <= set(skills)


@pytest.mark.parametrize("text, absent", [
    ("JavaScript developer", "java"),       # no substring match
    ("Gitlab and github user", "git"),      # whole word only
    ("Pythonic code", "python"),
    ("Experience in C and C#", "c++"),
    ("reactive programming", "react"),
])
def test_extract_skills_word_boundaries(preprocessor, text, absent):
    assert absent not in preprocessor.extract_skills(text)


def test_extract_skills_special_characters(preprocessor):
    assert preprocessor.extract_skills("Wrote C++ and Java code") == \
        ["java", "c++"]


def test_extract_skills_none(preprocessor):
    assert preprocessor.extract_skills("Gardening and cooking") == []


def test_keyword_overlap_full_and_partial(preprocessor):
    assert preprocessor.keyword_overlap(
        "python sql docker", "Python and SQL") == 100.0
    # job keywords after cleaning: python, sql, docker, kubernet -> 2 of 4
    assert preprocessor.keyword_overlap(
        "I use Python and SQL", "Python, SQL, Docker, Kubernetes") == 50.0


def test_keyword_overlap_counts_distinct_words(preprocessor):
    assert preprocessor.keyword_overlap(
        "python", "python python python java") == 50.0


def test_keyword_overlap_no_keywords(preprocessor):
    assert preprocessor.keyword_overlap("python", "the and of") == 0.0


def test_keyword_overlap_never_above_100(preprocessor):
    assert preprocessor.keyword_overlap(
        "python " * 50, "python") == 100.0


def test_extract_resume_text_docx(tmp_path, synthetic_docx):
    path = tmp_path / "resume.docx"
    path.write_bytes(synthetic_docx)
    assert "Data Scientist" in extract_resume_text(str(path))


def test_extract_resume_text_unsupported(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        extract_resume_text(str(path))
