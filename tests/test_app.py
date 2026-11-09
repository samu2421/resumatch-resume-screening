"""Flask test-client tests for /, /upload and /upload_process."""
import io
import os
import re
import tempfile

import pytest

import app as app_module

FORM = {
    "jobTitle": "Data Scientist",
    "jobDescription": "Looking for a Python, SQL and machine learning expert",
}


@pytest.fixture
def client(monkeypatch, trained_classifier):
    monkeypatch.setattr(app_module, "classifier", trained_classifier)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def post_resume(client, content, filename="synthetic_resume.docx", form=FORM):
    data = dict(form)
    data["resumeUpload"] = (io.BytesIO(content), filename)
    return client.post("/upload_process", data=data,
                       content_type="multipart/form-data")


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"HTML Codex" in response.data
    assert b"Adapted by Samiksha Dhere and Aswin" in response.data


def test_upload_page(client):
    response = client.get("/upload")
    assert response.status_code == 200


def test_upload_process_with_synthetic_docx(client, synthetic_docx):
    response = post_resume(client, synthetic_docx)
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Prediction: Data Science" in body
    assert "Top 3 predicted categories" in body
    top_items = re.findall(r"<li>(\d)\. ([^<]+?) &ndash; ([\d.]+)%</li>",
                           body)
    assert [int(rank) for rank, _, _ in top_items] == [1, 2, 3]
    assert top_items[0][1] == "Data Science"
    probabilities = [float(p) for _, _, p in top_items]
    assert probabilities == sorted(probabilities, reverse=True)
    assert f"Confidence Score: {top_items[0][2]}%" in body
    assert "Keyword overlap:" in body
    for skill in ("python", "sql", "machine learning"):
        assert skill in body


@pytest.mark.parametrize("form, status", [
    ({}, 400),
    ({"jobTitle": "x"}, 400),
    (FORM, 400),  # no file part
])
def test_upload_process_missing_fields(client, form, status):
    response = client.post("/upload_process", data=form)
    assert response.status_code == status


def test_upload_process_rejects_other_file_types(client):
    response = post_resume(client, b"hello", filename="resume.txt")
    assert response.status_code == 400


def test_upload_process_without_model(monkeypatch, synthetic_docx):
    monkeypatch.setattr(app_module, "classifier", None)
    client = app_module.app.test_client()
    response = post_resume(client, synthetic_docx)
    assert response.status_code == 503
    assert b"python train_model.py" in response.data


def _track_temp_files(monkeypatch):
    created = []
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(app_module.tempfile, "mkstemp", tracking_mkstemp)
    return created


def test_temp_file_deleted_after_success(monkeypatch, client, synthetic_docx):
    created = _track_temp_files(monkeypatch)
    assert post_resume(client, synthetic_docx).status_code == 200
    assert len(created) == 1 and not os.path.exists(created[0])


def test_temp_file_deleted_after_failure(monkeypatch, client):
    created = _track_temp_files(monkeypatch)
    response = post_resume(client, b"this is not a real docx file")
    assert response.status_code >= 400
    assert len(created) == 1 and not os.path.exists(created[0])
