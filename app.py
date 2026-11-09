"""
Flask application for resume screening and job category prediction.

Run locally:   python app.py
Production:    gunicorn app:app
"""
import logging
import os
import secrets
import tempfile
from typing import Any

from flask import Flask, request, render_template

# Importing preprocess also makes sure the NLTK resources are available.
from preprocess import ResumePreprocessor, extract_resume_text
from train_model import MODEL_PATH, ResumeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(16)

ALLOWED_EXTENSIONS: set[str] = {'pdf', 'docx'}

MODEL_MISSING_MESSAGE = (
    f"The trained model ({MODEL_PATH.name}) was not found in model/. "
    "Train it first with: python train_model.py"
)

preprocessor = ResumePreprocessor()

# Load the trained pipeline once at startup, not per request.
try:
    classifier = ResumeClassifier.load(MODEL_PATH)
    logger.info('Model loaded from %s', MODEL_PATH)
except FileNotFoundError:
    classifier = None
    logger.warning(MODEL_MISSING_MESSAGE)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')


@app.route('/upload')
def upload() -> str:
    """Render the upload page."""
    return render_template('upload.html')


@app.route('/upload_process', methods=['POST'])
def upload_process() -> Any:
    """Process resume upload and perform analysis."""
    job_title = request.form.get('jobTitle')
    job_description = request.form.get('jobDescription')

    if not job_title:
        return "Job title is required", 400
    if not job_description:
        return "Job description is required", 400
    if 'resumeUpload' not in request.files:
        return "No file part", 400

    resume_file = request.files['resumeUpload']
    if not resume_file.filename:
        return "No selected file", 400
    if not allowed_file(resume_file.filename):
        return "Invalid file type. Only PDF and DOCX allowed.", 400

    if classifier is None:
        return MODEL_MISSING_MESSAGE, 503

    extension = '.' + resume_file.filename.rsplit('.', 1)[1].lower()
    fd, temp_path = tempfile.mkstemp(suffix=extension)
    os.close(fd)
    try:
        resume_file.save(temp_path)
        resume_text = extract_resume_text(temp_path)

        skills = preprocessor.extract_skills(resume_text)
        keyword_overlap = preprocessor.keyword_overlap(resume_text,
                                                       job_description)
        # predict_top cleans the raw text itself (same as training)
        top_predictions = [
            (category, round(probability * 100, 2))
            for category, probability in classifier.predict_top(
                resume_text, k=3)
        ]
        job_category, confidence_score = top_predictions[0]

        return render_template(
            'result.html',
            job_title=job_title,
            job_description=job_description,
            job_category=job_category,
            confidence_score=confidence_score,
            top_predictions=top_predictions,
            keyword_overlap=keyword_overlap,
            total_skills=len(skills),
            skills=skills
        )

    except ValueError as e:
        logger.error('Value error: %s', e)
        return "Invalid input", 400
    except OSError as e:
        logger.error('File handling error: %s', e)
        return "File handling error", 500
    except Exception:  # pylint: disable=broad-except
        logger.exception('Unexpected error while processing a resume')
        return "An unexpected error occurred", 500
    finally:
        # Always remove the uploaded file, also when processing fails.
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


@app.route('/service')
def service() -> str:
    """Render the service page."""
    return render_template('service.html')


@app.route('/project')
def project() -> str:
    """Render the project page."""
    return render_template('project.html')


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    app.run(host=host, port=port, debug=debug)
