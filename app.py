"""
Flask application for resume screening and job category prediction.

This module provides a web interface for uploading and analyzing resumes,
using machine learning to predict job categories and extract skills.
"""

import os
import secrets
import logging
from typing import Any

from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import from local modules
from train_model import ResumeClassifier
from preprocess import ResumePreprocessor, extract_resume_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('resume_screening.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Security and upload configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = secrets.token_hex(16)

# Allowed file extensions
ALLOWED_EXTENSIONS: set[str] = {'pdf', 'docx'}

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
classifier = ResumeClassifier()
preprocessor = ResumePreprocessor()


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the file to check.

    Returns:
        bool: True if file extension is allowed, False otherwise.
    """
    return (
        '.' in filename and 
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def index() -> str:
    """
    Render the main index page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')


@app.route('/upload')
def upload() -> str:
    """
    Render the upload page.

    Returns:
        str: Rendered HTML template for the upload page.
    """
    return render_template('upload.html')


@app.route('/upload_process', methods=['POST'])
def upload_process() -> Any:
    """
    Process resume upload and perform analysis.

    Returns:
        Any: Rendered result template or error response.
    """
    try:
        job_title = request.form.get('jobTitle')
        if not job_title:
            return "Job title is required", 400

        if 'resumeUpload' not in request.files:
            return "No file part", 400

        resume_file = request.files['resumeUpload']

        if not resume_file.filename:
            return "No selected file", 400

        if not allowed_file(resume_file.filename):
            return "Invalid file type. Only PDF and DOCX allowed.", 400

        # Secure filename
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        resume_file.save(filepath)

        # Extract resume text
        resume_text = extract_resume_text(filepath)

        # Clean the resume text
        cleaned_text = preprocessor.clean_text(resume_text)

        # Extract skills
        skills = preprocessor.extract_skills(resume_text)

        # Prediction
        prediction_result = classifier.predict_category(cleaned_text)

        # Cleanup: Remove uploaded file
        os.remove(filepath)

        # Render result template with prediction data
        return render_template(
            'result.html',
            job_category=prediction_result[0],
            confidence_score=round(float(prediction_result[1]) * 100, 2),
            job_match=75,  # You can modify this with actual matching logic
            total_skills=len(skills),
            skills=skills
        )

    except Exception as e:
        logger.error('Upload processing error: %s', str(e))
        return "Resume processing failed", 500


@app.route('/service')
def service() -> str:
    """
    Render the service page.

    Returns:
        str: Rendered HTML template for the service page.
    """
    return render_template('service.html')


@app.route('/project')
def project() -> str:
    """
    Render the project page.

    Returns:
        str: Rendered HTML template for the project page.
    """
    return render_template('project.html')


if __name__ == '__main__':
    app.run(debug=True)
