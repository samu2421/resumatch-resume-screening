"""This module contains the Flask application for the Resume Screening API"""

import os
import secrets
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import from local modules
from train_model import ResumeClassifier
from preprocess import ResumePreprocessor, extract_resume_text

# Configure logging with lazy formatting
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

# Security configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = secrets.token_hex(16)  # Secure secret key

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
classifier = ResumeClassifier()
preprocessor = ResumePreprocessor()


def allowed_file(filename):
    """
    Validate file extension and MIME type

    Args:
        filename (str): Name of the file to validate

    Returns:
        bool: True if file is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_resume():
    """
    Secure resume upload and screening endpoint

    Returns:
        JSON response with prediction and processing details
    """
    try:
        job_title = request.form.get('job_title')
        if not job_title:
            logger.warning('No job title provided')
            return jsonify({
                'error': 'Job title is required',
                'status': 'failed'
            }), 400

        # Check if resume file is present
        if 'resume' not in request.files:
            logger.warning('No file part in the request')
            return jsonify({
                'error': 'No file part',
                'status': 'failed'
            }), 400

        resume_file = request.files['resume']

        # Additional security checks
        if not resume_file.filename:
            logger.warning('No selected file')
            return jsonify({
                'error': 'No selected file',
                'status': 'failed'
            }), 400

        if not allowed_file(resume_file.filename):
            logger.warning('Invalid file type: %s', resume_file.filename)
            return jsonify({
                'error': 'Invalid file type. Only PDF and DOCX allowed.',
                'status': 'failed'
            }), 400

        # Secure filename
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file securely
        resume_file.save(filepath)

        # Log upload activity without sensitive details
        logger.info('File uploaded: %s', filename)

        try:
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

            return jsonify({
                'prediction': prediction_result[0],
                'confidence': float(prediction_result[1]),
                'skills': skills,
                'status': 'success'
            })

        except ValueError as process_error:
            logger.error('Processing error: %s', str(process_error))
            # Remove file if processing fails
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'error': 'Resume processing failed',
                'status': 'failed'
            }), 500

    except (IOError, OSError, RuntimeError) as e:
        logger.critical('Critical upload error: %s', str(e))
        return jsonify({
            'error': 'Upload failed',
            'status': 'failed'
        }), 500


# Optional: Add a home route
@app.route('/', methods=['GET'])
def home():
    """
    Home route for the application
    """
    return "Resume Screening Application"


if __name__ == '__main__':
    app.run(debug=True)
