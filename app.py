"""
Flask application for resume screening and job category prediction.
"""
import os
import secrets
import logging
from typing import Any
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
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

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

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
def index():
    """
    Render the main index page.
    """
    return render_template('index.html')


@app.route('/upload')
def upload() -> str:
    """
    Render the upload page.
    """
    return render_template('upload.html')


@app.route('/upload_process', methods=['POST'])
def upload_process() -> Any:
    """
    Process resume upload and perform analysis.
    """
    try:
        # Get form data
        job_title = request.form.get('jobTitle')
        job_description = request.form.get('jobDescription')

        # Validate required fields
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

        # Secure filename and save file
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)

        # Extract resume text
        resume_text = extract_resume_text(filepath)

        # Clean the resume text
        cleaned_text = preprocessor.clean_text(resume_text)

        # Extract skills using spaCy-based method
        skills = preprocessor.extract_skills_spacy(resume_text)

        # Calculate job match score
        job_match = calculate_job_match(cleaned_text, job_description)

        # Prediction with probability confidence
        prediction_result, confidence = classifier.predict_category(
            cleaned_text)
        confidence_score = round(float(confidence) * 100, 2)

        # Cleanup: Remove uploaded file
        os.remove(filepath)

        # Render result template
        return render_template(
            'result.html',
            job_title=job_title,
            job_description=job_description,
            job_category=prediction_result,
            confidence_score=confidence_score,
            job_match=job_match,
            total_skills=len(skills),
            skills=skills
        )

    except FileNotFoundError as e:
        logger.error('File not found error: %s', str(e))
        return "File not found", 404

    except ValueError as e:
        logger.error('Value error: %s', str(e))
        return "Invalid input", 400

    except OSError as e:
        logger.error('File handling error: %s', str(e))
        return "File handling error", 500

    except RuntimeError as e:
        logger.error('Runtime error: %s', str(e))
        return "Runtime error occurred", 500

    except Exception as e:
        logger.error('Unexpected error: %s', str(e))
        return "An unexpected error occurred", 500


def calculate_job_match(resume_text: str, job_description: str) -> int:
    """
    Calculate how well the resume matches the job description.

    Args:
        resume_text (str): Cleaned text from the resume
        job_description (str): Job description from the form

    Returns:
        int: Match percentage (0-100)
    """
    # Clean the job description using the same preprocessor
    cleaned_job_desc = preprocessor.clean_text(job_description)

    # Convert texts to sets of words for comparison
    resume_words = set(resume_text.lower().split())
    job_words = set(cleaned_job_desc.lower().split())

    # Find common words (excluding very common words)
    common_words = resume_words.intersection(job_words)

    # Calculate match score with adjusted scaling
    if len(job_words) == 0:
        return 50  # Default value if job description is empty

    match_percentage = min(100, int(len(common_words) / len(job_words) * 120))

    return match_percentage


@app.route('/service')
def service() -> str:
    """
    Render the service page.
    """
    return render_template('service.html')


@app.route('/project')
def project() -> str:
    """
    Render the project page.
    """
    return render_template('project.html')


if __name__ == '__main__':
    app.run(debug=True)
