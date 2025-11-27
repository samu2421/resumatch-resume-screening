# ResuMatch 🎯

> **AI-Powered Resume Screening & Job Category Prediction System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [ML Model](#-ml-model)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**ResuMatch** automates resume screening using **Machine Learning** and **NLP**. Upload a resume, and it instantly:
- Predicts job category from 25 options (Data Science, Java Developer, DevOps, etc.)
- Extracts relevant skills using spaCy NLP
- Calculates job match percentage
- Provides confidence scores

**Impact**: Reduces screening time from 20-30 minutes to 2-5 seconds with 84% accuracy.

**Dataset**: Trained on 962 real-world resumes across 25 job categories.

---

## ✨ Key Features

- 🤖 **AI Classification**: Multinomial Naive Bayes + TF-IDF (84% accuracy)
- 📄 **Multi-Format**: Supports PDF and DOCX files
- 🔍 **Skill Extraction**: spaCy NER + fuzzy matching
- 📊 **Job Matching**: Semantic word matching with confidence scores
- 🎨 **Modern UI**: Responsive Bootstrap 5 interface
- 🔒 **Secure**: File validation, 16MB limit, auto-cleanup

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask 3.1.0 |
| **ML** | scikit-learn 1.6.1, TF-IDF + Multinomial NB |
| **NLP** | NLTK 3.9.1, spaCy 3.8.4 |
| **Text Extraction** | PyPDF2, python-docx |
| **Frontend** | Bootstrap 5, jQuery |
| **Server** | Gunicorn 23.0.0 |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/resumatch.git
cd resumatch
python3 -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Run Application
```bash
python app.py
# Visit http://127.0.0.1:5000
```

---

## 💻 Usage

### Web Interface
1. Navigate to `http://127.0.0.1:5000/upload`
2. Enter job title and description
3. Upload resume (PDF/DOCX, max 16MB)
4. Click "Analyze Resume"
5. View results: job category, confidence score, skills, match percentage

### Command Line
```bash
# Test model
python test_model.py

# Run tests
python -m unittest unit_testing.py
python -m unittest integration_testing.py

# Performance testing
locust -f performance_testing.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/upload` | GET | Upload form |
| `/upload_process` | POST | Process resume (returns HTML with results) |
| `/service` | GET | Services page |
| `/project` | GET | Project details |

### Example Request
```bash
curl -X POST http://127.0.0.1:5000/upload_process \
  -F "resumeUpload=@resume.pdf" \
  -F "jobTitle=Data Scientist" \
  -F "jobDescription=Looking for Python and ML expert"
```

---

## 🤖 ML Model

### Architecture
```
Resume Text → Preprocessing (NLTK) → TF-IDF Vectorization → Multinomial Naive Bayes → Prediction
```

### Model Details
- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF (10,000 max features, bigrams)
- **Training Data**: 962 resumes, 25 categories
- **Accuracy**: ~84%
- **Categories**: Data Science, Java Developer, Testing, DevOps Engineer, Python Developer, Web Designing, HR, Hadoop, Blockchain, ETL Developer, and 15 more

### Performance
- **Precision**: ~83%
- **Recall**: ~82%
- **F1-Score**: ~82%
- **Processing Time**: 2-5 seconds per resume

---

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m unittest unit_testing.py -v

# Integration tests
python -m unittest integration_testing.py -v

# Performance tests (Locust)
locust -f performance_testing.py
# Open http://localhost:8089 and configure load test
```

### Test Coverage
- **Unit Tests**: Text preprocessing, skill extraction, model prediction
- **Integration Tests**: API endpoints, file upload, end-to-end workflow
- **Performance Tests**: Load testing with concurrent users

---

## 🚢 Deployment

### Option 1: Gunicorn (Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Option 2: Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t resumatch .
docker run -p 5000:5000 resumatch
```

### Option 3: Heroku
```bash
# Create Procfile: web: gunicorn app:app
heroku create resumatch-app
git push heroku main
```

---

## 🔧 Troubleshooting

### Common Issues

**NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**Model File Not Found**
```bash
python train_model.py  # Train the model first
```

**File Upload Fails (413 Error)**
- File exceeds 16MB limit. Compress or split the resume.

**Low Prediction Confidence (<60%)**
- Resume may not match training categories
- Add more training data for edge cases
- Consider manual review

---

## 🤝 Contributing

Contributions welcome! Here's how:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

**Guidelines**:
- Follow PEP 8 style
- Add tests for new features
- Update documentation

---

## 📜 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 ResuMatch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Samiksha Dhere**  
MSc Advanced Data Science and Analytics

- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **spaCy** - NLP library
- **NLTK** - Natural language toolkit
- **Bootstrap** - Frontend framework

This project was developed as part of the MSc Advanced Data Science and Analytics program.

---

<div align="center">

**Made with ❤️ by Samiksha Dhere**

**ResuMatch** - Transforming Recruitment with AI

[⬆ Back to Top](#resumatch-)

</div>
