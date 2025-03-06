# test_model.py
from train_model import ResumeClassifier

# Test with different resume texts representing different job categories
test_texts = {
    "Java Developer": "Experienced Java developer with Spring Boot and Hibernate. Created RESTful APIs and microservices.",
    "Testing": "QA engineer with experience in manual and automated testing using Selenium and JUnit.",
    "Data Science": "Data scientist with expertise in Python, pandas, scikit-learn and TensorFlow."
}

classifier = ResumeClassifier()

for category, text in test_texts.items():
    pred_category, confidence = classifier.predict_category(text)
    print(f"Expected: {category}, Predicted: {pred_category}, Confidence: {confidence:.2f}")