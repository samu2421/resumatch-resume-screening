"""Integration tests for the Resume Screening API"""
import unittest
from app import app


class TestResumeScreeningAPI(unittest.TestCase):
    """Test the Resume Screening API"""
    def setUp(self):
        """Set up the test client before running tests"""
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_upload_page(self):
        """Test if the upload page loads correctly"""
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)

    def test_upload_process_missing_file(self):
        """Test if the API handles missing file error"""
        response = self.client.post('/upload_process', data={})
        self.assertEqual(response.status_code, 400)

    def test_upload_process_with_resume(self):
        """Test if the API processes a valid resume correctly"""
        data = {
            'jobTitle': 'Data Scientist',
            'jobDescription': 'Looking for a Python and ML expert'
        }

        with open('sample_resume.pdf', 'rb') as resume:
            files = {'resumeUpload': (resume, 'sample_resume.pdf',
                                      'application/pdf')}

            response = self.client.post(
                '/upload_process',
                data={**data, **files},
                content_type='multipart/form-data'
            )

        # Debug: Print the response to check what is returned
        print(response.data.decode())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction:', response.data)


if __name__ == '__main__':
    unittest.main()
