"""
This script is used to simulate a user uploading a resume
to the resume screening application.
"""
from locust import HttpUser, task, between
import requests
from typing import Any


class ResumeScreeningUser(HttpUser):
    """Simulate a user uploading resume to the resume screening application"""
    wait_time = between(1, 3)
    host = "http://127.0.0.1:5000"

    @task
    def upload_resume(self):
        """Simulate a user uploading a resume"""
        try:
            with open("sample_resume.pdf", "rb") as file:
                with self.client.post(
                    "/upload_process",
                    files={"resumeUpload": file},
                    data={
                        "jobTitle": "Software Engineer",
                        "jobDescription": "Looking for Python developer"
                    },
                    catch_response=True
                ) as response:
                    print(f"Response status: {response.status_code}")
                    if response.status_code == 200:
                        print("Success!")
                        # This is a valid method on LocustResponse objects, even if PyLance doesn't recognize it
                        if hasattr(response, 'success'):
                            response.success()
                    else:
                        print(f"Error response: {response.text[:50]}...")
                        if hasattr(response, 'failure'):
                            response.failure(f"Failed with status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"HTTP request error occurred: {str(e)}")
        except FileNotFoundError as e:
            print(f"File error occurred: {str(e)}")
