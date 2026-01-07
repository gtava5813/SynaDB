# WSGI configuration for PythonAnywhere

import sys

# Add your project directory to the path
path = '/home/gtava5813/mysite'
if path not in sys.path:
    sys.path.append(path)

# Import the Flask app
from flask_app import app as application
