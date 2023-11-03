import os
import site
import sys
from django.core.wsgi import get_wsgi_application

# Add the app’s directory to the PYTHONPATH


sys.path.append("C:/Users/joy14/PycharmProjects/AIServer/AI")
sys.path.append("C:/Users/joy14/PycharmProjects/AIServer/AI/AI")
#sys.path.append("C:/Users/joy14/PycharmProjects/AIServer/new_venv/Lib/site-packages")
site.addsitedir("C:/Users/joy14/PycharmProjects/AIServer/new_venv/Lib/site-packages")

'''
path = os.path.abspath(__file__+'/..')
if path not in sys.path:
    sys.path.append(path)
'''
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI.settings')
os.environ['DJANGO_SETTINGS_MODULE'] = 'AI.settings'


application = get_wsgi_application()
