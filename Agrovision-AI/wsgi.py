"""
WSGI config for Agrovision_AI project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'Agrovision_AI.settings'
)

application = get_wsgi_application()
