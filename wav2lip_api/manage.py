#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# --- FIX for ModuleNotFoundError ---
# This adds the parent directory (project root) and the Wav2Lip directory
# to Python's search path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
WAV2LIP_PATH = os.path.join(PARENT_ROOT, 'Wav2Lip')

if PARENT_ROOT not in sys.path:
    sys.path.append(PARENT_ROOT)
if WAV2LIP_PATH not in sys.path:
    sys.path.append(WAV2LIP_PATH)
# --- END OF FIX ---

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wav2lip_api.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
