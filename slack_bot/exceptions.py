import logging

class NoThreadCreatedError(Exception):
    """Raised when no thread_id is provided and no thread has been created yet."""
    logging.error('No thread_id provided and no thread created yet.')
