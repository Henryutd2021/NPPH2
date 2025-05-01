"""Centralised logging initialisation used by all modules."""
import logging
from config import LOG_FILE, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    filemode="w",  # overwrite on every run
)

logger = logging.getLogger(__name__)
