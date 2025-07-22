#!/usr/bin/env python
"""Script to load a database session and run the check_model_files function."""

import logging

from rlrmp.database import get_db_session, check_model_files


def main():
    """Load database session and check model files."""
    logging.basicConfig(level=logging.INFO)
    
    session = get_db_session()
    try:
        check_model_files(session, clean_orphaned_files='archive')
    finally:
        session.close()


if __name__ == "__main__":
    main()