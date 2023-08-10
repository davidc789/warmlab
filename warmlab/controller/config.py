import logging.config

logger = logging.getLogger(__name__)

class Config():
    # Location of the database.
    DB_LOCATION = "../../data.sqlite"

    # Frequency of database entry.
    TIME_STEP = 1_000_000

config = Config()
