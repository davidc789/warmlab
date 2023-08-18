import logging
import sqlite3

import aiosqlite

logger = logging.getLogger(__name__)


def dict_factory(cursor: aiosqlite.Cursor, row: aiosqlite.Row):
    """ Generates a dictionary for query results.

    :param cursor: The sqlite cursor.
    :param row: The native row object.
    :return: Dictionary representation of the row object.
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class DataManager(object):
    """ Manages database connection. """
    _is_open: bool
    _is_readonly: bool
    _use_fast: bool
    _database_path: str
    _use_dict_factory: bool
    _commit_on_exit: bool

    # The underlying connection
    _async_conn: aiosqlite.Connection
    _conn: sqlite3.Connection

    def __init__(self,
                 db_path: str,
                 is_readonly: bool = True,
                 use_fast: bool = False,
                 use_dict_factory: bool = False,
                 commit_on_exit: bool = False) -> None:
        """ Constructs the data manager.

        :param is_readonly: Whether the database is read-only.
        :param use_fast: Whether to use fast mode. Warning: using this makes it
        thread-unsafe and may result in a corrupted database.
        """
        self._is_open = False
        self._is_readonly = is_readonly
        self._use_fast = use_fast
        self._database_path = db_path
        self._use_dict_factory = use_dict_factory
        self._commit_on_exit = commit_on_exit

    @property
    def is_open(self) -> bool:
        """ Whether the database connection is opened.

        :return: Whether the database connection is opened.
        """
        return self._is_open

    @property
    def is_readonly(self) -> bool:
        """ Whether the database connection is read-only.

        :return: Whether the database connection is read-only.
        """
        return self._is_readonly

    async def setup(self) -> None:
        """ Sets up the database for the first time. """
        await self.async_open()
        await self.async_close()

    async def enable_edit(self) -> None:
        """ Enables database editing. """
        await self._async_conn.execute("PRAGMA QUERY_ONLY = FALSE;")

    async def disable_edit(self) -> None:
        """ Disables database editing, making it read-only. """
        await self._async_conn.execute("PRAGMA QUERY_ONLY = TRUE;")

    def open(self) -> sqlite3.Connection:
        """ Opens the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._conn = sqlite3.connect(self._database_path)
        self._is_open = True

        if self._use_dict_factory:
            self._conn.row_factory = dict_factory

        self._conn.execute("PRAGMA foreign_keys = ON")

        if self._use_fast:
            self._conn.execute("PRAGMA synchronous = OFF")
            self._conn.execute("PRAGMA journal_mode = MEMORY")

        logger.info("Database connection established.")
        return self._conn

    async def async_open(self) -> aiosqlite.Connection:
        """ Opens the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._async_conn = await aiosqlite.connect(self._database_path)
        self._is_open = True

        if self._use_dict_factory:
            self._async_conn.row_factory = dict_factory

        await self._async_conn.execute("PRAGMA foreign_keys = ON")

        if self._use_fast:
            await self._async_conn.execute("PRAGMA synchronous = OFF")
            await self._async_conn.execute("PRAGMA journal_mode = MEMORY")

        logger.info("Database connection established.")
        return self._async_conn

    def close(self):
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        if self._commit_on_exit:
            self._conn.commit()

        self._conn.close()
        self._is_open = False
        logger.info("Database connection closed.")

    async def async_close(self) -> None:
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        if self._commit_on_exit:
            await self._async_conn.commit()

        await self._async_conn.close()
        self._is_open = False
        logger.info("Database connection closed.")

    def __enter__(self) -> sqlite3.Connection:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self) -> aiosqlite.Connection:
        """ Executes when the context manager enters. """
        return await self.async_open()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Executes when the context manager exits. """
        await self.async_close()

        if exc_val is not None:
            logger.error(f"Upon DataManger exit: {exc_val}")
