import abc
import asyncio.subprocess
import io
import logging
import sqlite3
import sys
import traceback
import warnings
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, TypeVar, IO, TextIO, Type, Iterator, AnyStr, Iterable, Callable, Optional

import aiosqlite
import pandas as pd

import warm

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


class ContextManager(Generic[T], metaclass=abc.ABCMeta):
    """ A generic context manager. """
    def __init__(self) -> None:
        """ Constructs the context manager. """
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """ Whether the connection is opened.

        :return: Whether the connection is opened.
        """
        return self._is_open

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def connection(self) -> T:
        """ Retrieves the underlying raw connection.

        :return: The underlying connection.
        """
        pass

    def open(self) -> T:
        """ Opens the connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._is_open = True
        return self.connection

    def close(self):
        """ Closes the connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        self._is_open = False

    def __enter__(self) -> T:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        if exc_val is not None:
            logger.error(f"Upon {self.name} exit: {exc_val}\n{traceback.format_exc()}")


class AsyncContextManager(Generic[T], metaclass=abc.ABCMeta):
    """ A generic asynchronous context manager. """
    def __init__(self) -> None:
        """ Constructs the context manager. """
        self._is_open = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_open(self) -> bool:
        """ Whether the connection is opened.

        :return: Whether the connection is opened.
        """
        return self._is_open

    @property
    @abc.abstractmethod
    def async_connection(self) -> T:
        """ Retrieves the underlying raw connection.

        :return: The underlying connection.
        """
        pass

    async def async_open(self) -> T:
        """ Opens the connection asynchronously.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._is_open = True
        return self.async_connection

    async def async_close(self) -> None:
        """ Closes the connection asynchrounously.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        self._is_open = False

    async def __aenter__(self) -> aiosqlite.Connection:
        """ Executes when the context manager enters. """
        return await self.async_open()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Executes when the context manager exits. """
        await self.async_close()

        if exc_val is not None:
            logger.error(f"Upon DataManger exit: {exc_val}\n{traceback.format_exc()}")


class DatabaseManager(ContextManager[sqlite3.Connection], AsyncContextManager[aiosqlite.Connection]):
    """ Manages database connection. """
    _is_open: bool
    _is_readonly: bool
    _use_fast: bool
    _database_path: str | Path
    _use_dict_factory: bool
    _commit_on_exit: bool

    # The underlying connection
    _async_conn: aiosqlite.Connection
    _conn: sqlite3.Connection

    def __init__(self,
                 db_path: str | Path,
                 is_readonly: bool = True,
                 use_fast: bool = False,
                 use_dict_factory: bool = False,
                 commit_on_exit: bool = False) -> None:
        """ Constructs the data manager.

        :param is_readonly: Whether the database is read-only.
        :param use_fast: Whether to use fast mode. Warning: using this makes it
        thread-unsafe and may result in a corrupted database.
        """
        super().__init__()
        self._is_readonly = is_readonly
        self._use_fast = use_fast
        self._database_path = db_path
        self._use_dict_factory = use_dict_factory
        self._commit_on_exit = commit_on_exit

    @property
    def is_readonly(self) -> bool:
        """ Whether the database connection is read-only.

        :return: Whether the database connection is read-only.
        """
        return self._is_readonly

    @property
    def connection(self) -> T:
        return self._conn

    @property
    def async_connection(self) -> T:
        return self._async_conn

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
        super().open()
        self._conn = sqlite3.connect(self._database_path)

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
        await super().async_open()
        self._async_conn = await aiosqlite.connect(self._database_path)

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
        super().close()
        if self._commit_on_exit:
            self._conn.commit()

        self._conn.close()
        logger.info("Database connection closed.")

    async def async_close(self) -> None:
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        await super().async_close()
        if self._commit_on_exit:
            await self._async_conn.commit()

        await self._async_conn.close()
        logger.info("Database connection closed.")


class ProcessManager(AsyncContextManager[asyncio.subprocess.Process]):
    """ Manages the process. """
    _is_open: bool
    _program: str
    _args: tuple[str | bytes | PathLike[str] | PathLike[bytes]]
    _kwargs: dict[str, Any]

    # The underlying connection
    _async_proc: asyncio.subprocess.Process
    _proc: asyncio.subprocess.Process

    def __init__(self, program: str, *args: str | bytes | PathLike[str] | PathLike[bytes], **kwargs):
        super().__init__()
        self._program = program
        self._args = args
        self._kwargs = kwargs

    @property
    def async_connection(self):
        return self._async_proc

    async def async_open(self) -> asyncio.subprocess.Process:
        """ Opens the process connection.

        The context management syntax is recommended:
        `async with ProcessManager() as proc_manger: ...`
        """
        await super().async_open()
        self._async_proc = await asyncio.subprocess.create_subprocess_exec(
            self._program, *self._args, **self._kwargs)
        return self._async_proc

    async def async_close(self):
        """ Closes the process connection.

        The context management syntax is recommended:
        `async with ProcessManager() as proc_manger: ...`
        """
        await super().async_close()
        if self._async_proc.returncode is not None:
            warnings.warn("The process returned prematurely.")
        else:
            try:
                self._async_proc.kill()
                await self._async_proc.wait()
            except (InterruptedError, KeyboardInterrupt, asyncio.CancelledError):
                self._async_proc.terminate()


class WriteHandler(metaclass=abc.ABCMeta):
    """ Base handler that supports writing. """
    @abc.abstractmethod
    def write(self, df: Any):
        """ Writes the data.

        :param df: pandas dataframe.
        """
        pass

    def flush(self):
        """ Flush all the outputs into the underlying stream.

        By default, this feature is unused as write immediately triggers flush.
        """
        pass


class DBHandler(WriteHandler):
    _db_path: str
    _table_name: str

    def __init__(self, db_path: str, table_name: str):
        self._db_path = db_path
        self._table_name = table_name

    def write(self, df: pd.DataFrame, **kwargs):
        with DatabaseManager(self._db_path, is_readonly=False) as db:
            df.to_sql("SimInfo", con=db, **kwargs)


class CSVHandler(WriteHandler):
    _dir: PathLike
    _count: int
    _file_factory: Optional[str | Callable[[pd.DataFrame, int], str]]

    def __init__(self, dir: PathLike, file_factory: Optional[str | Callable[[pd.DataFrame, int], str]] = None):
        self._dir = dir
        self._count = 0
        self._file_factory = file_factory

    def write(self, df: pd.DataFrame, **kwargs):
        self._count += 1
        if type(self._file_factory) == "str":
            filename = self._file_factory
        elif self._file_factory is None:
            filename = f"{self._count}.csv"
        else:
            filename = self._file_factory(df, self._count)
        df.to_csv(filename, **kwargs)


class BufferHandler(WriteHandler):
    _buf: str | Path | TextIO

    def __init__(self, buf: str | Path | TextIO = sys.stdout):
        self._buf = buf

    def write(self, df: pd.DataFrame):
        df.to_string(buf=self._buf)


class DataManager(ContextManager["DataManager"]):
    """ An asynchronous data manager. Not thread-safe or process-safe.

    It uses a list under the hood as a storage buffer. The buffer
    is flushed on exit and when the data is full.
    """
    _lim: int
    _data: list[dict]
    _is_open: bool
    _program: str
    _handlers: list[WriteHandler]

    def __init__(self, lim: int = 100_000, handlers: list[WriteHandler] = (BufferHandler(),),
                 flush_on_close: bool = True):
        """ Creates the data manager.

        :param lim: Maximum number of rows of data to store.
        """
        super().__init__()
        self._lim = lim
        if len(handlers) == 0:
            logger.warning("No handlers registered with the data manager."
                           "All data will be lost when flushing!")
        self._handlers = handlers

    @property
    def data(self):
        """ Gives the raw data.

        :return: The raw data.
        """
        return self._data

    @property
    def connection(self):
        return self

    @property
    def lim(self):
        """ Returns the length limit before flush is automatically triggered.

        :returns: The length limit.
        """
        return self._lim

    def flush(self):
        """ Flush the buffered data with the handlers. """
        # For implementation simplicity, create a dataframe to hold the data.
        # And then pipe it to the streams wanted.
        if len(self._data) > 0:
            df = pd.DataFrame(self._data)

            for handler in self._handlers:
                handler.write(df)

            self._data = []

    def write(self, row: dict):
        """ Write a new row of data to the buffer; flush if the buffer is full.

        :param row: Row to insert.
        """
        self._data.append(row)

        if len(self._data) >= self._lim:
            self.flush()

    def open(self):
        super().open()
        self._data = []
        return self

    def close(self):
        super().close()
        self.flush()
