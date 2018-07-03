import inspect
import logging
from io import StringIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stream = StringIO()
handler = logging.StreamHandler(stream)


class IndentFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.baseline = len(inspect.stack())

    def format(self, rec):
        stack = inspect.stack()
        rec.indent = ' ' * (len(stack)-self.baseline)
        out = logging.Formatter.format(self, rec)
        del rec.indent
        return out


formatter = IndentFormatter(fmt='%(indent)s %(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.info("logger initialized")
