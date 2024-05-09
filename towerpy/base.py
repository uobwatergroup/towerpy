"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import traceback
import logging


class TowerpyError(Exception):
    """Error towerpy class."""

    def __init__(self, message, inner=None):
        super(TowerpyError, self).__init__()
        self.message = 'TowerpyError: {}'.format(message)
        self.inner = inner
        self.traceback = traceback.format_stack()

    def __str__(self):
        return self.message


class TowerpyLog(object):
    """Log Towerpy class."""

    def __init__(self, level=None):
        self.logger = logging.getLogger()
        if level:
            handler = logging.StreamHandler()
            self.logger.setLevel(level)
        else:
            handler = logging.NullHandler()
        self.logger.addHandler(handler)

    def debug(self, *args):
        self.logger.debug(args[0].format(*args[1:]))

    def info(self, *args):
        self.logger.info(args[0].format(*args[1:]))

    def warning(self, *args):
        self.logger.warning(args[0].format(*args[1:]))

    def error(self, *args):
        self.logger.error(args[0].format(*args[1:]))

    def exception(self, e):
        self.logger.error('{} {}', e, traceback.format_exc())


thelog = TowerpyLog()
