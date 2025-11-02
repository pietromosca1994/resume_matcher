
import logging
from abc import ABC
from functools import wraps
import inspect
from time import time
# from pydantic import BaseModel
from typing import Union

from .config_types import BaseConfig
from .ctx_types import BaseCtx

class BaseClass(ABC):
    def __init__(self,
                 config: Union[BaseConfig, None]=None,
                 ctx: Union[BaseCtx, None]=None,
                 verbose: int=logging.INFO):
        # initialize logger
        self._init_logger(verbose)

        # setup configs
        self.config: BaseConfig = config

        # setup context
        self.ctx: BaseCtx = ctx

    @property
    def config(self):
        return self._config
        
    @config.setter
    def config(self, value: Union[BaseConfig, None]):
        self._config=value

        if self._config is not None:
            self.logger.info(f'{self.__class__.__name__} configuration:\n{self._log_pretty(self._config.to_dict(), indent=2)}')

    @property 
    def ctx(self): 
        return self._ctx
    
    @ctx.setter
    def ctx(self, value: Union[BaseCtx, None]):
        self._ctx=value

        if self._ctx is not None:
            self.logger.info(f'{self.__class__.__name__} context:\n{self._log_pretty(self._ctx.to_dict(), indent=2)}')
    
    def _init_logger(self, verbose: int=logging.INFO):
        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)  # Get a logger unique to the class
        self.logger.setLevel(verbose)  # Set the logging level
        
        # Check if handlers are already added (to prevent duplicate logs)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # Console handler (logs to terminal)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Optional: File logging (uncomment if needed)
            # file_handler = logging.FileHandler(f"{self.__class__.__name__}.log")
            # file_handler.setFormatter(formatter)
            # self.logger.addHandler(file_handler)

            # Prevent logs from propagating to the root logger
            self.logger.propagate = False

    @staticmethod
    def _log_pretty(obj, indent=0):
        """
        Pretty-print dicts, lists, and nested structures with indentation and colors.
        Returns a formatted string.
        """
        OKBLUE = "\033[94m"
        END = "\033[0m"
        lines = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{'  '*indent}{OKBLUE}{k}{END}:")
                    lines.append(BaseClass._log_pretty(v, indent + 1))
                else:
                    lines.append(f"{'  '*indent}{OKBLUE}{k}{END}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{'  '*indent}- {BaseClass._log_pretty(item, indent + 1).lstrip()}")
                else:
                    lines.append(f"{'  '*indent}- {item}")
        else:
            lines.append(f"{'  '*indent}{obj}")

        return "\n".join(lines)
    
    @staticmethod
    def _monitor_execution_time(func=None, *, warn_threshold=None):
        """
        Decorator to monitor the execution time of a function or method.
        Can be used with or without parentheses:
            @_monitor_execution_time
            @_monitor_execution_time(warn_threshold=5)
        """
        def decorator(inner_func):
            @wraps(inner_func)
            def wrapper(*args, **kwargs):
                start_time = time()
                result = inner_func(*args, **kwargs)
                execution_time = time() - start_time

                # Resolve class name if called on a class instance
                class_name = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else None
                func_name = inner_func.__name__

                target_logger = getattr(args[0], "logger", logging.getLogger(inner_func.__module__))
                prefix = f"{class_name + '.' if class_name else ''}{func_name}"
                msg = f"⏰ Method '{prefix}' executed in {execution_time:.2f}s"

                if warn_threshold and execution_time > warn_threshold:
                    target_logger.warning(msg + f" ⚠️ Exceeded {warn_threshold:.3f}s")
                else:
                    target_logger.info(msg)

                return result
            return wrapper

        if func is not None and callable(func):
            return decorator(func)
        return decorator
    