import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  
ch = logging.StreamHandler() 
formatter = logging.Formatter("    \033[90m%(levelname)s: %(message)s\033[0m")
ch.setFormatter(formatter)
logger.addHandler(ch)


def repeat_decorator(limit, pause_s=0):
    def limited_repeat_decorator(function):
        def decorated(*args, **kwargs):
            last_exception = None
            for t in range(limit):
                try: 
                    logger.debug(f"Repeat {t} running function {function.__module__}.{function.__name__}.")
                    return function(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if pause_s: 
                        logger.debug(f"Exception, pausing to repeat after {pause_s}s. Exception: {e}")
                        time.sleep(pause_s)                        

            raise last_exception
        return decorated 
    return limited_repeat_decorator
