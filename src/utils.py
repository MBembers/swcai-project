# src/utils.py

import time
from functools import wraps

def log_time(name, start_time):
    """
    Logs the elapsed time since start_time with a label, and returns a new timestamp.
    """
    end_time = time.perf_counter()
    print(f"{name} took {end_time - start_time:.6f} seconds")
    return time.perf_counter()  # return new start for next measurement

TIMINGS = {}

def timeit(task_dict=None):
    """
    Decorator to measure execution time of a function and store it in a dict.
    
    Args:
        task_dict (dict, optional): dictionary to store timings. Defaults to TIMINGS.
    """
    if task_dict is None:
        task_dict = TIMINGS

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            task_dict[func.__name__] = end - start
            return result
        return wrapper
    return decorator
