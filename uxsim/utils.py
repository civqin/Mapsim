"""
General utilities for UXsim.
"""

import warnings
import functools
import traceback
import sys

# 汎用関数

def catch_exceptions_and_warn(warning_msg=""):
    """
    A decorator that catches exceptions in the decorated function and raises a warning with the specified message.

    Parameters
    ----------
    warning_msg : str, optional
        A custom message to be displayed along with the exception details. Default is an empty string.

    Returns
    -------
    decorator
        A decorator that wraps the decorated function to catch exceptions and raise warnings.

    Notes
    -----
    When an exception occurs in the decorated function, the decorator will raise a warning that includes:
    - The name of the function.
    - A custom message, if provided.
    - The type of the exception.
    - The exception message.
    - The filename and line number where the decorated function was called.

    This decorator is for inessential functions where random exceptions are expected (especially due to file I/O or something), but you want to avoid crashing the entire program and instead receive a warning. Mainly written by ChatGPT.
    """
    warnings.simplefilter('default', category=UserWarning)
    warnings.formatwarning = lambda message, category, filename, lineno, line: f"{category.__name__}: {message}\n"
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                #warnings.warn(f"{func.__name__}(){warning_msg}: {type(e).__name__}: {str(e)}")
                current_frame = sys._getframe()
                caller_frame = traceback.extract_stack(current_frame, limit=2)[0]
                filename = caller_frame.filename
                lineno = caller_frame.lineno
                warnings.warn(f"{filename}:{lineno}:{func.__name__}(){warning_msg}: {type(e).__name__}: {str(e)}")
        return wrapper
    return decorator

def lange(l):
    """
    Super lazy function for abbreviating range(len(l))

    Parameters
    ----
    l : list
    """
    return range(len(l))

def printtry(*args, **kwargs):
    """
    Print messages safely, evaluating any callable arguments and catching exceptions. Mainly written by GPT-4

    Parameters
    ----------
    *args : any
        The contents to print wrapped by lambda. E.g., lambda: (x, y, z)
    **kwargs : any
        Arbitrary keyword arguments. These are passed directly to the built-in print function.

    Examples
    --------
    >>> printtry(lambda: (mylist[100], mylist[1000], mylist[10000]))
    >>> printtry(lambda: (mylist[100], mylist[1000], mylist[10000]), sep=";", end="!")
    """
    try:
        # Evaluate each argument and print them
        evaluated_args = [arg() if callable(arg) else arg for arg in args][0]
        print(*evaluated_args, **kwargs)
    except Exception as e:
        # Print the exception if it occurs
        print("EXCEPTION:", e)
    

def get_font_for_matplotlib():
    """
    Get a multi-language (currently Japanese only) font for matplotlib.
    TODO: check if it works on different OS. It is only checked on Japanese Windows. 
    TODO: explore if it can be extended for other languages.

    Returns
    -------
    str
        The name of a Japanese font that is available on the system. If no Japanese font is found, "monospace" is returned.
    """
    from matplotlib import font_manager

    font_list = font_manager.findSystemFonts()

    japanese_font = None

    if any("Noto Sans Mono CJK JP" in font.lower() for font in font_list):
        japanese_font = "Noto Sans Mono CJK JP"
    elif any("msgothic" in font.lower() for font in font_list):
        japanese_font = "MS Gothic"
    else:
        japanese_font = "monospace"
        
    return japanese_font

def print_columns(*lists):
    """
    Convinient function to check contents of 1d lists. For debug.
    """
    # Determine the maximum length of the lists
    max_length = max(len(lst) for lst in lists)

    # Iterate through the maximum number of rows
    for i in range(max_length):
        # For each list, print the element at the current row or space if out of range
        for lst in lists:
            if i < len(lst):
                try:
                    print(f"{lst[i]:<10}", end=" ")  # Adjust the width as necessary
                except TypeError:
                    print(f"{str(lst[i]):<10}", end=" ")
            else:
                print(" " * 10, end=" ")  # Adjust spacing to match the above width
        print()  # Newline after each row

class LoggingWarning(UserWarning):
    """
    This warns that when vehicle_logging_timestep_interval is not 1 but called vehicle logging-related functions.
    """
    pass