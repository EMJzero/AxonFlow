from typing import Iterable, TypeVar, Generator, Any, Callable

from multiprocessing import current_process
from contextlib import contextmanager
from prettytable import PrettyTable
from termcolor import colored
from itertools import islice
import functools
import builtins
import colorsys
import time

from settings import *

T = TypeVar('T')

"""
Custom print to be used inside 'core' functions.
"""
@contextmanager
def hijack_print(prefix):
    original_print = builtins.print
    def custom_print(*args, **kwargs):
        if Settings.VERBOSE:
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            lines = sep.join(str(arg) for arg in args).split('\n')
            first_prefix = f"[{prefix}]" if not Settings.MULTIPROCESSING else f"[{prefix}@{current_process().name}]"
            other_prefix = ' '*len(first_prefix) + '└▶'
            for i, line in enumerate(lines):
                original_print(colored(first_prefix if i == 0 else other_prefix, Settings.VERBOSE_COLOR), line, end = end if i == len(lines) - 1 else '\n')
    builtins.print = custom_print
    try:
        yield
    finally:
        builtins.print = original_print

"""
Custom tag to be used like:
```
@core
def function():
    print("test")
```
This will result in the function's name being logged upon call and in every
print inside the function being preceded by the function's name.
"""
def core(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = f"{func.__name__}" if not Settings.MULTIPROCESSING else f"{func.__name__}@{current_process().name}"
        if Settings.VERBOSE:
            print(colored(f"[function call: {name}]", Settings.VERBOSE_COLOR))
            if Settings.TIMING:
                start = time.perf_counter()
        with hijack_print(func.__name__):
            result = func(*args, **kwargs)
        if Settings.VERBOSE and Settings.TIMING:
            end = time.perf_counter()
            elapsed = end - start
            print(colored(f"[{name}]", Settings.VERBOSE_COLOR), f"finished in {elapsed:.6f} seconds")
        return result
    return wrapper

"""
Returns a print function that, when called, prints only if 'interval'
time has passed since the last print and 'condition' is True.

Arguments:
- interval: imposed interval between prints in seconds.
- condition: condition to be satisfied for the print to occur.
"""
def getTimerPrinter(interval : int, condition : Callable = lambda : True):
    last_print_time = time.monotonic()
    def printer(string : str):
        nonlocal last_print_time
        now = time.monotonic()
        if now - last_print_time >= interval and condition():
            last_print_time = now
            print(string)
    return printer

"""
Returns a string with a pretty textual representation of the provided dictionary.
"""
def prettyFormatDict(dictionary : dict, indent_level : int = 0) -> str:
    string = ""
    for key, value in (dictionary.items() if isinstance(dictionary, dict) else zip(["" for i in dictionary], dictionary)):
        string += '    '*indent_level + (f"{key}: " if key != "" else "- ")
        if isinstance(value, dict):
            string += "\n" + prettyFormatDict(value, indent_level + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            string += "\n" + prettyFormatDict(value, indent_level + 1)
        else:
            string += str(value)
        string += "\n"
    return string.rstrip()

"""
Prints a nicely formatted textual representation of the provided dictionary.
"""
def prettyPrintDict(dictionary : dict, indent_level : int = 0) -> None:
    print(prettyFormatDict(dictionary, indent_level))

"""
Given an iterable, prints its elements in an equi-spaced grid with each row having
'elem_per_row' (default is 4) elements.
"""
def prettyPrintIterable(iterable : Iterable, elem_per_row : int = 4) -> None:
    def iter_in_chunks(iterable : Iterable[T], chunk_size : int) -> Generator[None, list[T], None]:
        it = iter(iterable)
        while True:
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            yield chunk

    table = PrettyTable([i for i in range(elem_per_row)])
    table.header = False
    table.border = False
    for chunk in iter_in_chunks(iterable, elem_per_row):
        table.add_row(chunk + ['' for _ in range(elem_per_row - len(chunk))])
    print(table)

"""
Given a file size in bytes, returns a string representing it in the closest unit
of measure between B, KB, MB, GB, and TB.
"""
def fileSizeString(filesize : int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(filesize)
    for unit in units:
        if size < 1024 or unit == 'TB':
            return f"{size:.1f} {unit}"
        size /= 1024

"""
Generates a sequence of RGB color tuples that are maximally
distinct from one another and readable on black.
"""
def color_generator(brightness : float = 1.0, saturation : float = 1.0) -> Generator[None, tuple[int, int, int], None]:
    golden_ratio_conjugate = 0.61803398875
    hue = 0.0 # initial hue

    while True:
        # generate color in HSV, then convert to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        yield (int(r * 255), int(g * 255), int(b * 255))
        hue = (hue + golden_ratio_conjugate) % 1.0