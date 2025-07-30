from __future__ import annotations

from typing import TypeVar, Generator, Callable, Iterable, Optional, Any
from collections import defaultdict
import multiprocessing
import itertools
import threading
import traceback
import textwrap
import inspect
import signal
import ast
import re

from datastructures import *
from settings import *
from prints import *

# MULTIPROCESSING:

# Patch 'signal' on platforms that don't support the alarm signal
if not hasattr(signal, "alarm"):
    def _no_alarm(_):
        pass
    signal.alarm = _no_alarm
    signal.SIGALRM = 22 # "22" is SIGABRT, the true SIGALRM would be "14"

"""
Defer polling operations to a thread.
As soon as 'condition' is satisfied, 'func' is called with the provided arguments.
"""
def wait_and_retry(func : Callable, condition : Callable[[], bool], check_interval : int, *args : tuple[Any, ...], **kwargs : dict[str, Any]) -> None:
    def check():
        if condition():
            func(*args, **kwargs)
        else:
            threading.Timer(check_interval, check).start()
    check()

"""
Spawns and immediately starts a new process to run 'func'.
Constructing this is non-blocking, the process will start as soon as an instance is available.
"""
class Worker():
    # global counter for assigning custom process IDs
    # start from 1, since 0 is reserved for main
    _process_counter = itertools.count(1)
    _colors_generator = color_generator()

    def __init__(self, func : Callable[..., Any], *args : tuple[Any, ...], **kwargs : dict[str, Any]):
        self.queue = multiprocessing.Queue()
        if not Settings.MULTIPROCESSING:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            try:
                signal.alarm(Settings.CORE_TIMEOUT)
                result = func(*args, **kwargs)
                self.queue.put(result)
            except (TimeoutError, KeyboardInterrupt) as e:
                self.queue.put(e)
            finally:
                signal.alarm(0)
        else:
            pid = next(self._process_counter)
            self.process = multiprocessing.Process(target = self._wrapper, args = (func, next(self._colors_generator), args, kwargs), name = f"{pid}")
            wait_and_retry(lambda : self._start(), lambda : len(multiprocessing.active_children()) < Settings.PROCESSES_COUNT, Settings.MULTIPROCESSING_SPINNING_INTERVAL)
    
    """
    Start the parallel process.
    """
    def _start(self) -> None:
        self.process.start()
        self._start = time.perf_counter()
    
    """
    Wraps and runs the function passed to Worker inside another process.
    """
    def _wrapper(self, func : Callable[..., Any], color : Optional[tuple[int, int, int]], args : tuple[Any, ...], kwargs : dict[str, Any]) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if color:
                Settings.VERBOSE_COLOR = color
            result = func(*args, **kwargs)
            self.queue.put(result)
        except Exception as e:
            self.queue.put(e)

    def _timeout_handler(signum : int, _) -> None:
        stack = traceback.extract_stack()
        function_name = stack[-2].name
        raise TimeoutError(f"Function (running '{function_name}') exceeded the timeout of {Settings.CORE_TIMEOUT} seconds and was terminated!")

    """
    Gets the latest result from the process.
    Raises an exception if the process's execution time limit is exceeded.
    This method is blocking.
    """
    def get(self) -> Any:
        if not Settings.MULTIPROCESSING:
            return self.queue.get()

        # NOTE: starting up processes takes a ton of time, so one may be able to finish in more than
        #       'Settings.CORE_TIMEOUT' simply because the CPU was busy and could not kill it...
        while self.process.is_alive():
            elapsed = time.perf_counter() - self._start
            # NOTE: using 'is_alive' here prevents a call to 'get' that happened after 'CORE_TIMEOUT' time to raise an issue even if the process has finished.
            #       This may lead to some processes using a bit more than 'CORE_TIMEOUT' time because there is not easy way to exactly track when they finish.
            if Settings.CORE_TIMEOUT and self.process.is_alive() and elapsed >= Settings.CORE_TIMEOUT:
                self.process.terminate()
                self.process.join()
                raise TimeoutError(f"Process {self.process.name} exceeded the timeout of {Settings.CORE_TIMEOUT} seconds and was terminated.")
            self.process.join(timeout = min(Settings.MULTIPROCESSING_SPINNING_INTERVAL, Settings.CORE_TIMEOUT - elapsed) if Settings.CORE_TIMEOUT else Settings.MULTIPROCESSING_SPINNING_INTERVAL)
        
        result = self.queue.get()
        if isinstance(result, Exception):
            raise result
        return result

"""
Wraps each instruction in an exception handler (try-except) that swallows the 'allowed_exceptions'.
"""
# NOTE: this is TERRIBLE code design, as it runs code line by line...
class ExceptionSwallowTransformer(ast.NodeTransformer):
    def __init__(self, allowed_exceptions : tuple[type, ...]):
        self.allowed_exceptions = allowed_exceptions

    def visit_FunctionDef(self, node : ast.FunctionDef) -> ast.FunctionDef:
        new_body = []
        for stmt in node.body:
            try_stmt = ast.Try(
                body = [stmt],
                handlers = [
                    ast.ExceptHandler(
                        type=ast.Tuple(elts = [ast.Name(exc.__name__, ctx = ast.Load()) for exc in self.allowed_exceptions], ctx = ast.Load()),
                        name='e',
                        body=[
                            ast.Expr(ast.Call(
                                func = ast.Name(id='print', ctx = ast.Load()),
                                args = [ast.Constant(f"Swallowed: {ast.unparse(stmt).strip()}"), ast.Name(id = 'e', ctx = ast.Load())],
                                keywords = []
                            ))
                        ]
                    )
                ],
                orelse = [],
                finalbody = []
            )
            new_body.append(try_stmt)
        node.body = new_body
        return node

"""
Runs the given function line by line.
Any line that results in an exception is skipped.
"""
def make_swallowing_wrapper(func : Callable[..., Any], allowed_exceptions : tuple[type, ...] = (AttributeError, TypeError, UnboundLocalError, TimeoutError)) -> dict[str, Any]:
    src = inspect.getsource(func)
    src = textwrap.dedent(src)

    mod_ast = ast.parse(src)
    mod_ast = ExceptionSwallowTransformer(allowed_exceptions).visit(mod_ast)
    ast.fix_missing_locations(mod_ast)

    # preserve the closure context
    code = compile(mod_ast, filename = "<ast>", mode = "exec")
    func_globals = func.__globals__.copy()
    exec(code, func_globals)
    return func_globals[func.__name__]

"""
Terminates all child processes.
<<<86's S1 ending plays in the background>>>
"""
def kill_all_children():
    for p in multiprocessing.active_children():
        print(f"Terminating child process {p.name}.")
        try:
            p.kill()
        except ProcessLookupError:
            print(f"Process {p.name} already terminated.")


# UTILITY FUNCTIONS:

"""
Compute the manhattan distance betwenn two points 'pt1' and 'pt2' in an n-dimensional lattice.
"""
def manhattan(pt1 : tuple[int, ...], pt2 : tuple[int, ...]) -> int:
    if len(pt1) != len(pt2):
        raise Exception("The two points must have the same dimensionality.")
    
    return sum(abs(coord1 - coord2) for coord1, coord2 in zip(pt1, pt2))

"""
Iterate over all coordinates in a rectangular sub-lattice diagonally.
Starts from (x_beg, y_beg) [included] and proceeds over minor diagonals
towards (x_end, y_end) [included/excluded depending on 'end_included'].

Note: this coincides with an enumeration by increasing manhattan distance from (x_beg, y_beg).
"""
def iter_major_diagonals(x_beg : int, y_beg : int, x_end : int, y_end : int, end_included : bool = False) -> Generator[Coord2D, None, None]:
    x_sign = 1 if x_beg < x_end else -1
    y_sign = 1 if y_beg < y_end else -1
    if end_included:
        x_end += x_sign
        y_end += y_sign
    width = abs(x_end - x_beg)
    height = abs(y_beg - y_end)
    for i in range(height):
        for j in range(min(i + 1, width)):
            yield Coord2D(x_beg + x_sign*j, y_beg + y_sign*(i - j))
    for j in range(1, width):
        for i in range(min(width - j, height)):
            yield Coord2D(x_beg + x_sign*(j + i), y_end - y_sign*(1 + i))

"""
Performs a foldL of the provided 'iterable', starting from 'initial', and return
every intermediate result, in order. Intermediate results include 'initial'.

The provided folding function shall take two arguments, of which the first will
be used for the accumulator.
"""
def scan_left(func : Callable[[T, U], T], iterable : Iterable[U], initial : T) -> list[T]:
    result = [initial]
    acc = initial
    for item in iterable:
        acc = func(acc, item)
        result.append(acc)
    return result

"""
Given multiple dictionaries potentially sharing the same key and having values
for which the '+' operation is defined, this returns the dictionary having all
keys of the originals and for values the sum of the values with the same key
among the original dictionaries.
"""
def dict_sum(*dicts : tuple[dict[T, U], ...]) -> dict[T, U]:
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

"""
Given N buckets all of size M and a set 'num' of positive integers, this algorithm
checks if there exists a way to distribute the elements of the set among buckets
such that the sum of the integers in each bucket is <=M.
"""
def can_distribute(nums : list[int], N : int, M : int) -> bool:
    if sum(nums) > N * M:
        return False  # Quick fail: total sum exceeds total capacity

    nums.sort(reverse=True)  # Start with biggest numbers to prune faster
    buckets = [0] * N

    def backtrack(index):
        if index == len(nums):
            return True
        num = nums[index]
        for i in range(N):
            if buckets[i] + num <= M:
                buckets[i] += num
                if backtrack(index + 1):
                    return True
                buckets[i] -= num
            # Optimization: If a number doesn't fit in an empty bucket, don't try placing it in other empty buckets (to avoid symmetry)
            if buckets[i] == 0:
                break
        return False

    return backtrack(0)

"""
Given N buckets all of size M and a list of sets of integers 'sets_list', this algorithm
checks if there exists a way to distribute the sets of the list among buckets such that
the count [or sum] of integers in each bucket is <=M and the number of sets added to a
bucket does not exceed K. However, each bucket is itself a set, as such, duplicate
elements do not count towards M.

NOTE: stupidly slow.

Args:
- sets_list: list of sets to distribute among buckets
- N: number of buckets
- if sum_not_count == False:
    - M: maximum number of distinct elements per bucket
  else:
    - M: maximum sum of distinct elements per bucket
- K: maximum number of sets per bucket
"""
def can_distribute_sets(sets_list : list[set[int]], N : int, M : int, K : int, sum_not_count : bool = False) -> bool:
    # Convert each set into a bitmask
    def set_to_bitmask(s):
        mask = 0
        for x in s:
            mask |= 1 << x
        return mask

    def bitcount(x):
        return bin(x).count('1')
    
    def bitvalue(x):
        value = 0
        for i, b in enumerate(bin(x)[:1:-1]):
            if b == 1:
                value += i
        return value

    bitsets = sorted([set_to_bitmask(s) for s in sets_list], key=bitcount, reverse=True)

    # Quick fail: total unique elements must fit
    total_union = 0
    for b in bitsets:
        total_union |= b
    if bitcount(total_union) > N * M or len(bitsets) > N * K:
        return False

    # Each bucket: (bitmask of elements, count of sets)
    initial_state = [(0, 0) for _ in range(N)]
    stack = [(0, initial_state)]

    while stack:
        index, buckets = stack.pop()
        if index == len(bitsets):
            return True  # All sets placed

        current = bitsets[index]

        for i in range(N):
            bits, count = buckets[i]
            if count >= K:
                continue
            merged = bits | current
            if (not sum_not_count and bitcount(merged) <= M) or (sum_not_count and bitvalue(merged) <= M):
                new_buckets = list(buckets)
                new_buckets[i] = (merged, count + 1)
                stack.append((index + 1, new_buckets))
            if count == 0:
                break  # symmetry breaking

    return False

"""
Generate 'num_points' approximately equi-spaced points on a 2D lattice based on Manhattan distance.

Parameters:
- width: the width of the lattice (number of columns). Must be > 0.
- height: the height of the lattice (number of rows). Must be > 0.
- num_points: the number of equi-spaced points to return. Must be ≤ width * height.

Returns: a list of Coord2D instances representing approximately equi-spaced points.
"""
def get_equispaced_lattice_points(width: int, height: int, num_points: int) -> list[Coord2D]:
    if width <= 0 or height <= 0:
        raise Exception("Width and height must be positive integers.")
    
    total_points = width * height
    if num_points > total_points:
        raise Exception(f"Cannot select {num_points} points from a lattice of size {width}x{height} ({total_points} points).")

    # flatten all possible lattice points in row-major order
    all_points = [Coord2D(x, y) for y in range(height) for x in range(width)]
    
    if num_points == 1:
        # choose the center point
        center_x = width // 2
        center_y = height // 2
        return [Coord2D(center_x, center_y)]

    # choose the points approximately equidistant in the list in terms of manhattan spread
    # distribute indices as evenly as possible
    step = total_points / num_points
    selected_points = []
    for i in range(num_points):
        index = int(round(i * step))
        if index >= total_points:
            index = total_points - 1
        selected_points.append(all_points[index])

    # deduplicate in rare edge rounding cases
    return list(dict.fromkeys(selected_points))

"""
Converts a string from camel to snake case notation.
"""
def camel_to_snake(s : str) -> str:
    # Insert _ before capital letters that are followed by a lowercase letter, and are not at the start
    s = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', s)
    return s.lower()

"""
Modifies an array of integers to contain only contigous numbers form 0 upward,
this is done by replacing higher values with missing lower ones.
"""
def force_array_of_contigous_integers(arr : list[int]) -> None:
    unique_vals = sorted(set(arr))
    value_map = {val: idx for idx, val in enumerate(unique_vals)}
    for i in range(len(arr)):
        arr[i] = value_map[arr[i]]