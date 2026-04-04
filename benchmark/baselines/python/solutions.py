"""Python reference implementations for Phase A benchmark tasks."""

from __future__ import annotations

import math
from functools import reduce
from typing import Any


def _unpack_pair(inp: list) -> tuple:
    """Unpack a two-element input list into (a, b)."""
    return inp[0], inp[1]


def _unpack_triple(inp: list) -> tuple:
    """Unpack a three-element input list into (a, b, c)."""
    return inp[0], inp[1], inp[2]


# ---------------------------------------------------------------------------
# Task registry: task_id -> callable(input) -> expected_output
# ---------------------------------------------------------------------------

SOLUTIONS: dict[str, Any] = {}


def task(task_id: str):
    """Decorator to register a task solution."""
    def decorator(fn):
        SOLUTIONS[task_id] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Arithmetic tasks (0001-0041)
# ---------------------------------------------------------------------------

@task("task-a-0001")
def sum_list(inp: list[int]) -> int:
    return sum(inp)


@task("task-a-0002")
def product_list(inp: list[int]) -> int:
    if not inp:
        return 1
    return reduce(lambda a, b: a * b, inp)


@task("task-a-0003")
def min_list(inp: list[int]) -> int:
    return min(inp)


@task("task-a-0004")
def max_list(inp: list[int]) -> int:
    return max(inp)


@task("task-a-0005")
def mean_floor(inp: list[int]) -> int:
    return sum(inp) // len(inp)


@task("task-a-0006")
def mod_ab(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return a % b


@task("task-a-0007")
def power(inp: list[int]) -> int:
    base, exp = _unpack_pair(inp)
    return base ** exp


@task("task-a-0008")
def gcd(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return math.gcd(a, b)


@task("task-a-0009")
def lcm(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


@task("task-a-0010")
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


@task("task-a-0011")
def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


@task("task-a-0012")
def factorial(n: int) -> int:
    return math.factorial(n)


@task("task-a-0013")
def absolute(n: int) -> int:
    return abs(n)


@task("task-a-0014")
def negate(n: int) -> int:
    return -n


@task("task-a-0015")
def isqrt(n: int) -> int:
    return math.isqrt(n)


@task("task-a-0016")
def clamp(inp: list[int]) -> int:
    v, lo, hi = _unpack_triple(inp)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@task("task-a-0017")
def sum_of_squares(inp: list[int]) -> int:
    return sum(x * x for x in inp)


@task("task-a-0018")
def count_even(inp: list[int]) -> int:
    return sum(1 for x in inp if x % 2 == 0)


@task("task-a-0019")
def count_odd(inp: list[int]) -> int:
    return sum(1 for x in inp if x % 2 != 0)


@task("task-a-0020")
def sum_positive(inp: list[int]) -> int:
    return sum(x for x in inp if x > 0)


@task("task-a-0021")
def count_positive(inp: list[int]) -> int:
    return sum(1 for x in inp if x > 0)


@task("task-a-0022")
def count_negative(inp: list[int]) -> int:
    return sum(1 for x in inp if x < 0)


@task("task-a-0023")
def range_of_list(inp: list[int]) -> int:
    return max(inp) - min(inp)


@task("task-a-0024")
def sum_even(inp: list[int]) -> int:
    return sum(x for x in inp if x % 2 == 0)


@task("task-a-0025")
def multiply(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return a * b


@task("task-a-0026")
def floor_div(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return a // b


@task("task-a-0027")
def square(n: int) -> int:
    return n * n


@task("task-a-0028")
def cube(n: int) -> int:
    return n * n * n


@task("task-a-0029")
def add(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return a + b


@task("task-a-0030")
def subtract(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return a - b


@task("task-a-0031")
def is_even(n: int) -> bool:
    return n % 2 == 0


@task("task-a-0032")
def is_odd(n: int) -> bool:
    return n % 2 != 0


@task("task-a-0033")
def is_divisible(inp: list[int]) -> bool:
    n, k = _unpack_pair(inp)
    return n % k == 0


@task("task-a-0034")
def max_of_two(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return max(a, b)


@task("task-a-0035")
def min_of_two(inp: list[int]) -> int:
    a, b = _unpack_pair(inp)
    return min(a, b)


@task("task-a-0036")
def length(inp: list[int]) -> int:
    return len(inp)


@task("task-a-0037")
def prefix_sum(inp: list[int]) -> list[int]:
    result = []
    s = 0
    for x in inp:
        s += x
        result.append(s)
    return result


@task("task-a-0038")
def elementwise_sum(inp: list) -> list[int]:
    a, b = _unpack_pair(inp)
    return [x + y for x, y in zip(a, b)]


@task("task-a-0039")
def count_zeros(inp: list[int]) -> int:
    return sum(1 for x in inp if x == 0)


@task("task-a-0040")
def max_abs(inp: list[int]) -> int:
    return max(abs(x) for x in inp)


@task("task-a-0041")
def count_divisible_by_3(inp: list[int]) -> int:
    return sum(1 for x in inp if x % 3 == 0)


# ---------------------------------------------------------------------------
# Array / list tasks (0042-0060)
# ---------------------------------------------------------------------------

@task("task-a-0042")
def sort_asc(inp: list[int]) -> list[int]:
    return sorted(inp)


@task("task-a-0043")
def sort_desc(inp: list[int]) -> list[int]:
    return sorted(inp, reverse=True)


@task("task-a-0044")
def reverse_list(inp: list[int]) -> list[int]:
    return list(reversed(inp))


@task("task-a-0045")
def filter_positive(inp: list[int]) -> list[int]:
    return [x for x in inp if x > 0]


@task("task-a-0046")
def filter_negative(inp: list[int]) -> list[int]:
    return [x for x in inp if x < 0]


@task("task-a-0047")
def filter_even(inp: list[int]) -> list[int]:
    return [x for x in inp if x % 2 == 0]


@task("task-a-0048")
def filter_odd(inp: list[int]) -> list[int]:
    return [x for x in inp if x % 2 != 0]


@task("task-a-0049")
def double_elements(inp: list[int]) -> list[int]:
    return [x * 2 for x in inp]


@task("task-a-0050")
def square_elements(inp: list[int]) -> list[int]:
    return [x * x for x in inp]


@task("task-a-0051")
def negate_elements(inp: list[int]) -> list[int]:
    return [-x for x in inp]


@task("task-a-0052")
def add_k(inp: list) -> list[int]:
    lst, k = _unpack_pair(inp)
    return [x + k for x in lst]


@task("task-a-0053")
def deduplicate(inp: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for x in inp:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


@task("task-a-0054")
def rotate_left(inp: list) -> list[int]:
    lst, k = _unpack_pair(inp)
    if not lst:
        return []
    k = k % len(lst)
    return lst[k:] + lst[:k]


@task("task-a-0055")
def rotate_right(inp: list) -> list[int]:
    lst, k = _unpack_pair(inp)
    if not lst:
        return []
    k = k % len(lst)
    return lst[-k:] + lst[:-k] if k else list(lst)


@task("task-a-0056")
def flatten(inp: list) -> list[int]:
    result: list[int] = []
    for sub in inp:
        result.extend(sub)
    return result


@task("task-a-0057")
def chunk(inp: list) -> list[list[int]]:
    lst, k = _unpack_pair(inp)
    if not lst:
        return []
    return [lst[i:i + k] for i in range(0, len(lst), k)]


@task("task-a-0058")
def sliding_window_max(inp: list) -> list[int]:
    lst, k = _unpack_pair(inp)
    if not lst:
        return []
    return [max(lst[i:i + k]) for i in range(len(lst) - k + 1)]


@task("task-a-0059")
def sliding_window_sum(inp: list) -> list[int]:
    lst, k = _unpack_pair(inp)
    if not lst:
        return []
    return [sum(lst[i:i + k]) for i in range(len(lst) - k + 1)]


@task("task-a-0060")
def first_element(inp: list[int]) -> int:
    return inp[0]
