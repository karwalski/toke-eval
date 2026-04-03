#!/usr/bin/env python3
"""Generate 500 new hidden test tasks (task-a-0501 through task-a-1000).

All tasks are scoped to I/O type signatures that the current tkc codegen
handles correctly.  No string input/output tasks are included.

Categories and signatures are chosen for diversity across:
  - arithmetic, array_list, mathematical_sequences, comparison_logic,
    search_selection, encoding, bitwise
  - i64->i64, [i64]->i64, [i64]->[i64], [i64]->bool, i64->bool,
    [i64,i64]->i64, [[i64],i64]->i64, [[i64],i64]->[i64],
    [[i64],i64]->bool, [[i64],[i64]]->[i64], i64->[i64],
    [i64,i64]->[i64], [i64,i64]->bool

Each task has 15-50 test cases with edge cases.
"""

import json
import os
import random
import sys
import yaml

random.seed(42)

TASKS_DIR = "hidden_tests"
START_ID = 501
COUNT = 500


# ---------------------------------------------------------------------------
# Task definitions — each is a (description, input_type, output_type,
#                                category, generate_fn) tuple.
# generate_fn(rng) -> list of {"input": ..., "expected": ...}
# ---------------------------------------------------------------------------

def _rand_list(rng, lo=-100, hi=100, min_len=1, max_len=20):
    n = rng.randint(min_len, max_len)
    return [rng.randint(lo, hi) for _ in range(n)]

def _rand_list_nonempty(rng, lo=-100, hi=100, max_len=15):
    return _rand_list(rng, lo, hi, min_len=1, max_len=max_len)


# ---------- i64 -> i64 tasks ----------

TASK_TEMPLATES = []

def _add(desc, in_ty, out_ty, cat, gen_fn):
    TASK_TEMPLATES.append((desc, in_ty, out_ty, cat, gen_fn))

# 1. Absolute value
def gen_abs(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(-1000, 1000)
        cases.append({"input": n, "expected": abs(n)})
    cases.append({"input": 0, "expected": 0})
    return cases
_add("Return the absolute value of the input integer", "i64", "i64", "arithmetic", gen_abs)

# 2. Square a number
def gen_square(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(-100, 100)
        cases.append({"input": n, "expected": n * n})
    cases.append({"input": 0, "expected": 0})
    return cases
_add("Return the square of the input integer", "i64", "i64", "arithmetic", gen_square)

# 3. Cube a number
def gen_cube(rng):
    cases = []
    for _ in range(25):
        n = rng.randint(-50, 50)
        cases.append({"input": n, "expected": n ** 3})
    return cases
_add("Return the cube of the input integer", "i64", "i64", "arithmetic", gen_cube)

# 4. Factorial (small)
def gen_factorial(rng):
    cases = []
    def fact(n):
        r = 1
        for i in range(2, n+1): r *= i
        return r
    for n in range(0, 13):
        cases.append({"input": n, "expected": fact(n)})
    for _ in range(10):
        n = rng.randint(0, 12)
        cases.append({"input": n, "expected": fact(n)})
    return cases
_add("Return the factorial of the input integer (0 <= n <= 12)", "i64", "i64", "arithmetic", gen_factorial)

# 5. Fibonacci
def gen_fib(rng):
    fibs = [0, 1]
    for i in range(2, 25):
        fibs.append(fibs[-1] + fibs[-2])
    cases = []
    for n in range(20):
        cases.append({"input": n, "expected": fibs[n]})
    for _ in range(10):
        n = rng.randint(0, 24)
        cases.append({"input": n, "expected": fibs[n]})
    return cases
_add("Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)", "i64", "i64", "mathematical_sequences", gen_fib)

# 6. Count digits
def gen_count_digits(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(0, 999999999)
        cases.append({"input": n, "expected": len(str(n))})
    cases.append({"input": 0, "expected": 1})
    return cases
_add("Count the number of decimal digits in a non-negative integer", "i64", "i64", "arithmetic", gen_count_digits)

# 7. Sum of digits
def gen_digit_sum(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(0, 999999)
        cases.append({"input": n, "expected": sum(int(d) for d in str(n))})
    cases.append({"input": 0, "expected": 0})
    return cases
_add("Return the sum of digits of a non-negative integer", "i64", "i64", "arithmetic", gen_digit_sum)

# 8. Power of 2 check
def gen_is_pow2(rng):
    cases = []
    for p in range(0, 20):
        cases.append({"input": 1 << p, "expected": True})
    for _ in range(15):
        n = rng.randint(3, 10000)
        if n & (n-1) == 0:
            n += 1
        cases.append({"input": n, "expected": False})
    cases.append({"input": 0, "expected": False})
    return cases
_add("Return true if the input is a power of 2, false otherwise", "i64", "bool", "comparison_logic", gen_is_pow2)

# 9. Is even
def gen_is_even(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(-1000, 1000)
        cases.append({"input": n, "expected": n % 2 == 0})
    return cases
_add("Return true if the input integer is even", "i64", "bool", "comparison_logic", gen_is_even)

# 10. Collatz steps
def gen_collatz(rng):
    def collatz(n):
        steps = 0
        while n != 1:
            if n % 2 == 0: n //= 2
            else: n = 3 * n + 1
            steps += 1
        return steps
    cases = []
    for n in range(1, 20):
        cases.append({"input": n, "expected": collatz(n)})
    for _ in range(10):
        n = rng.randint(1, 100)
        cases.append({"input": n, "expected": collatz(n)})
    return cases
_add("Return the number of Collatz steps to reach 1 from n", "i64", "i64", "mathematical_sequences", gen_collatz)

# 11. GCD
def gen_gcd(rng):
    from math import gcd
    cases = []
    for _ in range(30):
        a = rng.randint(1, 1000)
        b = rng.randint(1, 1000)
        cases.append({"input": [a, b], "expected": gcd(a, b)})
    cases.append({"input": [0, 5], "expected": 5})
    return cases
_add("Return the greatest common divisor of two positive integers", "[i64, i64]", "i64", "arithmetic", gen_gcd)

# 12. LCM
def gen_lcm(rng):
    from math import gcd
    def lcm(a, b): return abs(a * b) // gcd(a, b) if a and b else 0
    cases = []
    for _ in range(30):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        cases.append({"input": [a, b], "expected": lcm(a, b)})
    return cases
_add("Return the least common multiple of two positive integers", "[i64, i64]", "i64", "arithmetic", gen_lcm)

# ---------- [i64] -> i64 tasks ----------

# 13. Sum of list
def gen_sum(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(arr)})
    return cases
_add("Return the sum of all integers in the list", "[i64]", "i64", "arithmetic", gen_sum)

# 14. Product of list
def gen_product(rng):
    cases = [{"input": [], "expected": 1}]
    for _ in range(25):
        arr = _rand_list(rng, lo=-10, hi=10, min_len=1, max_len=8)
        p = 1
        for x in arr: p *= x
        cases.append({"input": arr, "expected": p})
    return cases
_add("Return the product of all integers in the list", "[i64]", "i64", "arithmetic", gen_product)

# 15. Maximum
def gen_max(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng)
        cases.append({"input": arr, "expected": max(arr)})
    return cases
_add("Return the maximum value in the list", "[i64]", "i64", "search_selection", gen_max)

# 16. Minimum
def gen_min(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng)
        cases.append({"input": arr, "expected": min(arr)})
    return cases
_add("Return the minimum value in the list", "[i64]", "i64", "search_selection", gen_min)

# 17. Length
def gen_length(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": len(arr)})
    return cases
_add("Return the number of elements in the list", "[i64]", "i64", "array_list", gen_length)

# 18. Count positive
def gen_count_pos(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(1 for x in arr if x > 0)})
    return cases
_add("Count how many positive numbers are in the list", "[i64]", "i64", "comparison_logic", gen_count_pos)

# 19. Count negative
def gen_count_neg(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(1 for x in arr if x < 0)})
    return cases
_add("Count how many negative numbers are in the list", "[i64]", "i64", "comparison_logic", gen_count_neg)

# 20. Count zeros
def gen_count_zero(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-5, hi=5)
        cases.append({"input": arr, "expected": sum(1 for x in arr if x == 0)})
    return cases
_add("Count how many zeros are in the list", "[i64]", "i64", "comparison_logic", gen_count_zero)

# ---------- [i64] -> [i64] tasks ----------

# 21. Reverse list
def gen_reverse(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": arr[::-1]})
    return cases
_add("Reverse the list", "[i64]", "[i64]", "array_list", gen_reverse)

# 22. Double each element
def gen_double(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": [x * 2 for x in arr]})
    return cases
_add("Double every element in the list", "[i64]", "[i64]", "array_list", gen_double)

# 23. Negate each element
def gen_negate(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": [-x for x in arr]})
    return cases
_add("Negate every element in the list", "[i64]", "[i64]", "array_list", gen_negate)

# 24. Filter positives
def gen_filter_pos(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": [x for x in arr if x > 0]})
    return cases
_add("Return a new list containing only the positive elements", "[i64]", "[i64]", "array_list", gen_filter_pos)

# 25. Filter even
def gen_filter_even(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": [x for x in arr if x % 2 == 0]})
    return cases
_add("Return a new list containing only the even elements", "[i64]", "[i64]", "array_list", gen_filter_even)

# 26. Prefix sums
def gen_prefix_sum(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        ps = []
        s = 0
        for x in arr:
            s += x
            ps.append(s)
        cases.append({"input": arr, "expected": ps})
    return cases
_add("Return the running cumulative sum (prefix sums) of the list", "[i64]", "[i64]", "array_list", gen_prefix_sum)

# 27. Sort ascending
def gen_sort(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, max_len=15)
        cases.append({"input": arr, "expected": sorted(arr)})
    return cases
_add("Sort the list in ascending order", "[i64]", "[i64]", "array_list", gen_sort)

# ---------- [i64] -> bool tasks ----------

# 28. Is sorted ascending
def gen_is_sorted(rng):
    cases = [{"input": [], "expected": True}]
    for _ in range(15):
        arr = sorted(_rand_list(rng, min_len=1, max_len=10))
        cases.append({"input": arr, "expected": True})
    for _ in range(15):
        arr = _rand_list(rng, min_len=2, max_len=10)
        cases.append({"input": arr, "expected": arr == sorted(arr)})
    return cases
_add("Return true if the list is sorted in non-decreasing order", "[i64]", "bool", "comparison_logic", gen_is_sorted)

# 29. Contains zero
def gen_has_zero(rng):
    cases = [{"input": [], "expected": False}]
    for _ in range(15):
        arr = _rand_list(rng, min_len=1, lo=-10, hi=10)
        cases.append({"input": arr, "expected": 0 in arr})
    for _ in range(10):
        arr = _rand_list(rng, min_len=1, lo=1, hi=100)
        cases.append({"input": arr, "expected": False})
    return cases
_add("Return true if the list contains at least one zero", "[i64]", "bool", "search_selection", gen_has_zero)

# 30. All positive
def gen_all_pos(rng):
    cases = [{"input": [], "expected": True}]
    for _ in range(15):
        arr = _rand_list(rng, min_len=1, lo=1, hi=100)
        cases.append({"input": arr, "expected": True})
    for _ in range(15):
        arr = _rand_list(rng, min_len=1, lo=-50, hi=50)
        cases.append({"input": arr, "expected": all(x > 0 for x in arr)})
    return cases
_add("Return true if all elements in the list are positive", "[i64]", "bool", "comparison_logic", gen_all_pos)

# ---------- [[i64], i64] -> i64 tasks ----------

# 31. Count occurrences of a value
def gen_count_val(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-10, hi=10)
        v = rng.randint(-10, 10)
        cases.append({"input": [arr, v], "expected": arr.count(v)})
    return cases
_add("Count how many times the second argument appears in the array", "[[i64], i64]", "i64", "search_selection", gen_count_val)

# 32. Element at index
def gen_elem_at(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng, max_len=15)
        idx = rng.randint(0, len(arr) - 1)
        cases.append({"input": [arr, idx], "expected": arr[idx]})
    return cases
_add("Return the element at the given index in the array", "[[i64], i64]", "i64", "array_list", gen_elem_at)

# ---------- [[i64], i64] -> [i64] tasks ----------

# 33. Add scalar to each element
def gen_add_scalar(rng):
    cases = [{"input": [[], 5], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        k = rng.randint(-50, 50)
        cases.append({"input": [arr, k], "expected": [x + k for x in arr]})
    return cases
_add("Add the second argument to every element of the array", "[[i64], i64]", "[i64]", "array_list", gen_add_scalar)

# 34. Multiply each by scalar
def gen_mul_scalar(rng):
    cases = [{"input": [[], 3], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-20, hi=20)
        k = rng.randint(-10, 10)
        cases.append({"input": [arr, k], "expected": [x * k for x in arr]})
    return cases
_add("Multiply every element of the array by the second argument", "[[i64], i64]", "[i64]", "array_list", gen_mul_scalar)

# 35. Filter greater than
def gen_filter_gt(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        threshold = rng.randint(-50, 50)
        cases.append({"input": [arr, threshold], "expected": [x for x in arr if x > threshold]})
    return cases
_add("Return elements greater than the given threshold", "[[i64], i64]", "[i64]", "search_selection", gen_filter_gt)

# 36. Take first N elements
def gen_take_n(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, max_len=15)
        n = rng.randint(0, len(arr) + 2)
        cases.append({"input": [arr, n], "expected": arr[:n]})
    return cases
_add("Return the first N elements of the array (or all if N > length)", "[[i64], i64]", "[i64]", "array_list", gen_take_n)

# ---------- [[i64], i64] -> bool tasks ----------

# 37. Contains value
def gen_contains(rng):
    cases = [{"input": [[], 5], "expected": False}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-20, hi=20)
        v = rng.randint(-20, 20)
        cases.append({"input": [arr, v], "expected": v in arr})
    return cases
_add("Return true if the array contains the given value", "[[i64], i64]", "bool", "search_selection", gen_contains)

# 38. All greater than
def gen_all_gt(rng):
    cases = [{"input": [[], 0], "expected": True}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=1)
        threshold = rng.randint(-50, 50)
        cases.append({"input": [arr, threshold], "expected": all(x > threshold for x in arr)})
    return cases
_add("Return true if all elements are greater than the threshold", "[[i64], i64]", "bool", "comparison_logic", gen_all_gt)

# ---------- i64 -> [i64] tasks ----------

# 39. Range 0 to N
def gen_range(rng):
    cases = [{"input": 0, "expected": []}]
    for _ in range(25):
        n = rng.randint(0, 20)
        cases.append({"input": n, "expected": list(range(n))})
    return cases
_add("Return a list of integers from 0 to n-1", "i64", "[i64]", "array_list", gen_range)

# 40. Divisors
def gen_divisors(rng):
    cases = []
    for _ in range(25):
        n = rng.randint(1, 100)
        divs = [i for i in range(1, n+1) if n % i == 0]
        cases.append({"input": n, "expected": divs})
    return cases
_add("Return all divisors of n in ascending order", "i64", "[i64]", "mathematical_sequences", gen_divisors)

# ---------- [[i64], [i64]] -> [i64] tasks ----------

# 41. Elementwise add
def gen_elemwise_add(rng):
    cases = [{"input": [[], []], "expected": []}]
    for _ in range(25):
        n = rng.randint(1, 15)
        a = [rng.randint(-100, 100) for _ in range(n)]
        b = [rng.randint(-100, 100) for _ in range(n)]
        cases.append({"input": [a, b], "expected": [x + y for x, y in zip(a, b)]})
    return cases
_add("Return elementwise sum of two lists of equal length", "[[i64], [i64]]", "[i64]", "array_list", gen_elemwise_add)

# 42. Merge sorted
def gen_merge_sorted(rng):
    cases = [{"input": [[], []], "expected": []}]
    for _ in range(25):
        a = sorted([rng.randint(-50, 50) for _ in range(rng.randint(0, 8))])
        b = sorted([rng.randint(-50, 50) for _ in range(rng.randint(0, 8))])
        cases.append({"input": [a, b], "expected": sorted(a + b)})
    return cases
_add("Merge two sorted lists into one sorted list", "[[i64], [i64]]", "[i64]", "array_list", gen_merge_sorted)

# ---------- Additional arithmetic i64->i64 ----------

# 43. Triangular number
def gen_triangular(rng):
    cases = []
    for n in range(0, 20):
        cases.append({"input": n, "expected": n * (n + 1) // 2})
    for _ in range(15):
        n = rng.randint(0, 100)
        cases.append({"input": n, "expected": n * (n + 1) // 2})
    return cases
_add("Return the nth triangular number (n*(n+1)/2)", "i64", "i64", "mathematical_sequences", gen_triangular)

# 44. Is prime
def gen_is_prime(rng):
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        i = 3
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True
    cases = []
    for n in range(0, 30):
        cases.append({"input": n, "expected": is_prime(n)})
    for _ in range(10):
        n = rng.randint(30, 200)
        cases.append({"input": n, "expected": is_prime(n)})
    return cases
_add("Return true if the input is a prime number", "i64", "bool", "mathematical_sequences", gen_is_prime)

# 45. Sum of first N natural numbers
def gen_sum_n(rng):
    cases = []
    for _ in range(30):
        n = rng.randint(0, 200)
        cases.append({"input": n, "expected": n * (n + 1) // 2})
    return cases
_add("Return the sum of integers from 1 to n", "i64", "i64", "arithmetic", gen_sum_n)

# 46. Modular arithmetic
def gen_mod(rng):
    cases = []
    for _ in range(30):
        a = rng.randint(0, 1000)
        b = rng.randint(1, 100)
        cases.append({"input": [a, b], "expected": a % b})
    return cases
_add("Return a modulo b (a mod b) where b > 0", "[i64, i64]", "i64", "arithmetic", gen_mod)

# 47. Power
def gen_power(rng):
    cases = []
    for _ in range(30):
        base = rng.randint(-5, 5)
        exp = rng.randint(0, 10)
        cases.append({"input": [base, exp], "expected": base ** exp})
    return cases
_add("Return base raised to the power exp (base^exp)", "[i64, i64]", "i64", "arithmetic", gen_power)

# 48. Second largest
def gen_second_largest(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng, max_len=15)
        if len(arr) < 2:
            arr.append(rng.randint(-100, 100))
        s = sorted(set(arr))
        if len(s) < 2:
            expected = s[0]
        else:
            expected = s[-2]
        cases.append({"input": arr, "expected": expected})
    return cases
_add("Return the second largest distinct value in the list", "[i64]", "i64", "search_selection", gen_second_largest)

# 49. Index of maximum
def gen_index_max(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng, max_len=15)
        cases.append({"input": arr, "expected": arr.index(max(arr))})
    return cases
_add("Return the index of the maximum element (first occurrence)", "[i64]", "i64", "search_selection", gen_index_max)

# 50. Dot product
def gen_dot_product(rng):
    cases = [{"input": [[], []], "expected": 0}]
    for _ in range(25):
        n = rng.randint(1, 10)
        a = [rng.randint(-20, 20) for _ in range(n)]
        b = [rng.randint(-20, 20) for _ in range(n)]
        cases.append({"input": [a, b], "expected": sum(x * y for x, y in zip(a, b))})
    return cases
_add("Return the dot product of two equal-length integer lists", "[[i64], [i64]]", "i64", "arithmetic", gen_dot_product)

# Now generate variations of these base tasks to fill 500
# We'll create parameter variations: different constant offsets, scaling, etc.

def gen_add_constant(k):
    """Return input + k"""
    def gen(rng):
        cases = []
        for _ in range(30):
            n = rng.randint(-500, 500)
            cases.append({"input": n, "expected": n + k})
        return cases
    return gen

def gen_multiply_constant(k):
    """Return input * k"""
    def gen(rng):
        cases = []
        for _ in range(30):
            n = rng.randint(-100, 100)
            cases.append({"input": n, "expected": n * k})
        return cases
    return gen

def gen_sum_plus_k(k):
    """Return sum of list + k"""
    def gen(rng):
        cases = [{"input": [], "expected": k}]
        for _ in range(25):
            arr = _rand_list(rng, min_len=0)
            cases.append({"input": arr, "expected": sum(arr) + k})
        return cases
    return gen

def gen_max_minus_min(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list_nonempty(rng)
        cases.append({"input": arr, "expected": max(arr) - min(arr)})
    return cases

def gen_count_even(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(1 for x in arr if x % 2 == 0)})
    return cases

def gen_count_odd(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(1 for x in arr if x % 2 != 0)})
    return cases

def gen_sum_even(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(x for x in arr if x % 2 == 0)})
    return cases

def gen_sum_odd(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": sum(x for x in arr if x % 2 != 0)})
    return cases

def gen_squares_list(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-20, hi=20)
        cases.append({"input": arr, "expected": [x*x for x in arr]})
    return cases

def gen_abs_list(rng):
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0)
        cases.append({"input": arr, "expected": [abs(x) for x in arr]})
    return cases

def gen_clamp(rng):
    cases = []
    for _ in range(30):
        arr = _rand_list(rng, min_len=1)
        lo = rng.randint(-50, 0)
        hi = rng.randint(1, 50)
        clamped = [max(lo, min(hi, x)) for x in arr]
        cases.append({"input": [arr, lo, hi], "expected": clamped})
    return cases

def gen_is_palindrome(rng):
    cases = [{"input": [], "expected": True}]
    for _ in range(15):
        half = [rng.randint(-20, 20) for _ in range(rng.randint(1, 5))]
        pal = half + half[::-1]
        cases.append({"input": pal, "expected": True})
    for _ in range(15):
        arr = _rand_list(rng, min_len=2, max_len=10)
        cases.append({"input": arr, "expected": arr == arr[::-1]})
    return cases

def gen_rotate_left(rng):
    cases = [{"input": [[], 3], "expected": []}]
    for _ in range(30):
        arr = _rand_list_nonempty(rng, max_len=15)
        k = rng.randint(0, len(arr) * 2)
        n = len(arr)
        rot = k % n if n > 0 else 0
        result = arr[rot:] + arr[:rot]
        cases.append({"input": [arr, k], "expected": result})
    return cases

def gen_unique(rng):
    """Remove duplicates preserving order"""
    cases = [{"input": [], "expected": []}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-10, hi=10)
        seen = set()
        result = []
        for x in arr:
            if x not in seen:
                seen.add(x)
                result.append(x)
        cases.append({"input": arr, "expected": result})
    return cases

def gen_intersection(rng):
    cases = [{"input": [[], []], "expected": []}]
    for _ in range(25):
        a = _rand_list(rng, min_len=0, lo=-10, hi=10, max_len=10)
        b = _rand_list(rng, min_len=0, lo=-10, hi=10, max_len=10)
        bset = set(b)
        result = sorted(set(x for x in a if x in bset))
        cases.append({"input": [a, b], "expected": result})
    return cases

def gen_difference(rng):
    cases = [{"input": [[], []], "expected": []}]
    for _ in range(25):
        a = _rand_list(rng, min_len=0, lo=-10, hi=10, max_len=10)
        b = _rand_list(rng, min_len=0, lo=-10, hi=10, max_len=10)
        bset = set(b)
        result = sorted(set(x for x in a if x not in bset))
        cases.append({"input": [a, b], "expected": result})
    return cases

def gen_has_duplicates(rng):
    cases = [{"input": [], "expected": False}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=1, lo=-10, hi=10)
        cases.append({"input": arr, "expected": len(arr) != len(set(arr))})
    return cases

def gen_sum_squares(rng):
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list(rng, min_len=0, lo=-30, hi=30)
        cases.append({"input": arr, "expected": sum(x*x for x in arr)})
    return cases

def gen_mean_floor(rng):
    """Return floor of arithmetic mean, or 0 for empty"""
    cases = [{"input": [], "expected": 0}]
    for _ in range(30):
        arr = _rand_list_nonempty(rng)
        cases.append({"input": arr, "expected": sum(arr) // len(arr)})
    return cases

# Additional extras
_add("Return the range (max - min) of the list", "[i64]", "i64", "arithmetic", gen_max_minus_min)
_add("Count even numbers in the list", "[i64]", "i64", "comparison_logic", gen_count_even)
_add("Count odd numbers in the list", "[i64]", "i64", "comparison_logic", gen_count_odd)
_add("Sum only the even numbers in the list", "[i64]", "i64", "arithmetic", gen_sum_even)
_add("Sum only the odd numbers in the list", "[i64]", "i64", "arithmetic", gen_sum_odd)
_add("Square each element in the list", "[i64]", "[i64]", "array_list", gen_squares_list)
_add("Return absolute values of all elements", "[i64]", "[i64]", "array_list", gen_abs_list)
_add("Return true if the list is a palindrome", "[i64]", "bool", "comparison_logic", gen_is_palindrome)
_add("Rotate the array left by k positions", "[[i64], i64]", "[i64]", "array_list", gen_rotate_left)
_add("Remove duplicates preserving first occurrence order", "[i64]", "[i64]", "array_list", gen_unique)
_add("Return sorted intersection of two lists (unique elements)", "[[i64], [i64]]", "[i64]", "search_selection", gen_intersection)
_add("Return sorted difference of first list minus second (unique)", "[[i64], [i64]]", "[i64]", "search_selection", gen_difference)
_add("Return true if the list has any duplicate values", "[i64]", "bool", "comparison_logic", gen_has_duplicates)
_add("Return the sum of squares of all elements", "[i64]", "i64", "arithmetic", gen_sum_squares)
_add("Return the floor of the arithmetic mean (or 0 for empty list)", "[i64]", "i64", "arithmetic", gen_mean_floor)
_add("Clamp each element between lo and hi", "[[i64], i64, i64]", "[i64]", "array_list", gen_clamp)

# Fill to 500 with arithmetic variations
for k in range(1, 150):
    _add(f"Return the input integer plus {k}", "i64", "i64", "arithmetic", gen_add_constant(k))

for k in [2, 3, 4, 5, 7, 10, -1, -2, -3, 11, 13, 17, 19, 23]:
    _add(f"Multiply the input by {k}", "i64", "i64", "arithmetic", gen_multiply_constant(k))

for k in range(1, 80):
    _add(f"Return the sum of the list plus {k}", "[i64]", "i64", "arithmetic", gen_sum_plus_k(k))

# Now generate the YAML files
def main():
    rng = random.Random(2024)

    # Ensure we have exactly 500 templates
    templates = TASK_TEMPLATES[:COUNT]
    if len(templates) < COUNT:
        # Repeat templates cyclically
        while len(templates) < COUNT:
            templates.append(templates[len(templates) % len(TASK_TEMPLATES)])

    for i, (desc, in_ty, out_ty, cat, gen_fn) in enumerate(templates):
        task_id = f"task-a-{START_ID + i:04d}"
        cases = gen_fn(rng)
        task = {
            "id": task_id,
            "phase": "A",
            "category": cat,
            "description": desc,
            "input_type": in_ty,
            "output_type": out_ty,
            "test_inputs": [
                {"input": c["input"], "expected": c["expected"]}
                for c in cases
            ],
        }
        path = os.path.join(TASKS_DIR, f"{task_id}.yaml")
        with open(path, "w") as f:
            yaml.dump(task, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {COUNT} tasks: task-a-{START_ID:04d} through task-a-{START_ID+COUNT-1:04d}")
    print(f"Written to {TASKS_DIR}/")

    # Summary stats
    from collections import Counter
    cats = Counter(t[3] for t in templates)
    sigs = Counter(f"{t[1]} -> {t[2]}" for t in templates)
    print("\nCategory distribution:")
    for k, v in cats.most_common():
        print(f"  {k}: {v}")
    print("\nI/O signature distribution:")
    for k, v in sigs.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
