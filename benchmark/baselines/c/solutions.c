/*
 * C reference implementations for Phase A benchmark tasks.
 *
 * Usage:
 *   ./solutions <task-id> <json-input>
 *
 * Examples:
 *   ./solutions task-a-0001 '[1,2,3]'
 *   ./solutions task-a-0010 '7'
 *   ./solutions task-a-0038 '[[1,2,3],[4,5,6]]'
 *
 * Output: JSON-encoded result on stdout.
 *
 * Standard C11, no external dependencies.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Simple JSON-ish parser for the subset we need:
 *   - integers (possibly negative)
 *   - flat arrays of integers:  [1, -2, 3]
 *   - pairs/triples:            [42, 7]
 *   - nested array pairs:       [[1,2,3],[4,5,6]]
 *   - booleans: true / false (parsed from input, printed on output)
 * ----------------------------------------------------------------------- */

#define MAX_ELEMS  4096
#define MAX_LISTS  64
#define MAX_LIST_LEN 4096

typedef struct {
    int64_t elems[MAX_ELEMS];
    int     len;
} IntArray;

typedef struct {
    IntArray lists[MAX_LISTS];
    int      num_lists;
} NestedArray;

/* Skip whitespace; return pointer to next non-space char. */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a single integer from the string; advance *pp past it. */
static int64_t parse_int(const char **pp) {
    const char *p = *pp;
    p = skip_ws(p);
    int64_t sign = 1;
    if (*p == '-') { sign = -1; p++; }
    int64_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *pp = p;
    return sign * val;
}

/* Parse a flat array of ints: [1, -2, 3].  Returns element count. */
static int parse_int_array(const char **pp, int64_t *out, int max) {
    const char *p = *pp;
    p = skip_ws(p);
    if (*p != '[') { fprintf(stderr, "Expected '['\n"); exit(2); }
    p++;
    p = skip_ws(p);
    int n = 0;
    if (*p == ']') { p++; *pp = p; return 0; }
    while (1) {
        if (n >= max) { fprintf(stderr, "Array too large\n"); exit(2); }
        out[n++] = parse_int(&p);
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        fprintf(stderr, "Unexpected char '%c' in array\n", *p);
        exit(2);
    }
    *pp = p;
    return n;
}

/* Parse a nested array: [[1,2],[3,4]] into NestedArray. */
static void parse_nested_array(const char **pp, NestedArray *na) {
    const char *p = *pp;
    p = skip_ws(p);
    if (*p != '[') { fprintf(stderr, "Expected '[' for nested\n"); exit(2); }
    p++;
    p = skip_ws(p);
    na->num_lists = 0;
    if (*p == ']') { p++; *pp = p; return; }
    while (1) {
        p = skip_ws(p);
        int idx = na->num_lists++;
        na->lists[idx].len = parse_int_array(&p, na->lists[idx].elems, MAX_LIST_LEN);
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        fprintf(stderr, "Unexpected char '%c' in nested array\n", *p);
        exit(2);
    }
    *pp = p;
}

/*
 * Detect whether the input is:
 *   - a scalar integer (possibly negative)
 *   - a flat array  [1,2,3]
 *   - a nested array [[1,2],[3,4]]
 *   - a mixed pair  [[1,2,3], 5]
 */
typedef enum { INPUT_SCALAR, INPUT_FLAT_ARRAY, INPUT_NESTED_ARRAY, INPUT_MIXED_PAIR } InputKind;

static InputKind detect_input_kind(const char *json) {
    const char *p = skip_ws(json);
    if (*p != '[') return INPUT_SCALAR;
    p++; p = skip_ws(p);
    if (*p == ']') return INPUT_FLAT_ARRAY; /* empty array */
    if (*p == '[') return INPUT_NESTED_ARRAY;
    /* It's [something...  -- check if all elements are scalars */
    /* Scan for nested '[' after the opening one */
    int depth = 1;
    while (*p && depth > 0) {
        if (*p == '[') return INPUT_NESTED_ARRAY; /* found inner array */
        if (*p == ']') depth--;
        p++;
    }
    return INPUT_FLAT_ARRAY;
}

/* -----------------------------------------------------------------------
 * Output helpers
 * ----------------------------------------------------------------------- */

static void print_int(int64_t v) {
    printf("%lld\n", (long long)v);
}

static void print_bool(bool v) {
    printf("%s\n", v ? "true" : "false");
}

static void print_int_array(const int64_t *arr, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)arr[i]);
    }
    printf("]\n");
}

static void print_nested_array(const int64_t *flat, const int *chunk_lens, int num_chunks) {
    printf("[");
    int offset = 0;
    for (int c = 0; c < num_chunks; c++) {
        if (c > 0) printf(", ");
        printf("[");
        for (int i = 0; i < chunk_lens[c]; i++) {
            if (i > 0) printf(", ");
            printf("%lld", (long long)flat[offset + i]);
        }
        printf("]");
        offset += chunk_lens[c];
    }
    printf("]\n");
}

/* -----------------------------------------------------------------------
 * GCD helper
 * ----------------------------------------------------------------------- */
static int64_t gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int64_t t = b; b = a % b; a = t; }
    return a;
}

/* -----------------------------------------------------------------------
 * Integer square root (floor)
 * ----------------------------------------------------------------------- */
static int64_t isqrt(int64_t n) {
    if (n < 0) return 0;
    if (n == 0) return 0;
    int64_t x = (int64_t)sqrt((double)n);
    /* Correct for floating-point imprecision */
    while (x * x > n) x--;
    while ((x + 1) * (x + 1) <= n) x++;
    return x;
}

/* -----------------------------------------------------------------------
 * Sorting helpers
 * ----------------------------------------------------------------------- */
static int cmp_asc(const void *a, const void *b) {
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
}

static int cmp_desc(const void *a, const void *b) {
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (vb > va) - (vb < va);
}

/* -----------------------------------------------------------------------
 * Task implementations
 * ----------------------------------------------------------------------- */

/* task-a-0001: sum of list */
static void task_a_0001(const int64_t *arr, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += arr[i];
    print_int(s);
}

/* task-a-0002: product of list */
static void task_a_0002(const int64_t *arr, int n) {
    int64_t p = 1;
    for (int i = 0; i < n; i++) p *= arr[i];
    print_int(p);
}

/* task-a-0003: min of list */
static void task_a_0003(const int64_t *arr, int n) {
    int64_t m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] < m) m = arr[i];
    print_int(m);
}

/* task-a-0004: max of list */
static void task_a_0004(const int64_t *arr, int n) {
    int64_t m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    print_int(m);
}

/* task-a-0005: mean (floor division) */
static void task_a_0005(const int64_t *arr, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += arr[i];
    /* Python // is floor division (rounds toward -inf) */
    int64_t q = s / n;
    if ((s % n != 0) && ((s ^ n) < 0)) q--;
    print_int(q);
}

/* task-a-0006: a % b (Python-style modulo) */
static void task_a_0006(const int64_t *arr, int n) {
    (void)n;
    int64_t a = arr[0], b = arr[1];
    int64_t r = a % b;
    if (r != 0 && ((r ^ b) < 0)) r += b;
    print_int(r);
}

/* task-a-0007: power (base ** exp) */
static void task_a_0007(const int64_t *arr, int n) {
    (void)n;
    int64_t base = arr[0], exp = arr[1];
    int64_t result = 1;
    /* Handle negative exponent: integer result is 0 for |base|>1 */
    if (exp < 0) {
        if (base == 1) { print_int(1); return; }
        if (base == -1) { print_int((exp % 2 == 0) ? 1 : -1); return; }
        print_int(0);
        return;
    }
    int64_t b = base;
    int64_t e = exp;
    while (e > 0) {
        if (e & 1) result *= b;
        b *= b;
        e >>= 1;
    }
    print_int(result);
}

/* task-a-0008: gcd */
static void task_a_0008(const int64_t *arr, int n) {
    (void)n;
    print_int(gcd(arr[0], arr[1]));
}

/* task-a-0009: lcm */
static void task_a_0009(const int64_t *arr, int n) {
    (void)n;
    int64_t a = arr[0], b = arr[1];
    if (a == 0 || b == 0) { print_int(0); return; }
    int64_t aa = a < 0 ? -a : a;
    int64_t ab = b < 0 ? -b : b;
    print_int(aa / gcd(aa, ab) * ab);
}

/* task-a-0010: is_prime */
static void task_a_0010_impl(int64_t num) {
    if (num < 2) { print_bool(false); return; }
    if (num < 4) { print_bool(true); return; }
    if (num % 2 == 0 || num % 3 == 0) { print_bool(false); return; }
    for (int64_t i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) { print_bool(false); return; }
    }
    print_bool(true);
}

/* task-a-0011: fibonacci(n) */
static void task_a_0011_impl(int64_t num) {
    if (num <= 0) { print_int(0); return; }
    int64_t a = 0, b = 1;
    for (int64_t i = 1; i < num; i++) {
        int64_t t = a + b;
        a = b;
        b = t;
    }
    print_int(b);
}

/* task-a-0012: factorial(n) */
static void task_a_0012_impl(int64_t num) {
    int64_t r = 1;
    for (int64_t i = 2; i <= num; i++) r *= i;
    print_int(r);
}

/* task-a-0013: absolute value */
static void task_a_0013_impl(int64_t num) {
    print_int(num < 0 ? -num : num);
}

/* task-a-0014: negate */
static void task_a_0014_impl(int64_t num) {
    print_int(-num);
}

/* task-a-0015: integer square root */
static void task_a_0015_impl(int64_t num) {
    print_int(isqrt(num));
}

/* task-a-0016: clamp(v, lo, hi) */
static void task_a_0016(const int64_t *arr, int n) {
    (void)n;
    int64_t v = arr[0], lo = arr[1], hi = arr[2];
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    print_int(v);
}

/* task-a-0017: sum of squares */
static void task_a_0017(const int64_t *arr, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += arr[i] * arr[i];
    print_int(s);
}

/* task-a-0018: count even */
static void task_a_0018(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 2 == 0) c++;
    print_int(c);
}

/* task-a-0019: count odd */
static void task_a_0019(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 2 != 0) c++;
    print_int(c);
}

/* task-a-0020: sum positive */
static void task_a_0020(const int64_t *arr, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) if (arr[i] > 0) s += arr[i];
    print_int(s);
}

/* task-a-0021: count positive */
static void task_a_0021(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] > 0) c++;
    print_int(c);
}

/* task-a-0022: count negative */
static void task_a_0022(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] < 0) c++;
    print_int(c);
}

/* task-a-0023: range (max - min) */
static void task_a_0023(const int64_t *arr, int n) {
    int64_t mn = arr[0], mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }
    print_int(mx - mn);
}

/* task-a-0024: sum even */
static void task_a_0024(const int64_t *arr, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 2 == 0) s += arr[i];
    print_int(s);
}

/* task-a-0025: multiply a*b */
static void task_a_0025(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0] * arr[1]);
}

/* task-a-0026: floor division a//b (Python-style) */
static void task_a_0026(const int64_t *arr, int n) {
    (void)n;
    int64_t a = arr[0], b = arr[1];
    int64_t q = a / b;
    if ((a % b != 0) && ((a ^ b) < 0)) q--;
    print_int(q);
}

/* task-a-0027: square */
static void task_a_0027_impl(int64_t num) {
    print_int(num * num);
}

/* task-a-0028: cube */
static void task_a_0028_impl(int64_t num) {
    print_int(num * num * num);
}

/* task-a-0029: add a + b */
static void task_a_0029(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0] + arr[1]);
}

/* task-a-0030: subtract a - b */
static void task_a_0030(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0] - arr[1]);
}

/* task-a-0031: is_even */
static void task_a_0031_impl(int64_t num) {
    print_bool(num % 2 == 0);
}

/* task-a-0032: is_odd */
static void task_a_0032_impl(int64_t num) {
    print_bool(num % 2 != 0);
}

/* task-a-0033: is_divisible(n, k) */
static void task_a_0033(const int64_t *arr, int n) {
    (void)n;
    print_bool(arr[0] % arr[1] == 0);
}

/* task-a-0034: max of two */
static void task_a_0034(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0] > arr[1] ? arr[0] : arr[1]);
}

/* task-a-0035: min of two */
static void task_a_0035(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0] < arr[1] ? arr[0] : arr[1]);
}

/* task-a-0036: length */
static void task_a_0036(const int64_t *arr, int n) {
    (void)arr;
    print_int(n);
}

/* task-a-0037: prefix sum */
static void task_a_0037(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int64_t s = 0;
    for (int i = 0; i < n; i++) {
        s += arr[i];
        out[i] = s;
    }
    print_int_array(out, n);
}

/* task-a-0038: elementwise sum of two arrays */
static void task_a_0038(const NestedArray *na) {
    const IntArray *a = &na->lists[0];
    const IntArray *b = &na->lists[1];
    int len = a->len < b->len ? a->len : b->len;
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < len; i++) {
        out[i] = a->elems[i] + b->elems[i];
    }
    print_int_array(out, len);
}

/* task-a-0039: count zeros */
static void task_a_0039(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] == 0) c++;
    print_int(c);
}

/* task-a-0040: max absolute value */
static void task_a_0040(const int64_t *arr, int n) {
    int64_t m = 0;
    for (int i = 0; i < n; i++) {
        int64_t a = arr[i] < 0 ? -arr[i] : arr[i];
        if (a > m) m = a;
    }
    print_int(m);
}

/* task-a-0041: count divisible by 3 */
static void task_a_0041(const int64_t *arr, int n) {
    int64_t c = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 3 == 0) c++;
    print_int(c);
}

/* task-a-0042: sort ascending */
static void task_a_0042(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    memcpy(out, arr, n * sizeof(int64_t));
    qsort(out, n, sizeof(int64_t), cmp_asc);
    print_int_array(out, n);
}

/* task-a-0043: sort descending */
static void task_a_0043(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    memcpy(out, arr, n * sizeof(int64_t));
    qsort(out, n, sizeof(int64_t), cmp_desc);
    print_int_array(out, n);
}

/* task-a-0044: reverse list */
static void task_a_0044(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < n; i++) out[i] = arr[n - 1 - i];
    print_int_array(out, n);
}

/* task-a-0045: filter positive */
static void task_a_0045(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < n; i++) if (arr[i] > 0) out[m++] = arr[i];
    print_int_array(out, m);
}

/* task-a-0046: filter negative */
static void task_a_0046(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < n; i++) if (arr[i] < 0) out[m++] = arr[i];
    print_int_array(out, m);
}

/* task-a-0047: filter even */
static void task_a_0047(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 2 == 0) out[m++] = arr[i];
    print_int_array(out, m);
}

/* task-a-0048: filter odd */
static void task_a_0048(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < n; i++) if (arr[i] % 2 != 0) out[m++] = arr[i];
    print_int_array(out, m);
}

/* task-a-0049: double elements */
static void task_a_0049(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < n; i++) out[i] = arr[i] * 2;
    print_int_array(out, n);
}

/* task-a-0050: square elements */
static void task_a_0050(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < n; i++) out[i] = arr[i] * arr[i];
    print_int_array(out, n);
}

/* task-a-0051: negate elements */
static void task_a_0051(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < n; i++) out[i] = -arr[i];
    print_int_array(out, n);
}

/* task-a-0052: add constant k to each element; input = [[list], k] */
static void task_a_0052(const NestedArray *na, const char *json) {
    /*
     * Input is [list, k] where list is an array and k is a scalar.
     * We need to re-parse: the first element is an array, the second is a scalar.
     */
    const char *p = skip_ws(json);
    if (*p != '[') { fprintf(stderr, "task-a-0052: expected '['\n"); exit(2); }
    p++;
    p = skip_ws(p);

    int64_t list_elems[MAX_ELEMS];
    int list_len = parse_int_array(&p, list_elems, MAX_ELEMS);

    p = skip_ws(p);
    if (*p == ',') p++;
    int64_t k = parse_int(&p);

    int64_t out[MAX_ELEMS];
    for (int i = 0; i < list_len; i++) out[i] = list_elems[i] + k;
    print_int_array(out, list_len);
    (void)na;
}

/* task-a-0053: deduplicate (preserve order) */
static void task_a_0053(const int64_t *arr, int n) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < n; i++) {
        bool dup = false;
        for (int j = 0; j < m; j++) {
            if (out[j] == arr[i]) { dup = true; break; }
        }
        if (!dup) out[m++] = arr[i];
    }
    print_int_array(out, m);
}

/* Helper: parse [list, k] where list is array and k is scalar int */
static int parse_list_and_scalar(const char *json, int64_t *list_out, int max, int64_t *k_out) {
    const char *p = skip_ws(json);
    if (*p != '[') { fprintf(stderr, "Expected outer '['\n"); exit(2); }
    p++;
    p = skip_ws(p);
    int len = parse_int_array(&p, list_out, max);
    p = skip_ws(p);
    if (*p == ',') p++;
    *k_out = parse_int(&p);
    return len;
}

/* task-a-0054: rotate left by k */
static void task_a_0054(const char *json) {
    int64_t list_elems[MAX_ELEMS];
    int64_t k;
    int len = parse_list_and_scalar(json, list_elems, MAX_ELEMS, &k);
    if (len == 0) { print_int_array(list_elems, 0); return; }
    k = ((k % len) + len) % len; /* handle negative k */
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < len; i++) {
        out[i] = list_elems[(i + k) % len];
    }
    print_int_array(out, len);
}

/* task-a-0055: rotate right by k */
static void task_a_0055(const char *json) {
    int64_t list_elems[MAX_ELEMS];
    int64_t k;
    int len = parse_list_and_scalar(json, list_elems, MAX_ELEMS, &k);
    if (len == 0) { print_int_array(list_elems, 0); return; }
    k = ((k % len) + len) % len;
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < len; i++) {
        out[i] = list_elems[(i - k + len) % len];
    }
    print_int_array(out, len);
}

/* task-a-0056: flatten (list of lists -> flat list) */
static void task_a_0056(const NestedArray *na) {
    int64_t out[MAX_ELEMS];
    int m = 0;
    for (int i = 0; i < na->num_lists; i++) {
        for (int j = 0; j < na->lists[i].len; j++) {
            out[m++] = na->lists[i].elems[j];
        }
    }
    print_int_array(out, m);
}

/* task-a-0057: chunk list into sublists of size k; input = [list, k] */
static void task_a_0057(const char *json) {
    int64_t list_elems[MAX_ELEMS];
    int64_t k;
    int len = parse_list_and_scalar(json, list_elems, MAX_ELEMS, &k);
    if (len == 0 || k <= 0) { printf("[]\n"); return; }

    int num_chunks = (len + (int)k - 1) / (int)k;
    int chunk_lens[MAX_ELEMS];
    for (int c = 0; c < num_chunks; c++) {
        int start = c * (int)k;
        int clen = (int)k;
        if (start + clen > len) clen = len - start;
        chunk_lens[c] = clen;
    }
    print_nested_array(list_elems, chunk_lens, num_chunks);
}

/* task-a-0058: sliding window max; input = [list, k] */
static void task_a_0058(const char *json) {
    int64_t list_elems[MAX_ELEMS];
    int64_t k;
    int len = parse_list_and_scalar(json, list_elems, MAX_ELEMS, &k);
    if (len == 0) { print_int_array(list_elems, 0); return; }
    int win = (int)k;
    int out_len = len - win + 1;
    if (out_len <= 0) { print_int_array(list_elems, 0); return; }
    int64_t out[MAX_ELEMS];
    for (int i = 0; i < out_len; i++) {
        int64_t mx = list_elems[i];
        for (int j = 1; j < win; j++) {
            if (list_elems[i + j] > mx) mx = list_elems[i + j];
        }
        out[i] = mx;
    }
    print_int_array(out, out_len);
}

/* task-a-0059: sliding window sum; input = [list, k] */
static void task_a_0059(const char *json) {
    int64_t list_elems[MAX_ELEMS];
    int64_t k;
    int len = parse_list_and_scalar(json, list_elems, MAX_ELEMS, &k);
    if (len == 0) { print_int_array(list_elems, 0); return; }
    int win = (int)k;
    int out_len = len - win + 1;
    if (out_len <= 0) { print_int_array(list_elems, 0); return; }
    int64_t out[MAX_ELEMS];
    int64_t s = 0;
    for (int j = 0; j < win; j++) s += list_elems[j];
    out[0] = s;
    for (int i = 1; i < out_len; i++) {
        s += list_elems[i + win - 1] - list_elems[i - 1];
        out[i] = s;
    }
    print_int_array(out, out_len);
}

/* task-a-0060: first element */
static void task_a_0060(const int64_t *arr, int n) {
    (void)n;
    print_int(arr[0]);
}

/* -----------------------------------------------------------------------
 * Dispatch table
 * ----------------------------------------------------------------------- */

/*
 * Task categories for dispatch:
 *   SCALAR   - input is a single integer
 *   FLAT     - input is a flat array of ints
 *   NESTED   - input is a nested array (e.g. [[1,2],[3,4]])
 *   MIXED    - input is [array, scalar] -- needs raw JSON re-parse
 */

typedef enum { CAT_SCALAR, CAT_FLAT, CAT_NESTED, CAT_MIXED } TaskCategory;

typedef struct {
    const char   *id;
    TaskCategory  cat;
    int           task_num; /* for dispatch */
} TaskEntry;

static const TaskEntry TASKS[] = {
    {"task-a-0001", CAT_FLAT,   1},
    {"task-a-0002", CAT_FLAT,   2},
    {"task-a-0003", CAT_FLAT,   3},
    {"task-a-0004", CAT_FLAT,   4},
    {"task-a-0005", CAT_FLAT,   5},
    {"task-a-0006", CAT_FLAT,   6},
    {"task-a-0007", CAT_FLAT,   7},
    {"task-a-0008", CAT_FLAT,   8},
    {"task-a-0009", CAT_FLAT,   9},
    {"task-a-0010", CAT_SCALAR, 10},
    {"task-a-0011", CAT_SCALAR, 11},
    {"task-a-0012", CAT_SCALAR, 12},
    {"task-a-0013", CAT_SCALAR, 13},
    {"task-a-0014", CAT_SCALAR, 14},
    {"task-a-0015", CAT_SCALAR, 15},
    {"task-a-0016", CAT_FLAT,   16},
    {"task-a-0017", CAT_FLAT,   17},
    {"task-a-0018", CAT_FLAT,   18},
    {"task-a-0019", CAT_FLAT,   19},
    {"task-a-0020", CAT_FLAT,   20},
    {"task-a-0021", CAT_FLAT,   21},
    {"task-a-0022", CAT_FLAT,   22},
    {"task-a-0023", CAT_FLAT,   23},
    {"task-a-0024", CAT_FLAT,   24},
    {"task-a-0025", CAT_FLAT,   25},
    {"task-a-0026", CAT_FLAT,   26},
    {"task-a-0027", CAT_SCALAR, 27},
    {"task-a-0028", CAT_SCALAR, 28},
    {"task-a-0029", CAT_FLAT,   29},
    {"task-a-0030", CAT_FLAT,   30},
    {"task-a-0031", CAT_SCALAR, 31},
    {"task-a-0032", CAT_SCALAR, 32},
    {"task-a-0033", CAT_FLAT,   33},
    {"task-a-0034", CAT_FLAT,   34},
    {"task-a-0035", CAT_FLAT,   35},
    {"task-a-0036", CAT_FLAT,   36},
    {"task-a-0037", CAT_FLAT,   37},
    {"task-a-0038", CAT_NESTED, 38},
    {"task-a-0039", CAT_FLAT,   39},
    {"task-a-0040", CAT_FLAT,   40},
    {"task-a-0041", CAT_FLAT,   41},
    {"task-a-0042", CAT_FLAT,   42},
    {"task-a-0043", CAT_FLAT,   43},
    {"task-a-0044", CAT_FLAT,   44},
    {"task-a-0045", CAT_FLAT,   45},
    {"task-a-0046", CAT_FLAT,   46},
    {"task-a-0047", CAT_FLAT,   47},
    {"task-a-0048", CAT_FLAT,   48},
    {"task-a-0049", CAT_FLAT,   49},
    {"task-a-0050", CAT_FLAT,   50},
    {"task-a-0051", CAT_FLAT,   51},
    {"task-a-0052", CAT_MIXED,  52},
    {"task-a-0053", CAT_FLAT,   53},
    {"task-a-0054", CAT_MIXED,  54},
    {"task-a-0055", CAT_MIXED,  55},
    {"task-a-0056", CAT_NESTED, 56},
    {"task-a-0057", CAT_MIXED,  57},
    {"task-a-0058", CAT_MIXED,  58},
    {"task-a-0059", CAT_MIXED,  59},
    {"task-a-0060", CAT_FLAT,   60},
    {NULL, 0, 0}
};

static const TaskEntry *find_task(const char *id) {
    for (int i = 0; TASKS[i].id != NULL; i++) {
        if (strcmp(TASKS[i].id, id) == 0) return &TASKS[i];
    }
    return NULL;
}

/* -----------------------------------------------------------------------
 * Scalar-input dispatch
 * ----------------------------------------------------------------------- */
static void dispatch_scalar(int task_num, int64_t val) {
    switch (task_num) {
        case 10: task_a_0010_impl(val); break;
        case 11: task_a_0011_impl(val); break;
        case 12: task_a_0012_impl(val); break;
        case 13: task_a_0013_impl(val); break;
        case 14: task_a_0014_impl(val); break;
        case 15: task_a_0015_impl(val); break;
        case 27: task_a_0027_impl(val); break;
        case 28: task_a_0028_impl(val); break;
        case 31: task_a_0031_impl(val); break;
        case 32: task_a_0032_impl(val); break;
        default:
            fprintf(stderr, "Unknown scalar task %d\n", task_num);
            exit(2);
    }
}

/* -----------------------------------------------------------------------
 * Flat-array-input dispatch
 * ----------------------------------------------------------------------- */
static void dispatch_flat(int task_num, const int64_t *arr, int n) {
    switch (task_num) {
        case  1: task_a_0001(arr, n); break;
        case  2: task_a_0002(arr, n); break;
        case  3: task_a_0003(arr, n); break;
        case  4: task_a_0004(arr, n); break;
        case  5: task_a_0005(arr, n); break;
        case  6: task_a_0006(arr, n); break;
        case  7: task_a_0007(arr, n); break;
        case  8: task_a_0008(arr, n); break;
        case  9: task_a_0009(arr, n); break;
        case 16: task_a_0016(arr, n); break;
        case 17: task_a_0017(arr, n); break;
        case 18: task_a_0018(arr, n); break;
        case 19: task_a_0019(arr, n); break;
        case 20: task_a_0020(arr, n); break;
        case 21: task_a_0021(arr, n); break;
        case 22: task_a_0022(arr, n); break;
        case 23: task_a_0023(arr, n); break;
        case 24: task_a_0024(arr, n); break;
        case 25: task_a_0025(arr, n); break;
        case 26: task_a_0026(arr, n); break;
        case 29: task_a_0029(arr, n); break;
        case 30: task_a_0030(arr, n); break;
        case 33: task_a_0033(arr, n); break;
        case 34: task_a_0034(arr, n); break;
        case 35: task_a_0035(arr, n); break;
        case 36: task_a_0036(arr, n); break;
        case 37: task_a_0037(arr, n); break;
        case 39: task_a_0039(arr, n); break;
        case 40: task_a_0040(arr, n); break;
        case 41: task_a_0041(arr, n); break;
        case 42: task_a_0042(arr, n); break;
        case 43: task_a_0043(arr, n); break;
        case 44: task_a_0044(arr, n); break;
        case 45: task_a_0045(arr, n); break;
        case 46: task_a_0046(arr, n); break;
        case 47: task_a_0047(arr, n); break;
        case 48: task_a_0048(arr, n); break;
        case 49: task_a_0049(arr, n); break;
        case 50: task_a_0050(arr, n); break;
        case 51: task_a_0051(arr, n); break;
        case 53: task_a_0053(arr, n); break;
        case 60: task_a_0060(arr, n); break;
        default:
            fprintf(stderr, "Unknown flat task %d\n", task_num);
            exit(2);
    }
}

/* -----------------------------------------------------------------------
 * Nested-array-input dispatch
 * ----------------------------------------------------------------------- */
static void dispatch_nested(int task_num, const NestedArray *na) {
    switch (task_num) {
        case 38: task_a_0038(na); break;
        case 56: task_a_0056(na); break;
        default:
            fprintf(stderr, "Unknown nested task %d\n", task_num);
            exit(2);
    }
}

/* -----------------------------------------------------------------------
 * Mixed-input dispatch (needs raw JSON)
 * ----------------------------------------------------------------------- */
static void dispatch_mixed(int task_num, const char *json, const NestedArray *na) {
    switch (task_num) {
        case 52: task_a_0052(na, json); break;
        case 54: task_a_0054(json); break;
        case 55: task_a_0055(json); break;
        case 57: task_a_0057(json); break;
        case 58: task_a_0058(json); break;
        case 59: task_a_0059(json); break;
        default:
            fprintf(stderr, "Unknown mixed task %d\n", task_num);
            exit(2);
    }
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <task-id> <json-input>\n", argv[0]);
        fprintf(stderr, "Example: %s task-a-0001 '[1,2,3]'\n", argv[0]);
        return 1;
    }

    const char *task_id = argv[1];
    const char *json    = argv[2];

    const TaskEntry *te = find_task(task_id);
    if (!te) {
        fprintf(stderr, "Unknown task: %s\n", task_id);
        return 1;
    }

    switch (te->cat) {
        case CAT_SCALAR: {
            const char *p = json;
            int64_t val = parse_int(&p);
            dispatch_scalar(te->task_num, val);
            break;
        }
        case CAT_FLAT: {
            int64_t arr[MAX_ELEMS];
            const char *p = json;
            int n = parse_int_array(&p, arr, MAX_ELEMS);
            dispatch_flat(te->task_num, arr, n);
            break;
        }
        case CAT_NESTED: {
            NestedArray na;
            const char *p = json;
            parse_nested_array(&p, &na);
            dispatch_nested(te->task_num, &na);
            break;
        }
        case CAT_MIXED: {
            NestedArray na;
            /* Some mixed tasks need the raw JSON, others need parsed nested array */
            dispatch_mixed(te->task_num, json, &na);
            break;
        }
    }

    return 0;
}
