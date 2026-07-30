# Within-window substep coefficients

## Partition

The finite later-time interval has 15 windows:

```text
[3/8, 6] = union_(j=1)^15 [3j/8, 3(j+1)/8].
```

Each window is divided into ten subslabs of length `3/80 = 0.0375`.
The endpoint at `t=6` anchors the separate post-terminal tail and does not
start a sixteenth finite window.

## Coefficients

On the already-certified stored spectrum `[1.9, 8000]`, the substep
semigroup is expanded through degree 112. Directed 80-decimal interval
arithmetic encloses the scaled modified-Bessel coefficients. The exact
positive series is used for orders zero and one, forward recurrence supplies
orders through 112, and direct positive series plus a geometric order tail
encloses every omitted coefficient.

The SciPy binary64 coefficients are compared with the exact intervals. They
need not be bitwise contained: for every order the maximum distance from the
central value to either exact endpoint encloses its implementation error.
The sum of those per-order errors and the infinite coefficient tail are
separate charges for the later sparse-action certificate.

## Scope

This is a scalar coefficient certificate only. It does not yet certify the
degree-112 sparse recurrence, the 160 substep states, any within-window
supremum, or the post-time-6 tail.

The executable is
`scripts/neutral_strip_within_window_substep_coefficients_certificate.py`.
