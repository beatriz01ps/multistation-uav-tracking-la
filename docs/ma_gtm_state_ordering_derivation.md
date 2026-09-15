# Ma-inspired GTM — state ordering

This note records the state-vector ordering used when extending the Ma-inspired equations from 2D to 3D.

## Paper notation

The paper shows the state as:

`[px, vx, py, vy]^T`

in the simulation description.

Its transition matrix, however, is written in block form:

```text
F = [ I(n)   Δt I(n) ]
    [ 0(n)      I(n) ]
```

That matrix only works directly when the state is ordered as:

`[position block, velocity block]`.

For `n = 2`, this means:

`[px, py, vx, vy]^T`

With that ordering, the transition implements:

`p(t + Δt) = p(t) + Δt v(t)`.

## Ordering used here

The project already uses:

`[x, y, z, vx, vy, vz]`

with the position block first and the velocity block second.

This is the same ordering extended from 2D to 3D.

No state permutation is needed before applying the Ma-inspired Constant Velocity equations.

The functions `_cv_transition_matrix(dt, n=3)` and `_cv_process_noise(dt, sigma_v, n=3)` therefore operate directly on `[x, y, z, vx, vy, vz]`.

## Sanity check

Using different nonzero values for `vx`, `vy`, and `vz`, prediction through a gap must satisfy, axis by axis:

`p(t + Δt) = p(t) + Δt v(t)`.

A wrong state ordering would fail this check immediately.
