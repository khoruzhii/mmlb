# Rank-one steps from matrix multiplication

The files are row-aligned datasets for `T + z`, where `T` is a binary
matrix-multiplication tensor and `z` runs through the representatives generated
by `scripts/rank1_orbits.cpp`.

## Format

- `tXYZ.npy`: C-order `uint64`, shape `(N, 12)`. Each row is one packed 9x9x9
  F2 tensor; coordinate `(i,j,k)` is bit `(9*i+j)*9+k`. The upper 39 padding
  bits of the last word are zero.
- `dXYZ.npy`: C-order `uint16`, shape `(N,)`. Each value is the row-aligned
  upper bound returned by `fgs::ubd(tensor, 1'000'000'000)` with seed 42,
  initial reduce2, and plus every 2k successful flips.

For M222 and M333 the ordered rank-one orbits are merged under the square S3
action implemented by the generator. M223 and M233 use its ordered
representatives because the generator does not implement the additional
partial symmetry merge when only two matrix dimensions agree.

| tensor | tensor shape | upper-bound shape | mean UB | UB distribution |
|---|---:|---:|---:|---|
| M222 | `(13,12)` | `(13,)` | 7.000000 | 6:2, 7:9, 8:2 |
| M223 | `(41,12)` | `(41,)` | 10.756098 | 10:10, 11:31 |
| M233 | `(79,12)` | `(79,)` | 14.974684 | 14:6, 15:69, 16:4 |
| M333 | `(67,12)` | `(67,)` | 22.611940 | 22:26, 23:41 |

All 200 decompositions were reconstructed exactly before writing the upper
bounds. The 1B-flip pass took 773.201628 seconds wall time on 12 outer worker
threads.

## Reproduction

```powershell
g++ -std=c++20 -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic `
  -isystem third_party scripts/rank1_orbits.cpp -o bin/rank1_orbits
bin/rank1_orbits --save

g++ -std=c++20 -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic `
  -Isrc -isystem third_party -pthread scripts/rank1_ub.cpp `
  -o bin/rank1_ub
bin/rank1_ub
```
