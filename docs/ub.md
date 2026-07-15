# Upper bounds over F2

The upper-bound code works with dense binary tensors of shape 9x9x9:

```cpp
namespace fgs::bco {
using Tensor = std::array<std::uint64_t, 12>;
}
```

Tensor entries occupy 729 bits; the upper 39 bits of the last word are zero.
A decomposition is a `std::vector<fgs::Scheme::Term>`, where a term contains
three `U16` bit vectors.

## Basis change optimization

`bco.h` provides greedy basis change optimization:

```cpp
std::uint16_t fgs::bco::nnz(const Tensor &tensor);
std::uint16_t fgs::bco::optimize(Tensor &tensor);
std::uint16_t fgs::bco::optimize(
    Tensor &tensor,
    std::vector<fgs::bco::Transvection> &transvections);
```

It repeatedly applies the coordinate transvection giving the largest strict
decrease in tensor Hamming weight, stopping at a local minimum. The second
overload records the changes so they can later be undone. BCO is an auxiliary
tool and is not currently part of the main `ub` pipeline.

## 2-reduction

`reduce.h` provides:

```cpp
std::uint16_t fgs::reduce2(std::vector<fgs::Scheme::Term> &terms);
```

It searches for 2-reductions in the `UV`, `UW`, and `VW` directions, repeating
the three passes until the decomposition is 2-irreducible. It modifies the
decomposition in place, preserves its tensor exactly, and returns the new rank.

## Flip graph search

The primary API is in `ub.h`:

```cpp
fgs::U16 fgs::ub(const fgs::bco::Tensor &tensor,
                 fgs::U32 flips = 1'000'000);

std::vector<fgs::Scheme::Term>
fgs::ubd(const fgs::bco::Tensor &tensor,
         fgs::U32 flips = 1'000'000);
```

`ubd` starts from the coordinate decomposition, applies `reduce2`, and then
performs a random walk in the flip graph. It performs a plus
transition every 2,000 successful flips, and also tries plus when no flip is
available. The best decomposition seen during the walk is returned. `ub`
returns only its rank; currently it calls `ubd` internally.

Both functions are single-threaded. Independent tensors can be processed in
parallel by the caller. Passing zero flips runs only the initial 2-reduction.
