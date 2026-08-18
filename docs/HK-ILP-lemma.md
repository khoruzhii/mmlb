# HK/ILP Lemma

A primitive version of the substitution method proves rather easily that taking the $4\times 6 \times 6$ tensor $\langle 2,2,3 \rangle$ and then applying any single substitution to the 4 dimensional space gives a tensor of rank exactly 9. In fact it even shows that these tensors have a zero rank drop substitution you can make from them. However, without a proof that the first substitution drops the rank by 2, we are stuck. So the idea is to instead prove that the two substitutions together drop the rank by at least 2. This is done by constructing an integer linear problem that must hold assuming it doesn't happen. 

## Setup

Take tensor $T \in A \otimes B \otimes C$ and let $G \leq Aut(T)$ act on $A$. Let the orbits of $A$ under action by $G$ be $O_1, O_2, \dots, O_n$, and let the orbits of $Gr(d,dim(A))$ (the Grassmannian over $\mathbb{F}_2$) under the action of $G$ be $P_1, P_2, \dots P_m$. Fix a rank $r$ decomposition $T = \sum_{t=1}^r a_t \otimes b_t \otimes c_t$. Suppose no element of $O_i$ is repeated more than $k_i$ times, i.e $|{t : a_t = o_i}| \leq k_i$ for all $o_i \in O_i$, and let $x_i$ denote the number of $a_t$ that are in the orbit $O_i$. Suppose that at most $l_j$ of the $a_t$ lie in a single subspace from the orbit $P_j$.

## Constructing the ILP

1. $\sum_i x_i = r$, as this counts the total number of $a_t$ terms, including multiplicity.
2. $0 \leq x_i \leq k_i \cdot |O_i|$ for all $i$ as we have an upper bound on the number of times each $o_i \in O_i$ can appear.
3. Let $v_{ij}$ denote the number of vectors in $O_i$ that are contained in $P_j$. Then for each $p \in P_j$, setting $p = 0$ (via substitutions) would reduce the rank by however many of the $a_t$ lie in $p$. Not knowing the $a_t$, we don't know which $p$ to choose, so instead we look at the average number of $a_t$ that would be zeroed when $p$ is chosen uniformly at random from $P_j$. Each vector from $O_i$ has a probability $\frac{v_{ij}}{|O_i|}$ of being zeroed, so by linearity of expectation, the total number of expected zeroed terms is $\sum_i x_i \frac{v_{ij}}{|O_i|}$. This cannot exceed $l_j$ so $\sum_i x_i \frac{v_{ij}}{|O_i|} \leq l_j$ for all $j$.

## The lemma and our use of it

The lemma simply says that for the setup to hold, the ILP must have solutions. In the context of our lower bound program, we have a tensor $T$ and we want to prove the rank is at least `target_rank`. Proving this may be hard, because some substitutions drop the rank by 2, and we cannot prove this without further lemmas and theorems. The HK/ILP lemma shows us, via proving that the ILP has no solutions with `target_rank-1`, that something "good" must happen, even if we don't know which good thing it is, allowing us to branch over all possibilities.

### Example: $\langle 2,2,3 \rangle$

Running the program without HK/ILP, we are easily able to show $\langle 2,2,3 \rangle$ has rank at least 10, but we don't know how to prove the rank is at least 11 without millions of branches. This is because the initial substitutions the method makes drop the rank by 2, but we could only prove that it dropped by 1.

We find that the $A$ space has two non-zero orbits of elements, let's call them $O_1, O_2$, characterised by their rank. We also find 5 orbits for 2D subspaces, but we need only look at the orbit of $p = \{0, a_{00}, a_{11}, a_{00} + a_{11}\}$, which we call $P_1$. Note that $p$ (and all subspaces in its orbit) contain two elements of $O_1$ and one element of $O_2$.

Suppose we have a rank 10 decomposition, and that each $a_t$ is distinct (so that no single substitution would drop the rank by at least 2) meaning that in the lemma we are taking $r=10,k_1=k_2=1$, and since we are looking at 2D subspaces, this is $d=2$. We know $x_1+x_2=10$, $0 \leq x_1 \leq 9, 0 \leq x_2 \leq 6$. Now suppose further that the 2D subspaces in $P_1$ contain at most 1 vector from the $a_t$'s, so $l_1 = 1$. Then we have $x_1 \frac{v_{11}}{9} + x_2 \frac{v_{21}}{6} \leq 1$. We know $v_{11} = 2, v_{21} = 1$ and so this completes the system. Of course we could also get constraints for the other 4 2D subspace orbits, but this turns out to be sufficient.

Given that $x_1 + x_2 = 10$ and $0 \leq x_2 \leq 6$ we know $4 \leq x_1 \leq 9$. For each $x_1$ we just need to check if $x_1 \frac29 + (10-x_1)\frac16 \leq 1$ i.e $x_1 \leq -12$. Of course this cannot happen, since $x_1 \geq 4$.

Therefore, one of the following must happen:
1. Any rank 10 decomposition of $\langle 2,2,3 \rangle$ contains two $a_t$ terms that are identical (violating $k_i=1$)
2. Any rank 10 decomposition of $\langle 2,2,3 \rangle$ contains two $a_t$ terms that lie in the same 2D subspace from orbit $P_1$ (violating $l_1=1$)

Now, the primitive version of our program checked that all substitutions on $A$ give a tensor of at least 9, so the first cannot happen. Moreover, the primitive version of our program checked that the resulting tensor after substituting away $p \in P_1$ has rank at least 9, so the second cannot happen. Thus the rank cannot be 10, and must be at least 11, meeting the upper bound.
