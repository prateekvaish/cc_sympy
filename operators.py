from sympy import symbols, Dummy, Rational, zeros
from sympy.physics.secondquant import (
    F,
    Fd,
    NO,
    AntiSymmetricTensor,
    wicks,
    substitute_dummies,
    evaluate_deltas,
    Commutator,
)


def get_ts():
    """
    Return T1 operator
    """
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)
    return AntiSymmetricTensor("t", upper=(a,), lower=(i,)) * NO(Fd(a) * F(i))


def get_tsd():
    """
    returns adjoint of T1 operator
    """
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)
    return AntiSymmetricTensor("t", upper=(a,), lower=(i,)) * NO(Fd(i) * F(a))


def get_td():
    """
    Return T2 operator
    """
    i, j = symbols("i j", below_fermi=True, cls=Dummy)
    a, b = symbols("a b", above_fermi=True, cls=Dummy)
    return (
        Rational(1, 4)
        * AntiSymmetricTensor("t", (a, b), (i, j))
        * NO(Fd(a) * F(i) * Fd(b) * F(j))
    )


def get_tdd():
    """
    Return adjoint of T2 operator
    """
    i, j = symbols("i j", below_fermi=True, cls=Dummy)
    a, b = symbols("a b", above_fermi=True, cls=Dummy)
    return (
        Rational(1, 4)
        * AntiSymmetricTensor("t", (a, b), (i, j))
        * NO(Fd(j) * F(b) * Fd(i) * F(a))
    )


def get_f():
    """
    returns normal-ordered one-electron operator
    """
    p, q = symbols("p q", cls=Dummy)
    return AntiSymmetricTensor("f", (p,), (q,)) * NO(Fd(p) * F(q))


def get_v():
    """
    returns normal-ordered two-electron operator
    """
    p, q, r, s = symbols("p q r s", cls=Dummy)
    v = (
        Rational(1, 4)
        * AntiSymmetricTensor("g", (p, q), (r, s))
        * NO(Fd(p) * Fd(q) * F(s) * F(r))
    )
    return v


def BCH_expand(h, t, num_terms):
    """
    Expand the Baker-Campbell-Hausdorff series up to a given order
    e^{-T} H e^{T} = H + [H, T] + 1/2 [[H, T], T] + 1/6 [[[H, T], T], T] + ...
    """
    symbol_rules = {"above": "defg", "below": "lmno", "general": "pqrst"}

    bch = zeros(5)
    bch[0] = h
    for i in range(num_terms):
        comm = Commutator(bch[i], t)
        bch[i + 1] = wicks(comm)
        bch[i + 1] = evaluate_deltas(bch[i + 1])
        bch[i + 1] = substitute_dummies(bch[i + 1], new_indices=True)

    # BCH series
    BCH = bch[0] + bch[1] + bch[2] / 2 + bch[3] / 6 + bch[4] / 24

    # tidy up and compact
    BCH = BCH.expand()
    BCH = evaluate_deltas(BCH)
    BCH = substitute_dummies(BCH, new_indices=True, pretty_indices=symbol_rules)

    return BCH
