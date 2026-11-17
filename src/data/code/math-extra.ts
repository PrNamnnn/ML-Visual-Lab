import type { CodeExample } from "@/types/curriculum";
import { py, r } from "./helpers";

/* Generated from verified runs: the output of every program below was captured from an actual execution
   (python -I, standard library only) and is re-checked by tests/code-run.test.ts. */

export const linearAlgebraExtraCode: Record<string, CodeExample> = {
  "four-fundamental-subspaces": py(r`
    A = [[1.0, 2.0], [2.0, 4.0]]

    # reduced row echelon form of a 2 x 2 matrix
    R = [row[:] for row in A]
    R[0] = [v / R[0][0] for v in R[0]]
    R[1] = [b - R[1][0] * a for a, b in zip(R[0], R[1])]
    rank = sum(1 for row in R if any(abs(v) > 1e-12 for v in row))
    print("rref(A) =", R, " rank =", rank)

    row_space = R[0]                      # the pivot row spans the row space
    null_space = [-R[0][1], 1.0]          # free variable x2 = 1
    print("row space basis: ", row_space)
    print("null space basis:", null_space)
    print("A n =", [sum(a * n for a, n in zip(row, null_space)) for row in A])
    print("row . null =", sum(a * b for a, b in zip(row_space, null_space)))
    print("rank + nullity =", rank, "+", 2 - rank, "=", 2)

    # split x into its row-space and null-space parts
    x = [2.0, 0.5]
    c = sum(a * b for a, b in zip(x, row_space)) / sum(a * a for a in row_space)
    x_row = [c * a for a in row_space]
    x_null = [xi - ri for xi, ri in zip(x, x_row)]
    Ax = [sum(a * v for a, v in zip(row, x)) for row in A]
    Ax_row = [sum(a * v for a, v in zip(row, x_row)) for row in A]
    print("x_row =", [round(v, 4) for v in x_row], " x_null =", [round(v, 4) for v in x_null])
    print("A x =", Ax, " A x_row =", [round(v, 4) for v in Ax_row])
  `, r`
    rref(A) = [[1.0, 2.0], [0.0, 0.0]]  rank = 1
    row space basis:  [1.0, 2.0]
    null space basis: [-2.0, 1.0]
    A n = [0.0, 0.0]
    row . null = 0.0
    rank + nullity = 1 + 1 = 2
    x_row = [0.6, 1.2]  x_null = [1.4, -0.7]
    A x = [3.0, 6.0]  A x_row = [3.0, 6.0]
  `, "Row reduction exposes one pivot, so the rank is 1 and the null space is one-dimensional. The null-space vector is orthogonal to the row space, and removing the null-space part of x leaves Ax unchanged."),
  "projection-matrices": py(r`
    A = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]        # columns span a plane in R^3
    b = [1.0, 2.0, 3.0]
    m, k = len(A), len(A[0])

    def matmul(X, Y):
        return [[sum(X[i][t] * Y[t][j] for t in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]

    At = [[A[i][j] for i in range(m)] for j in range(k)]
    G = matmul(At, A)                                # A^T A  (2 x 2)
    det = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    Ginv = [[G[1][1] / det, -G[0][1] / det], [-G[1][0] / det, G[0][0] / det]]
    P = matmul(matmul(A, Ginv), At)                  # P = A (A^T A)^-1 A^T

    p = [sum(P[i][j] * b[j] for j in range(m)) for i in range(m)]
    e = [bi - pi for bi, pi in zip(b, p)]
    print("P =")
    for row in P:
        print("   ", [round(v, 4) for v in row])
    print("projection p =", [round(v, 4) for v in p])
    print("error e =     ", [round(v, 4) for v in e])
    print("A^T e =", [round(abs(sum(At[j][i] * e[i] for i in range(m))), 10) for j in range(k)])

    P2 = matmul(P, P)
    print("max |P^2 - P| =", round(max(abs(P2[i][j] - P[i][j]) for i in range(m) for j in range(m)), 12))
    print("symmetric:", all(abs(P[i][j] - P[j][i]) < 1e-12 for i in range(m) for j in range(m)))
  `, r`
    P =
        [0.8333, -0.1667, 0.3333]
        [-0.1667, 0.8333, 0.3333]
        [0.3333, 0.3333, 0.3333]
    projection p = [1.5, 2.5, 2.0]
    error e =      [-0.5, -0.5, 1.0]
    A^T e = [0.0, 0.0]
    max |P^2 - P| = 0.0
    symmetric: True
  `, "P is assembled from the formula A(AᵀA)⁻¹Aᵀ. The error b − Pb is orthogonal to both columns of A, and P is symmetric and unchanged by squaring: the two properties that characterise an orthogonal projection."),
  "spectral-theorem": py(r`
    import math

    A = [[2.0, 1.0], [1.0, 2.0]]
    a, b, d = A[0][0], A[0][1], A[1][1]

    half, disc = (a + d) / 2, math.sqrt(((a - d) / 2) ** 2 + b * b)
    lam = [half + disc, half - disc]                       # real, because the matrix is symmetric
    q1 = [lam[0] - d, b]
    n1 = math.hypot(*q1)
    q1 = [v / n1 for v in q1]
    q2 = [-q1[1], q1[0]]
    print("eigenvalues:", lam)
    print("q1 =", [round(v, 4) for v in q1], " q2 =", [round(v, 4) for v in q2])
    print("q1 . q2 =", q1[0] * q2[0] + q1[1] * q2[1])

    def outer(u):
        return [[u[i] * u[j] for j in range(2)] for i in range(2)]

    def combine(f):                                         # sum_i f(lambda_i) q_i q_i^T
        P1, P2 = outer(q1), outer(q2)
        return [[f(lam[0]) * P1[i][j] + f(lam[1]) * P2[i][j] for j in range(2)] for i in range(2)]

    show = lambda M: [[round(v, 4) for v in row] for row in M]
    print("sum lambda_i q_i q_i^T =", show(combine(lambda l: l)))
    print("A^-1   (1/lambda)      =", show(combine(lambda l: 1 / l)))
    print("A^1/2  (sqrt lambda)   =", show(combine(math.sqrt)))
    S = combine(math.sqrt)
    print("A^1/2 squared          =", show([[sum(S[i][t] * S[t][j] for t in range(2)) for j in range(2)] for i in range(2)]))
    print("trace =", a + d, "= sum of eigenvalues =", lam[0] + lam[1])
  `, r`
    eigenvalues: [3.0, 1.0]
    q1 = [0.7071, 0.7071]  q2 = [-0.7071, 0.7071]
    q1 . q2 = 0.0
    sum lambda_i q_i q_i^T = [[2.0, 1.0], [1.0, 2.0]]
    A^-1   (1/lambda)      = [[0.6667, -0.3333], [-0.3333, 0.6667]]
    A^1/2  (sqrt lambda)   = [[1.366, 0.366], [0.366, 1.366]]
    A^1/2 squared          = [[2.0, 1.0], [1.0, 2.0]]
    trace = 4.0 = sum of eigenvalues = 4.0
  `, "The eigenvectors of the symmetric matrix are orthonormal, and weighting their rank-one projections by the eigenvalues rebuilds A. Applying a function to the eigenvalues alone gives the inverse and the square root, whose square returns A."),
  "pseudoinverse": py(r`
    import math

    # fewer equations than unknowns: x1 + 2 x2 = 4
    A = [1.0, 2.0]
    b = 4.0
    AAt = sum(a * a for a in A)
    A_plus = [a / AAt for a in A]                     # A^T (A A^T)^-1
    x_plus = [a * b for a in A_plus]
    print("A+ =", A_plus, " x+ =", x_plus, " |x+| =", round(math.hypot(*x_plus), 4))
    for other in ([4.0, 0.0], [0.0, 2.0], [2.0, 1.0]):
        print(f"   another solution {other}: A x = {A[0] * other[0] + A[1] * other[1]}, norm = {math.hypot(*other):.4f}")

    # more equations than unknowns: three lines, one unknown pair
    M = [[1.0, 1.0], [1.0, -1.0], [0.5, 2.0]]
    rhs = [3.0, 0.5, 3.5]
    G = [[sum(r[i] * r[j] for r in M) for j in range(2)] for i in range(2)]      # M^T M
    h = [sum(r[i] * v for r, v in zip(M, rhs)) for i in range(2)]                # M^T b
    det = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    x = [(G[1][1] * h[0] - G[0][1] * h[1]) / det, (G[0][0] * h[1] - G[1][0] * h[0]) / det]
    res = [r[0] * x[0] + r[1] * x[1] - v for r, v in zip(M, rhs)]
    print("least-squares x+ =", [round(v, 4) for v in x])
    print("residuals =", [round(v, 4) for v in res], " sum of squares =", round(sum(v * v for v in res), 4))
    print("M^T residual =", [round(abs(sum(r[i] * v for r, v in zip(M, res))), 10) for i in range(2)])
  `, r`
    A+ = [0.2, 0.4]  x+ = [0.8, 1.6]  |x+| = 1.7889
       another solution [4.0, 0.0]: A x = 4.0, norm = 4.0000
       another solution [0.0, 2.0]: A x = 4.0, norm = 2.0000
       another solution [2.0, 1.0]: A x = 4.0, norm = 2.2361
    least-squares x+ = [1.76, 1.29]
    residuals = [0.05, -0.03, -0.04]  sum of squares = 0.005
    M^T residual = [0.0, 0.0]
  `, "For the wide system the pseudoinverse Aᵀ(AAᵀ)⁻¹ returns the exact solution of smallest norm; every other solution listed is longer. For the tall system (MᵀM)⁻¹Mᵀ returns the point whose residual is orthogonal to the columns, the least-squares solution."),
};

export const calculusExtraCode: Record<string, CodeExample> = {
  "matrix-calculus": py(r`
    A = [[2.0, 1.0], [0.0, 1.0]]            # deliberately not symmetric
    b = [1.0, -1.0]
    x = [1.0, 2.0]

    def matvec(M, v):
        return [sum(m * vi for m, vi in zip(row, v)) for row in M]

    def numeric_grad(f, x, h=1e-6):          # central differences, one coordinate at a time
        g = []
        for i in range(len(x)):
            xp, xm = x[:], x[:]
            xp[i] += h
            xm[i] -= h
            g.append((f(xp) - f(xm)) / (2 * h))
        return g

    At = [[A[j][i] for j in range(2)] for i in range(2)]
    show = lambda v: [round(t, 5) for t in v]

    linear = lambda v: sum(bi * vi for bi, vi in zip(b, v))
    print("a.x          formula:", show(b), "            numeric:", show(numeric_grad(linear, x)))

    quad = lambda v: sum(vi * wi for vi, wi in zip(v, matvec(A, v)))
    sym = [[A[i][j] + At[i][j] for j in range(2)] for i in range(2)]
    print("x^T A x      formula:", show(matvec(sym, x)), "             numeric:", show(numeric_grad(quad, x)))
    print("             the shortcut 2 A x would give", show([2 * t for t in matvec(A, x)]))

    lsq = lambda v: sum((r - bi) ** 2 for r, bi in zip(matvec(A, v), b))
    resid = [r - bi for r, bi in zip(matvec(A, x), b)]
    print("|A x - b|^2  formula:", show([2 * t for t in matvec(At, resid)]), "           numeric:", show(numeric_grad(lsq, x)))
  `, r`
    a.x          formula: [1.0, -1.0]             numeric: [1.0, -1.0]
    x^T A x      formula: [6.0, 5.0]              numeric: [6.0, 5.0]
                 the shortcut 2 A x would give [8.0, 4.0]
    |A x - b|^2  formula: [12.0, 12.0]            numeric: [12.0, 12.0]
  `, "Each identity is evaluated and compared with a finite-difference estimate that knows nothing about the formula. For the non-symmetric A the correct gradient of xᵀAx is (A + Aᵀ)x; the familiar 2Ax gives a different, wrong vector."),
};

export const statisticsExtraCode: Record<string, CodeExample> = {
  "joint-marginal-distributions": py(r`
    import math

    # joint probabilities P(X = x, Y = y); rows are x = 0, 1 and columns y = 0, 1
    joint = [[0.1, 0.3],
             [0.2, 0.4]]

    px = [sum(row) for row in joint]                                 # sum over y
    py = [sum(joint[x][y] for x in range(2)) for y in range(2)]      # sum over x
    print("P(X):", [round(v, 4) for v in px])
    print("P(Y):", [round(v, 4) for v in py])
    print("total:", round(sum(px), 10))

    # conditional distribution of X given each value of Y: a column, renormalised
    for y in range(2):
        cond = [joint[x][y] / py[y] for x in range(2)]
        print(f"P(X | Y = {y}):", [round(v, 4) for v in cond])

    # independence would require P(x, y) = P(x) P(y) in every cell
    gap = max(abs(joint[x][y] - px[x] * py[y]) for x in range(2) for y in range(2))
    print("largest |P(x,y) - P(x)P(y)|:", round(gap, 4), "->", "independent" if gap < 1e-12 else "dependent")

    mi = sum(joint[x][y] * math.log2(joint[x][y] / (px[x] * py[y])) for x in range(2) for y in range(2))
    print(f"mutual information: {mi:.5f} bits")

    # Bayes' theorem recovers P(Y | X) from P(X | Y) and P(Y)
    x = 1
    post = [joint[x][y] / py[y] * py[y] / px[x] for y in range(2)]
    print(f"P(Y | X = {x}):", [round(v, 4) for v in post])
  `, r`
    P(X): [0.4, 0.6]
    P(Y): [0.3, 0.7]
    total: 1.0
    P(X | Y = 0): [0.3333, 0.6667]
    P(X | Y = 1): [0.4286, 0.5714]
    largest |P(x,y) - P(x)P(y)|: 0.02 -> dependent
    mutual information: 0.00580 bits
    P(Y | X = 1): [0.3333, 0.6667]
  `, "Marginals are row and column sums of the joint table; conditionals are single columns rescaled to sum to one. The cells differ from the products of the marginals, so the variables are dependent, though only weakly — the mutual information is small."),
};

export const optimizationExtraCode: Record<string, CodeExample> = {
  "convex-sets": py(r`
    import math

    def in_disc(p):                     # L2 ball of radius 2
        return p[0] ** 2 + p[1] ** 2 <= 4

    def in_ring(p):                     # between radius 1 and radius 2
        return 1 <= p[0] ** 2 + p[1] ** 2 <= 4

    def in_diamond(p):                  # L1 ball of radius 2
        return abs(p[0]) + abs(p[1]) <= 2

    def lcg(seed):
        while True:
            seed = (seed * 1103515245 + 12345) % 2 ** 31
            yield seed / 2 ** 31

    def escaping_fraction(inside, pairs=2000):
        rng = lcg(3)
        pts = []
        while len(pts) < 400:           # sample points of the set by rejection
            p = (4 * next(rng) - 2, 4 * next(rng) - 2)
            if inside(p):
                pts.append(p)
        bad = 0
        for _ in range(pairs):
            a = pts[int(next(rng) * len(pts))]
            b = pts[int(next(rng) * len(pts))]
            if any(not inside((t * a[0] + (1 - t) * b[0], t * a[1] + (1 - t) * b[1])) for t in (i / 20 for i in range(1, 20))):
                bad += 1
        return bad / pairs

    for name, inside in (("disc", in_disc), ("diamond", in_diamond), ("ring", in_ring)):
        print(f"{name:8s}: {escaping_fraction(inside):.1%} of random segments leave the set")

    a, b = (1.5, 0.0), (-1.5, 0.0)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    print("ring: a and b inside:", in_ring(a), in_ring(b), "; midpoint", mid, "inside:", in_ring(mid))
    both = lambda p: in_disc(p) and p[0] + p[1] <= 1
    print(f"disc intersected with a half-plane: {escaping_fraction(both):.1%} of segments leave")
  `, r`
    disc    : 0.0% of random segments leave the set
    diamond : 0.0% of random segments leave the set
    ring    : 47.9% of random segments leave the set
    ring: a and b inside: True True ; midpoint (0.0, 0.0) inside: False
    disc intersected with a half-plane: 0.0% of segments leave
  `, "Random pairs of points are joined and the segment is tested at intermediate positions. No segment ever leaves the disc, the diamond or their intersection with a half-plane; for the ring a substantial fraction do, and one explicit counter-example is enough to prove it is not convex."),
};
