import type { CodeExample } from "@/types/curriculum";
import { py, r } from "./helpers";

/* Generated from verified runs: the output of every program below was captured from an actual execution
   (python -I, standard library only) and is re-checked by tests/code-run.test.ts. */

/** Python reference implementations for the IITM MLT week lessons (keys are topic slugs). */
export const mltCode: Record<string, CodeExample> = {
  "mlt-pca-centering-covariance": py(r`
    # four data points in the plane (each inner list is one point)
    X = [[3, 3], [2, 4], [-1, 1], [0, 0]]
    n, d = len(X), len(X[0])

    mu = [sum(x[j] for x in X) / n for j in range(d)]            # mean of each feature
    Xc = [[x[j] - mu[j] for j in range(d)] for x in X]           # centred points

    # C = (1/n) * sum of outer products x x^T of the centred points
    C = [[sum(x[a] * x[b] for x in Xc) / n for b in range(d)] for a in range(d)]

    print("mean:", mu)
    print("centred:", Xc)
    print("C =", C)

    # variance along the unit direction w equals w^T C w
    w = [2 ** -0.5, 2 ** -0.5]
    proj = [x[0] * w[0] + x[1] * w[1] for x in Xc]
    var_direct = sum(p * p for p in proj) / n
    var_formula = sum(w[a] * C[a][b] * w[b] for a in range(d) for b in range(d))
    print(f"variance along w: {var_direct:.4f} (direct), {var_formula:.4f} (w^T C w)")
  `, r`
    mean: [1.0, 2.0]
    centred: [[2.0, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]
    C = [[2.5, 2.0], [2.0, 2.5]]
    variance along w: 4.5000 (direct), 4.5000 (w^T C w)
  `, "The mean is subtracted from every point, then C is built as an average of outer products. The last two numbers agree because the variance of the projections xᵢᵀw is exactly the quadratic form wᵀCw."),
  "mlt-pca-reconstruction-error": py(r`
    import math

    Xc = [[2, 1], [1, 2], [-2, -1], [-1, -2]]       # centred data
    n = len(Xc)
    total = sum(x[0] ** 2 + x[1] ** 2 for x in Xc) / n   # (1/n) sum ||x||^2 = tr C

    print("angle  variance  error   sum")
    best = None
    for deg in range(0, 180, 15):
        t = math.radians(deg)
        w = (math.cos(t), math.sin(t))
        var = err = 0.0
        for x in Xc:
            z = x[0] * w[0] + x[1] * w[1]                 # coordinate along w
            rec = (z * w[0], z * w[1])                    # reconstruction (x^T w) w
            var += z * z / n
            err += ((x[0] - rec[0]) ** 2 + (x[1] - rec[1]) ** 2) / n
        print(f"{deg:5d}  {var:8.4f}  {err:6.4f}  {var + err:.4f}")
        if best is None or err < best[1]:
            best = (deg, err, var)

    print(f"smallest error {best[1]:.4f} at {best[0]} degrees, where the variance {best[2]:.4f} is largest")
    print(f"tr C = {total:.4f}")
  `, r`
    angle  variance  error   sum
        0    2.5000  2.5000  5.0000
       15    3.5000  1.5000  5.0000
       30    4.2321  0.7679  5.0000
       45    4.5000  0.5000  5.0000
       60    4.2321  0.7679  5.0000
       75    3.5000  1.5000  5.0000
       90    2.5000  2.5000  5.0000
      105    1.5000  3.5000  5.0000
      120    0.7679  4.2321  5.0000
      135    0.5000  4.5000  5.0000
      150    0.7679  4.2321  5.0000
      165    1.5000  3.5000  5.0000
    smallest error 0.5000 at 45 degrees, where the variance 4.5000 is largest
    tr C = 5.0000
  `, "Every direction is tried in turn. Error and variance are computed separately from the points, yet their sum is always tr C = 5, so the direction with the least error (45°, the eigenvector (1, 1)/√2) is the one with the most variance."),
  "mlt-pca-principal-components": py(r`
    import math

    Xc = [[2, 1], [1, 2], [-2, -1], [-1, -2]]
    n = len(Xc)
    C = [[sum(x[a] * x[b] for x in Xc) / n for b in range(2)] for a in range(2)]

    # eigen-decomposition of a symmetric 2 x 2 matrix [[a, b], [b, d]]
    a, b, d = C[0][0], C[0][1], C[1][1]
    half, disc = (a + d) / 2, math.sqrt(((a - d) / 2) ** 2 + b * b)
    lam = [half + disc, half - disc]
    v1 = [lam[0] - d, b]
    norm = math.hypot(*v1)
    w1 = [v1[0] / norm, v1[1] / norm]
    w2 = [-w1[1], w1[0]]                              # orthogonal to w1
    print(f"eigenvalues: {lam[0]:.4f}, {lam[1]:.4f}")
    print(f"w1 = ({w1[0]:.4f}, {w1[1]:.4f}), w2 = ({w2[0]:.4f}, {w2[1]:.4f})")

    err = 0.0
    for x in Xc:
        c1 = x[0] * w1[0] + x[1] * w1[1]              # coordinates in the principal basis
        c2 = x[0] * w2[0] + x[1] * w2[1]
        rec = [c1 * w1[0], c1 * w1[1]]                # keep only the first coordinate (k = 1)
        e = (x[0] - rec[0]) ** 2 + (x[1] - rec[1]) ** 2
        err += e / n
        print(f"x = {x}: c = ({c1:+.4f}, {c2:+.4f}), reconstruction = ({rec[0]:+.2f}, {rec[1]:+.2f}), error = {e:.4f}")
    print(f"mean reconstruction error = {err:.4f} = lambda_2")
  `, r`
    eigenvalues: 4.5000, 0.5000
    w1 = (0.7071, 0.7071), w2 = (-0.7071, 0.7071)
    x = [2, 1]: c = (+2.1213, -0.7071), reconstruction = (+1.50, +1.50), error = 0.5000
    x = [1, 2]: c = (+2.1213, +0.7071), reconstruction = (+1.50, +1.50), error = 0.5000
    x = [-2, -1]: c = (-2.1213, +0.7071), reconstruction = (-1.50, -1.50), error = 0.5000
    x = [-1, -2]: c = (-2.1213, -0.7071), reconstruction = (-1.50, -1.50), error = 0.5000
    mean reconstruction error = 0.5000 = lambda_2
  `, "The covariance matrix is diagonalised in closed form. Each point's coordinates in the principal basis are dot products with w₁ and w₂; dropping the second coordinate leaves an error of c₂², whose average is the discarded eigenvalue λ₂ = 0.5."),
  "mlt-pca-choosing-k": py(r`
    lam = [4.2, 2.1, 0.9, 0.5, 0.2, 0.1]       # eigenvalues of C, largest first
    total = sum(lam)
    threshold = 0.95

    k_star, running = None, 0.0
    for k, l in enumerate(lam, start=1):
        running += l
        frac = running / total
        mark = ""
        if k_star is None and frac >= threshold:
            k_star, mark = k, "  <- smallest k reaching 95%"
        print(f"k = {k}: variance retained = {frac:.4f}, reconstruction error = {abs(total - running):.2f}{mark}")

    d, n = len(lam), 1000
    print(f"storage: d*n = {d * n}, k*(d+n) = {k_star * (d + n)}")
  `, r`
    k = 1: variance retained = 0.5250, reconstruction error = 3.80
    k = 2: variance retained = 0.7875, reconstruction error = 1.70
    k = 3: variance retained = 0.9000, reconstruction error = 0.80
    k = 4: variance retained = 0.9625, reconstruction error = 0.30  <- smallest k reaching 95%
    k = 5: variance retained = 0.9875, reconstruction error = 0.10
    k = 6: variance retained = 1.0000, reconstruction error = 0.00
    storage: d*n = 6000, k*(d+n) = 4024
  `, "The cumulative sum of the sorted eigenvalues gives the variance retained for every k, and the remainder is the reconstruction error. The first k to reach the threshold is selected."),
  "mlt-kpca-limitations": py(r`
    import math

    # 12 points evenly spaced on a circle of radius 2 (already centred)
    P = [(2 * math.cos(2 * math.pi * i / 12), 2 * math.sin(2 * math.pi * i / 12)) for i in range(12)]
    n = len(P)
    C = [[sum(p[a] * p[b] for p in P) / n for b in range(2)] for a in range(2)]
    print(f"C = [[{C[0][0]:.3f}, {C[0][1]:.3f}], [{C[1][0]:.3f}, {C[1][1]:.3f}]]")

    half = (C[0][0] + C[1][1]) / 2
    disc = math.sqrt(((C[0][0] - C[1][1]) / 2) ** 2 + C[0][1] ** 2)
    l1, l2 = half + disc, half - disc
    print(f"eigenvalues: {l1:.3f}, {l2:.3f}")
    print(f"variance lost with one component: {l2 / (l1 + l2):.1%}")

    # cost of the two eigendecompositions for image-like data
    d, n_img = 10_000, 200
    print(f"d^3 = {d ** 3:.1e}, n^3 = {n_img ** 3:.1e}, ratio = {d ** 3 / n_img ** 3:,.0f}")
  `, r`
    C = [[2.000, 0.000], [0.000, 2.000]]
    eigenvalues: 2.000, 2.000
    variance lost with one component: 50.0%
    d^3 = 1.0e+12, n^3 = 8.0e+06, ratio = 125,000
  `, "Points on a circle have equal variance in every direction, so C is a multiple of the identity and any single principal component loses half of the variance. The last line compares the cubic costs of decomposing the d × d and the n × n matrix."),
  "mlt-kpca-gram-duality": py(r`
    import math

    X = [[2, 1], [1, 2], [1, 1]]                     # d = 3 rows (features), n = 2 columns (points)
    d, n = len(X), len(X[0])

    K = [[sum(X[r][i] * X[r][j] for r in range(d)) for j in range(n)] for i in range(n)]      # X^T X
    G = [[sum(X[a][i] * X[b][i] for i in range(n)) for b in range(d)] for a in range(d)]      # X X^T
    print("X^T X =", K)
    print("X X^T =", G)

    # eigenpairs of the small symmetric 2 x 2 matrix K
    a, b, c = K[0][0], K[0][1], K[1][1]
    half, disc = (a + c) / 2, math.sqrt(((a - c) / 2) ** 2 + b * b)
    for mu in (half + disc, half - disc):
        beta = [b, mu - a]
        nb = math.hypot(*beta)
        beta = [v / nb for v in beta]
        Xb = [sum(X[r][i] * beta[i] for i in range(n)) for r in range(d)]
        w = [v / math.sqrt(mu) for v in Xb]                       # w = X beta / sqrt(mu)
        Gw = [sum(G[r][s] * w[s] for s in range(d)) for r in range(d)]
        resid = max(abs(Gw[r] - mu * w[r]) for r in range(d))
        print(f"mu = {mu:.0f}: w = ({w[0]:+.4f}, {w[1]:+.4f}, {w[2]:+.4f}), |w| = {math.hypot(*w):.4f}, max|XX^T w - mu w| = {resid:.1e}")
    print("trace of both matrices:", sum(K[i][i] for i in range(n)), sum(G[i][i] for i in range(d)))
  `, r`
    X^T X = [[6, 5], [5, 6]]
    X X^T = [[5, 4, 3], [4, 5, 3], [3, 3, 2]]
    mu = 11: w = (+0.6396, +0.6396, +0.4264), |w| = 1.0000, max|XX^T w - mu w| = 0.0e+00
    mu = 1: w = (+0.7071, -0.7071, +0.0000), |w| = 1.0000, max|XX^T w - mu w| = 0.0e+00
    trace of both matrices: 12 12
  `, "Eigenvectors are computed only for the 2 × 2 matrix XᵀX. Multiplying by X and dividing by √μ turns each into a unit eigenvector of the 3 × 3 matrix XXᵀ with the same eigenvalue, as the residual confirms."),
  "mlt-kpca-kernels": py(r`
    import math

    def phi(x):                     # explicit feature map of the kernel (1 + x.y)^2 in two dimensions
        s = math.sqrt(2)
        return [1, s * x[0], s * x[1], x[0] ** 2, x[1] ** 2, s * x[0] * x[1]]

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    x, y = (1, 2), (3, -1)
    print("x.y =", dot(x, y))
    print("kernel (1 + x.y)^2  =", (1 + dot(x, y)) ** 2)
    print("phi(x).phi(y)       =", round(dot(phi(x), phi(y)), 10))

    # feature-space dimension of the polynomial kernel: C(d + p, p)
    for d, p in [(2, 2), (2, 5), (100, 2), (100, 5)]:
        print(f"d = {d}, p = {p}: {math.comb(d + p, p):,} features")

    sigma = 1.5
    dist2 = sum((a - b) ** 2 for a, b in zip(x, y))
    print(f"Gaussian kernel, sigma = {sigma}: {math.exp(-dist2 / (2 * sigma ** 2)):.5f}")
  `, r`
    x.y = 1
    kernel (1 + x.y)^2  = 4
    phi(x).phi(y)       = 4.0
    d = 2, p = 2: 6 features
    d = 2, p = 5: 21 features
    d = 100, p = 2: 5,151 features
    d = 100, p = 5: 96,560,646 features
    Gaussian kernel, sigma = 1.5: 0.05564
  `, "The kernel value needs one inner product in ℝ²; the explicit route builds two six-dimensional vectors and multiplies them, giving the same number. The table shows how quickly the explicit dimension grows, which is why the map is never formed."),
  "mlt-kpca-algorithm": py(r`
    import math

    x = [1.0, 2.0, 6.0]                               # three one-dimensional points
    n = len(x)
    K = [[a * b for b in x] for a in x]               # linear kernel

    row = [sum(r) / n for r in K]
    grand = sum(row) / n
    Kc = [[K[i][j] - row[i] - row[j] + grand for j in range(n)] for i in range(n)]
    for r in Kc:
        print(r)

    # top eigenvector of Kc by power iteration
    b = [1.0, 0.5, -1.0]
    for _ in range(100):
        v = [sum(Kc[i][j] * b[j] for j in range(n)) for i in range(n)]
        nv = math.sqrt(sum(t * t for t in v))
        b = [t / nv for t in v]
    mu = sum(b[i] * sum(Kc[i][j] * b[j] for j in range(n)) for i in range(n))    # eigenvalue n * lambda
    proj = [math.sqrt(mu) * t for t in b]                                        # coordinates of the points
    if proj[2] < 0:
        proj = [-p for p in proj]
    print(f"n*lambda = {mu:.4f}, lambda = {mu / n:.4f}")
    print("projections:", [round(p, 4) for p in proj])
    print("centred x:  ", [xi - sum(x) / n for xi in x])
  `, r`
    [4.0, 2.0, -6.0]
    [2.0, 1.0, -3.0]
    [-6.0, -3.0, 9.0]
    n*lambda = 14.0000, lambda = 4.6667
    projections: [-2.0, -1.0, 3.0]
    centred x:   [-2.0, -1.0, 3.0]
  `, "The kernel matrix is centred using only its own row means and grand mean. With the linear kernel the leading kernel principal coordinates reproduce the centred data, confirming that kernel PCA reduces to ordinary PCA in that case."),
  "mlt-kmeans-objective": py(r`
    from itertools import product

    x = [1, 2, 9, 10]
    k = 2

    def objective(z):
        F = 0.0
        for j in range(k):
            members = [xi for xi, zi in zip(x, z) if zi == j]
            if members:
                mu = sum(members) / len(members)          # the mean is the best centre
                F += sum((xi - mu) ** 2 for xi in members)
        return F

    print("F(0,0,1,1) =", objective((0, 0, 1, 1)))
    print("F(0,1,0,1) =", objective((0, 1, 0, 1)))

    # brute force over all k^n assignments (only possible for tiny n)
    best = min(product(range(k), repeat=len(x)), key=objective)
    print("assignments tried:", k ** len(x))
    print("best:", best, "with F =", objective(best))
  `, r`
    F(0,0,1,1) = 1.0
    F(0,1,0,1) = 64.0
    assignments tried: 16
    best: (0, 0, 1, 1) with F = 1.0
  `, "The objective adds, for each cluster, the squared distances of its points to their mean. Enumerating all 2⁴ assignments finds the optimum here, but the count kⁿ makes this impossible for realistic n."),
  "mlt-kmeans-lloyd": py(r`
    x = [0, 2, 4, 10]
    mu = [0.0, 2.0]                       # initial means

    def cost(z, mu):
        return sum((xi - mu[zi]) ** 2 for xi, zi in zip(x, z))

    z = None
    for it in range(1, 10):
        # reassignment: nearest mean; on a tie the point keeps its current cluster
        new = []
        for i, xi in enumerate(x):
            d = [(xi - m) ** 2 for m in mu]
            j = d.index(min(d))
            if z is not None and d[z[i]] == d[j]:
                j = z[i]
            new.append(j)
        if new == z:
            print(f"iteration {it}: no point moved -> converged")
            break
        z = new
        print(f"iteration {it}: assign {z}  F = {cost(z, mu):.3f}")
        # mean computation
        mu = [sum(xi for xi, zi in zip(x, z) if zi == j) / z.count(j) for j in range(2)]
        print(f"             means  [{mu[0]:.3f}, {mu[1]:.3f}]  F = {cost(z, mu):.3f}")
  `, r`
    iteration 1: assign [0, 1, 1, 1]  F = 68.000
                 means  [0.000, 5.333]  F = 34.667
    iteration 2: assign [0, 0, 1, 1]  F = 27.556
                 means  [1.000, 7.000]  F = 20.000
    iteration 3: no point moved -> converged
  `, "Each iteration reassigns points to the nearest mean and then recomputes the means. The printed objective goes 68 → 34.667 → 27.556 → 20 and never increases, which is the heart of the convergence proof."),
  "mlt-kmeans-geometry": py(r`
    mu1, mu2 = (0.0, 0.0), (4.0, 2.0)

    # boundary: 2 (mu2 - mu1) . x = |mu2|^2 - |mu1|^2
    w = (2 * (mu2[0] - mu1[0]), 2 * (mu2[1] - mu1[1]))
    c = (mu2[0] ** 2 + mu2[1] ** 2) - (mu1[0] ** 2 + mu1[1] ** 2)
    print(f"boundary: {w[0]:.0f} x1 + {w[1]:.0f} x2 = {c:.0f}")

    mid = ((mu1[0] + mu2[0]) / 2, (mu1[1] + mu2[1]) / 2)
    print("midpoint on boundary:", w[0] * mid[0] + w[1] * mid[1] == c)

    for p in [(1, 1), (3, 0), (2, 1), (0, 5)]:
        side = w[0] * p[0] + w[1] * p[1] - c           # < 0: closer to mu1
        d1 = (p[0] - mu1[0]) ** 2 + (p[1] - mu1[1]) ** 2
        d2 = (p[0] - mu2[0]) ** 2 + (p[1] - mu2[1]) ** 2
        cluster = 1 if side < 0 else 2 if side > 0 else "tie"
        print(f"point {p}: w.x - c = {side:+.0f}, squared distances {d1:.0f} and {d2:.0f} -> cluster {cluster}")
  `, r`
    boundary: 8 x1 + 4 x2 = 20
    midpoint on boundary: True
    point (1, 1): w.x - c = -8, squared distances 2 and 10 -> cluster 1
    point (3, 0): w.x - c = +4, squared distances 9 and 5 -> cluster 2
    point (2, 1): w.x - c = +0, squared distances 5 and 5 -> cluster tie
    point (0, 5): w.x - c = +0, squared distances 25 and 25 -> cluster tie
  `, "The sign of a single linear expression decides which mean is closer, and it agrees with comparing the two squared distances directly. That linear test is why each k-means region is an intersection of half-spaces."),
  "mlt-kmeans-plus-plus": py(r`
    x = [0, 1, 9, 10]
    first = 0                                   # index of the first centre (chosen uniformly)

    D2 = [(xi - x[first]) ** 2 for xi in x]     # squared distance to the nearest chosen centre
    total = sum(D2)
    probs = [d / total for d in D2]
    for xi, d, p in zip(x, D2, probs):
        print(f"x = {xi:2d}: D^2 = {d:3d}, probability = {p:.4f}")

    good = probs[2] + probs[3]                  # second centre lands in the other cluster
    print(f"P(second centre in the far cluster) = {good:.4f}")
    print(f"with a uniform choice among the other points: {2 / 3:.4f}")

    # expected objective after seeding (each cluster keeps the centre that was drawn)
    def cost(centres):
        return sum(min((xi - c) ** 2 for c in centres) for xi in x)
    expected = sum(p * cost([x[first], xi]) for xi, p in zip(x, probs))
    print(f"expected F after k-means++ seeding = {expected:.3f}, optimum = 1.0")
  `, r`
    x =  0: D^2 =   0, probability = 0.0000
    x =  1: D^2 =   1, probability = 0.0055
    x =  9: D^2 =  81, probability = 0.4451
    x = 10: D^2 = 100, probability = 0.5495
    P(second centre in the far cluster) = 0.9945
    with a uniform choice among the other points: 0.6667
    expected F after k-means++ seeding = 2.786, optimum = 1.0
  `, "Points are sampled in proportion to D², so the second centre almost surely falls in the cluster that has none yet. The expected objective straight after seeding is already close to the optimum of 1."),
  "mlt-mle": py(r`
    import math

    # Bernoulli: ten coin tosses
    tosses = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    n, nh = len(tosses), sum(tosses)

    def loglik(p):
        return nh * math.log(p) + (n - nh) * math.log(1 - p)

    p_hat = nh / n
    print(f"p_hat = {p_hat}")
    for p in (0.5, 0.6, 0.7, 0.8):
        print(f"  log-likelihood at p = {p}: {loglik(p):.4f}")

    # Gaussian: four measurements
    xs = [2, 4, 4, 6]
    m = len(xs)
    mu = sum(xs) / m
    var = sum((v - mu) ** 2 for v in xs) / m            # divides by n: the ML estimate

    def gauss_loglik(mu, var):
        return -0.5 * m * math.log(2 * math.pi * var) - sum((v - mu) ** 2 for v in xs) / (2 * var)

    print(f"mu_hat = {mu}, sigma2_hat = {var}, unbiased variance = {var * m / (m - 1):.4f}")
    print(f"  log-likelihood at the estimate: {gauss_loglik(mu, var):.4f}")
    print(f"  log-likelihood at (3.5, 2.0):   {gauss_loglik(3.5, 2.0):.4f}")
  `, r`
    p_hat = 0.7
      log-likelihood at p = 0.5: -6.9315
      log-likelihood at p = 0.6: -6.3247
      log-likelihood at p = 0.7: -6.1086
      log-likelihood at p = 0.8: -6.3903
    mu_hat = 4.0, sigma2_hat = 2.0, unbiased variance = 2.6667
      log-likelihood at the estimate: -7.0620
      log-likelihood at (3.5, 2.0):   -7.3120
  `, "The Bernoulli estimate is the fraction of ones and the Gaussian estimates are the sample mean and the 1/n variance. In both cases every other parameter value tried gives a smaller log-likelihood."),
  "mlt-bayesian-estimation": py(r`
    alpha, beta = 2, 2                     # prior Beta(2, 2): mild belief in a fair coin
    tosses = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]

    print(" n  heads  posterior      mean    MAP     MLE")
    nh = 0
    for n, t in enumerate(tosses, start=1):
        nh += t
        a, b = alpha + nh, beta + (n - nh)          # posterior Beta(a, b)
        mean = a / (a + b)
        mode = (a - 1) / (a + b - 2)
        print(f"{n:2d}  {nh:5d}  Beta({a:2d}, {b:2d})  {mean:.4f}  {mode:.4f}  {nh / n:.4f}")

    # the posterior mean is a weighted average of the prior mean and the data frequency
    n = len(tosses)
    w_prior = (alpha + beta) / (alpha + beta + n)
    print(f"check: {w_prior:.4f} * 0.5 + {1 - w_prior:.4f} * {nh / n} = {w_prior * 0.5 + (1 - w_prior) * nh / n:.4f}")
  `, r`
     n  heads  posterior      mean    MAP     MLE
     1      1  Beta( 3,  2)  0.6000  0.6667  1.0000
     2      1  Beta( 3,  3)  0.5000  0.5000  0.5000
     3      2  Beta( 4,  3)  0.5714  0.6000  0.6667
     4      3  Beta( 5,  3)  0.6250  0.6667  0.7500
     5      3  Beta( 5,  4)  0.5556  0.5714  0.6000
     6      4  Beta( 6,  4)  0.6000  0.6250  0.6667
     7      5  Beta( 7,  4)  0.6364  0.6667  0.7143
     8      6  Beta( 8,  4)  0.6667  0.7000  0.7500
     9      6  Beta( 8,  5)  0.6154  0.6364  0.6667
    10      7  Beta( 9,  5)  0.6429  0.6667  0.7000
    check: 0.2857 * 0.5 + 0.7143 * 0.7 = 0.6429
  `, "Each toss adds one to α (heads) or β (tails). The posterior mean and MAP start near the prior mean 0.5 and move towards the maximum-likelihood estimate as data accumulate; after one toss the MLE is already 1.0 while the Bayesian estimates stay moderate."),
  "mlt-gmm": py(r`
    import math

    def normal_pdf(x, mu, sigma):
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    pi = [0.5, 0.5]
    mu = [0.0, 4.0]
    sigma = [1.0, 1.0]

    def density(x):
        return sum(p * normal_pdf(x, m, s) for p, m, s in zip(pi, mu, sigma))

    def responsibilities(x):
        parts = [p * normal_pdf(x, m, s) for p, m, s in zip(pi, mu, sigma)]
        total = sum(parts)
        return [v / total for v in parts]

    for x in (1.0, 2.0, 3.5):
        lam = responsibilities(x)
        print(f"x = {x}: f(x) = {density(x):.5f}, responsibilities = ({lam[0]:.4f}, {lam[1]:.4f})")

    data = [-0.3, 0.8, 1.1, 3.2, 4.4, 5.0]
    ll = sum(math.log(density(x)) for x in data)      # log of a sum for every point
    print(f"log-likelihood of the sample: {ll:.4f}")
    print("free parameters for K = 2:", 3 * 2 - 1)
  `, r`
    x = 1.0: f(x) = 0.12320, responsibilities = (0.9820, 0.0180)
    x = 2.0: f(x) = 0.05399, responsibilities = (0.5000, 0.5000)
    x = 3.5: f(x) = 0.17647, responsibilities = (0.0025, 0.9975)
    log-likelihood of the sample: -11.4990
    free parameters for K = 2: 5
  `, "The mixture density is a weighted sum of component densities, and Bayes' rule converts the weighted components at a point into responsibilities. At x = 2, midway between the means, the two components are equally responsible."),
  "mlt-em-algorithm": py(r`
    import math

    def pdf(x, mu, var):
        return math.exp(-(x - mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)

    x = [1.0, 2.0, 6.0, 7.0]
    n, K = len(x), 2
    pi, mu, var = [0.5, 0.5], [2.0, 5.0], [1.0, 1.0]

    def loglik():
        return sum(math.log(sum(pi[k] * pdf(xi, mu[k], var[k]) for k in range(K))) for xi in x)

    print(f"start:       mu = ({mu[0]:.3f}, {mu[1]:.3f})  log-likelihood = {loglik():.4f}")
    for it in range(1, 6):
        # E-step: responsibilities under the current parameters
        R = []
        for xi in x:
            w = [pi[k] * pdf(xi, mu[k], var[k]) for k in range(K)]
            R.append([v / sum(w) for v in w])
        # M-step: weighted means, variances and mixing weights
        for k in range(K):
            nk = sum(r[k] for r in R)
            mu[k] = sum(r[k] * xi for r, xi in zip(R, x)) / nk
            var[k] = max(1e-6, sum(r[k] * (xi - mu[k]) ** 2 for r, xi in zip(R, x)) / nk)
            pi[k] = nk / n
        print(f"iteration {it}: mu = ({mu[0]:.3f}, {mu[1]:.3f})  log-likelihood = {loglik():.4f}")
    print(f"variances = ({var[0]:.3f}, {var[1]:.3f}), weights = ({pi[0]:.3f}, {pi[1]:.3f})")
  `, r`
    start:       mu = (2.000, 5.000)  log-likelihood = -9.4362
    iteration 1: mu = (1.499, 6.474)  log-likelihood = -5.7429
    iteration 2: mu = (1.500, 6.500)  log-likelihood = -5.6758
    iteration 3: mu = (1.500, 6.500)  log-likelihood = -5.6758
    iteration 4: mu = (1.500, 6.500)  log-likelihood = -5.6758
    iteration 5: mu = (1.500, 6.500)  log-likelihood = -5.6758
    variances = (0.250, 0.250), weights = (0.500, 0.500)
  `, "The E-step computes each point's responsibilities; the M-step re-estimates the parameters as responsibility-weighted averages. The log-likelihood rises at every iteration and levels off once the two components sit on the two groups of points."),
  "mlt-supervised-learning": py(r`
    import math

    # regression: real-valued labels, squared-error loss
    pred = [2.5, 0.0, 2.1, 7.8]
    y = [3.0, -0.5, 2.0, 7.0]
    sse = 0.5 * sum((p - t) ** 2 for p, t in zip(pred, y))
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y))
    print(f"regression:     SSE = {sse:.3f}, RMSE = {rmse:.3f}")

    # classification: labels in {-1, +1}, zero-one loss
    pred_c = [1, -1, -1, 1]
    y_c = [1, 1, -1, 1]
    mistakes = sum(p != t for p, t in zip(pred_c, y_c))
    print(f"classification: mistakes = {mistakes}, zero-one loss = {mistakes / len(y_c)}")

    # a linear model produces both kinds of prediction from the same score w.x
    w = [0.5, -1.0]
    for x in ([4, 1], [1, 2]):
        score = sum(wi * xi for wi, xi in zip(w, x))
        print(f"x = {x}: score = {score:+.1f}, regression output = {score:+.1f}, class = {'+1' if score >= 0 else '-1'}")
  `, r`
    regression:     SSE = 0.575, RMSE = 0.536
    classification: mistakes = 1, zero-one loss = 0.25
    x = [4, 1]: score = +1.0, regression output = +1.0, class = +1
    x = [1, 2]: score = -1.5, regression output = -1.5, class = -1
  `, "Both tasks compare predictions with labels; only the loss differs. A linear model's score wᵀx is used directly for regression and through its sign for classification."),
  "mlt-linear-regression": py(r`
    # data points (x, y); the first feature row is the constant 1 (the bias)
    pts = [(1, 1), (2, 2), (3, 2)]
    X = [[1, 1, 1], [p[0] for p in pts]]          # d x n, columns are data points
    y = [p[1] for p in pts]
    d, n = len(X), len(y)

    A = [[sum(X[a][i] * X[b][i] for i in range(n)) for b in range(d)] for a in range(d)]   # X X^T
    b = [sum(X[a][i] * y[i] for i in range(n)) for a in range(d)]                           # X y
    print("X X^T =", A, " X y =", b)

    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    w = [(A[1][1] * b[0] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - A[1][0] * b[0]) / det]
    print(f"w* = ({w[0]:.4f}, {w[1]:.4f})")

    pred = [w[0] * X[0][i] + w[1] * X[1][i] for i in range(n)]
    res = [pred[i] - y[i] for i in range(n)]
    print("predictions:", [round(p, 4) for p in pred])
    print(f"loss L(w*) = {0.5 * sum(r * r for r in res):.4f}")
    grad = [sum(res[i] * X[a][i] for i in range(n)) for a in range(d)]                      # X (X^T w - y)
    print("gradient at w*:", [round(abs(g), 10) for g in grad])
  `, r`
    X X^T = [[3, 6], [6, 14]]  X y = [5, 11]
    w* = (0.6667, 0.5000)
    predictions: [1.1667, 1.6667, 2.1667]
    loss L(w*) = 0.0833
    gradient at w*: [0.0, 0.0]
  `, "XXᵀ and Xy are accumulated from the data and the 2 × 2 normal equations are solved directly. The gradient X(Xᵀw − y) evaluated at the solution is zero, confirming that w* is the minimiser."),
  "mlt-gd-sgd": py(r`
    X = [[1, 1, 1], [1, 2, 3]]            # rows: constant feature, x
    y = [1, 2, 2]
    n = 3

    def loss(w):
        return 0.5 * sum((w[0] * X[0][i] + w[1] * X[1][i] - y[i]) ** 2 for i in range(n))

    def grad(w, idx):
        g = [0.0, 0.0]
        for i in idx:
            e = w[0] * X[0][i] + w[1] * X[1][i] - y[i]
            g[0] += e * X[0][i]
            g[1] += e * X[1][i]
        return g

    eta = 0.05
    w = [0.0, 0.0]
    print(f"GD  step    0: loss = {loss(w):.4f}")
    for t in range(1, 2001):
        g = grad(w, range(n))                       # full gradient: all n points
        w = [w[0] - eta * g[0], w[1] - eta * g[1]]
        if t in (1, 10, 100, 2000):
            print(f"GD  step {t:4d}: w = ({w[0]:.4f}, {w[1]:.4f}), loss = {loss(w):.4f}")

    # SGD: one point per step, visited in a fixed pseudo-random order, smaller step
    w, seed, avg = [0.0, 0.0], 1, [0.0, 0.0]
    T = 4000
    for t in range(1, T + 1):
        seed = (seed * 1103515245 + 12345) % 2 ** 31
        i = seed % n
        g = grad(w, [i])
        w = [w[0] - 0.02 * n * g[0], w[1] - 0.02 * n * g[1]]
        if t > T // 2:
            avg = [avg[0] + w[0] / (T // 2), avg[1] + w[1] / (T // 2)]
    print(f"SGD last iterate:   w = ({w[0]:.4f}, {w[1]:.4f}), loss = {loss(w):.4f}")
    print(f"SGD averaged:       w = ({avg[0]:.4f}, {avg[1]:.4f}), loss = {loss(avg):.4f}")
    print("exact solution:     w = (0.6667, 0.5000), loss = 0.0833")
  `, r`
    GD  step    0: loss = 4.5000
    GD  step    1: w = (0.2500, 0.5500), loss = 0.2362
    GD  step   10: w = (0.3546, 0.6373), loss = 0.1043
    GD  step  100: w = (0.6060, 0.5267), loss = 0.0841
    GD  step 2000: w = (0.6667, 0.5000), loss = 0.0833
    SGD last iterate:   w = (0.6764, 0.4921), loss = 0.0835
    SGD averaged:       w = (0.6797, 0.4944), loss = 0.0834
    exact solution:     w = (0.6667, 0.5000), loss = 0.0833
  `, "Gradient descent uses all three points per step and converges to the normal-equation solution, slowly at the end because the loss surface is elongated. SGD uses one point per step and keeps fluctuating, but the average of its later iterates is close to the optimum."),
  "mlt-regression-views": py(r`
    import math

    # geometric view: n = 2 data points, d = 1 feature
    a = [1.0, 2.0]                 # the feature row of X
    y = [2.0, 3.0]
    w = sum(ai * yi for ai, yi in zip(a, y)) / sum(ai * ai for ai in a)
    yhat = [w * ai for ai in a]
    res = [yi - yh for yi, yh in zip(y, yhat)]
    print(f"w* = {w}, projection = {yhat}, residual = ({res[0]:.1f}, {res[1]:.1f})")
    print(f"residual . feature row = {sum(r * ai for r, ai in zip(res, a)):.10f}")

    # probabilistic view: maximise the Gaussian log-likelihood over a grid of w
    sigma = 0.7
    def sse(w):
        return sum((yi - w * ai) ** 2 for ai, yi in zip(a, y))
    def loglik(w):
        return -0.5 * len(y) * math.log(2 * math.pi * sigma ** 2) - sse(w) / (2 * sigma ** 2)

    grid = [i / 100 for i in range(100, 221)]
    w_ml = max(grid, key=loglik)
    w_ls = min(grid, key=sse)
    print(f"argmax log-likelihood = {w_ml}, argmin squared error = {w_ls}")
    print(f"ML estimate of the noise variance = {sse(w) / len(y):.2f}")
  `, r`
    w* = 1.6, projection = [1.6, 3.2], residual = (0.4, -0.2)
    residual . feature row = -0.0000000000
    argmax log-likelihood = 1.6, argmin squared error = 1.6
    ML estimate of the noise variance = 0.10
  `, "The fitted values are the projection of y onto the feature row, so the residual has zero inner product with it. Searching a grid of w shows that the log-likelihood under Gaussian noise peaks exactly where the squared error is smallest."),
  "mlt-kernel-regression": py(r`
    def k(a, b):                         # polynomial kernel of degree 2
        return (1 + a * b) ** 2

    x = [0.0, 1.0, 2.0]
    y = [1.0, 2.0, 5.0]                  # generated by y = 1 + x^2
    n = len(x)
    K = [[k(a, b) for b in x] for a in x]
    print("K =", K)

    # solve K alpha = y by Gaussian elimination
    M = [K[i][:] + [y[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    alpha = [0.0] * n
    for i in reversed(range(n)):
        alpha[i] = (M[i][n] - sum(M[i][j] * alpha[j] for j in range(i + 1, n))) / M[i][i]
    print("alpha =", [round(a, 6) for a in alpha])

    def predict(t):
        return sum(a * k(xi, t) for a, xi in zip(alpha, x))

    for t in (0.0, 1.5, 3.0):
        print(f"f({t}) = {predict(t):.4f}   (true value {1 + t * t:.4f})")
  `, r`
    K = [[1.0, 1.0, 1.0], [1.0, 4.0, 9.0], [1.0, 9.0, 25.0]]
    alpha = [1.5, -1.0, 0.5]
    f(0.0) = 1.0000   (true value 1.0000)
    f(1.5) = 3.2500   (true value 3.2500)
    f(3.0) = 10.0000   (true value 10.0000)
  `, "Learning solves the n × n system Kα = y; prediction is a kernel-weighted sum over the training points. Because the degree-2 feature space contains the true function, the fit is exact even outside the training range."),
  "mlt-overfitting": py(r`
    train = [(0, 1.0), (1, 3.0), (2, 2.0)]
    test = [(0.5, 1.8), (1.5, 2.6), (3.0, 2.9)]

    def mse(f, data):
        return sum((f(x) - y) ** 2 for x, y in data) / len(data)

    # degree 1: least-squares line through the three points
    n = len(train)
    sx, sy = sum(x for x, _ in train), sum(y for _, y in train)
    sxx, sxy = sum(x * x for x, _ in train), sum(x * y for x, y in train)
    b1 = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b0 = (sy - b1 * sx) / n
    line = lambda x: b0 + b1 * x

    # degree 2: three coefficients for three points -> exact interpolation
    a = train[0][1]                                  # f(0) = a
    c = (train[2][1] - 2 * train[1][1] + a) / 2      # from f(1) and f(2)
    b = train[1][1] - a - c
    quad = lambda x: a + b * x + c * x * x

    print(f"line:      f(x) = {b0:.3f} + {b1:.3f} x")
    print(f"quadratic: f(x) = {a:.1f} + {b:.1f} x + {c:.1f} x^2")
    for name, f in (("line", line), ("quadratic", quad)):
        print(f"{name:9s}: train MSE = {mse(f, train):.4f}, test MSE = {mse(f, test):.4f}, f(3) = {f(3):.2f}")
  `, r`
    line:      f(x) = 1.500 + 0.500 x
    quadratic: f(x) = 1.0 + 3.5 x + -1.5 x^2
    line     : train MSE = 0.5000, test MSE = 0.0450, f(3) = 3.00
    quadratic: train MSE = 0.0000, test MSE = 8.1388, f(3) = -2.00
  `, "The quadratic has as many coefficients as there are training points, so it fits them exactly, yet it is worse than the straight line on the held-out points and extrapolates wildly at x = 3. Zero training error is not evidence of a good model."),
  "mlt-ridge": py(r`
    import math

    A = [[3.0, 6.0], [6.0, 14.0]]      # X X^T for the points (1,1), (2,2), (3,2) with a bias feature
    b = [5.0, 11.0]                    # X y

    def solve(lam):
        a00, a11, a01 = A[0][0] + lam, A[1][1] + lam, A[0][1]
        det = a00 * a11 - a01 * a01
        return [(a11 * b[0] - a01 * b[1]) / det, (a00 * b[1] - a01 * b[0]) / det]

    # eigenvalues of X X^T
    half = (A[0][0] + A[1][1]) / 2
    disc = math.sqrt(((A[0][0] - A[1][1]) / 2) ** 2 + A[0][1] ** 2)
    eig = [half + disc, half - disc]
    print(f"eigenvalues of X X^T: {eig[0]:.3f}, {eig[1]:.3f}")

    print("lambda   w                    |w|^2    trace((XX^T + lambda I)^-1)")
    for lam in (0.0, 0.1, 1.0, 10.0, 100.0):
        w = solve(lam)
        trace = sum(1 / (e + lam) for e in eig)
        print(f"{lam:6.1f}   ({w[0]:.4f}, {w[1]:.4f})   {w[0] ** 2 + w[1] ** 2:.4f}   {trace:.4f}")
  `, r`
    eigenvalues of X X^T: 16.639, 0.361
    lambda   w                    |w|^2    trace((XX^T + lambda I)^-1)
       0.0   (0.6667, 0.5000)   0.6944   2.8333
       0.1   (0.5837, 0.5318)   0.6234   2.2309
       1.0   (0.3750, 0.5833)   0.4809   0.7917
      10.0   (0.1957, 0.4094)   0.2059   0.1341
     100.0   (0.0431, 0.0942)   0.0107   0.0185
  `, "Adding λ to the diagonal of XXᵀ shrinks the weights and, more dramatically, the trace that controls the estimator's variance: the small eigenvalue 0.36 is what makes the unregularised estimate fragile."),
  "mlt-lasso": py(r`
    def soft(a, lam):                       # soft-thresholding operator S(a, lambda)
        return (a - lam) if a > lam else (a + lam) if a < -lam else 0.0

    # orthonormal features: LASSO soft-thresholds, ridge rescales
    w_ml = [3.0, -0.5, 1.2]
    lam = 1.0
    print("least squares:", w_ml)
    print("LASSO:        ", [round(soft(a, lam), 4) for a in w_ml])
    print("ridge:        ", [round(a / (1 + lam), 4) for a in w_ml])

    # two correlated features: cyclic coordinate descent on (1/2) w^T A w - b^T w + lam |w|_1
    A = [[10.0, 8.0], [8.0, 10.0]]          # X X^T
    b = [14.0, 10.0]                        # X y
    print("\nlambda   w1      w2")
    for lam in (0.0, 1.0, 2.0, 4.0, 14.0):
        w = [0.0, 0.0]
        for _ in range(500):
            w[0] = soft(b[0] - A[0][1] * w[1], lam) / A[0][0]
            w[1] = soft(b[1] - A[1][0] * w[0], lam) / A[1][1]
        print(f"{lam:6.1f}   {w[0]:.4f}  {w[1]:.4f}")
  `, r`
    least squares: [3.0, -0.5, 1.2]
    LASSO:         [2.0, 0.0, 0.2]
    ridge:         [1.5, -0.25, 0.6]

    lambda   w1      w2
       0.0   1.6667  -0.3333
       1.0   1.3000  0.0000
       2.0   1.2000  0.0000
       4.0   1.0000  0.0000
      14.0   0.0000  0.0000
  `, "With orthonormal features LASSO subtracts λ from each magnitude and clips at zero, removing the small weight entirely. In the correlated example the second weight reaches exactly zero at λ = 2 while the first is still non-zero, and both vanish once λ reaches 14."),
  "mlt-cross-validation": py(r`
    # six (x, y) pairs; model y = w x fitted by ridge regression: w = sum(xy) / (sum(x^2) + lambda)
    data = [(-3, -1.9), (-2, 0.4), (-1, -1.1), (1, 1.6), (2, -0.6), (3, 1.2)]   # weak trend, strong noise
    k = 3
    folds = [data[i::k] for i in range(k)]            # every k-th point goes to the same fold

    def fit(train, lam):
        return sum(x * y for x, y in train) / (sum(x * x for x, _ in train) + lam)

    def mse(w, pts):
        return sum((y - w * x) ** 2 for x, y in pts) / len(pts)

    best = None
    for lam in (0.0, 1.0, 10.0, 100.0, 1000.0):
        errs = []
        for j in range(k):
            train = [p for i, f in enumerate(folds) if i != j for p in f]
            w = fit(train, lam)                        # trained without fold j
            errs.append(mse(w, folds[j]))              # validated on fold j
        cv = sum(errs) / k
        train_err = mse(fit(data, lam), data)
        print(f"lambda = {lam:6.1f}: fold errors = {[round(e, 4) for e in errs]}, CV = {cv:.4f}, training MSE = {train_err:.4f}")
        if best is None or cv < best[1]:
            best = (lam, cv)
    print(f"selected lambda = {best[0]}, final w on all data = {fit(data, best[0]):.4f}")
  `, r`
    lambda =    0.0: fold errors = [2.1025, 2.9, 0.3746], CV = 1.7924, training MSE = 0.9614
    lambda =    1.0: fold errors = [2.1486, 2.709, 0.403], CV = 1.7535, training MSE = 0.9621
    lambda =   10.0: fold errors = [2.4276, 1.7, 0.6145], CV = 1.5807, training MSE = 1.0027
    lambda =  100.0: fold errors = [2.9206, 0.5, 1.124], CV = 1.5149, training MSE = 1.3247
    lambda = 1000.0: fold errors = [3.0657, 0.2841, 1.3007], CV = 1.5501, training MSE = 1.5247
    selected lambda = 100.0, final w on all data = 0.0781
  `, "For each λ the model is trained three times, each time leaving one fold out and measuring the error on it. The training error is smallest at λ = 0, as it always is, but on these noisy data the held-out error keeps improving until λ = 100: cross-validation selects that value and the model is refitted on all the data."),
  "mlt-ridge-bayesian": py(r`
    A = [[3.0, 6.0], [6.0, 14.0]]      # X X^T
    b = [5.0, 11.0]                    # X y
    yy = 9.0                           # y^T y  (labels 1, 2, 2)

    def ridge(lam):
        a00, a11, a01 = A[0][0] + lam, A[1][1] + lam, A[0][1]
        det = a00 * a11 - a01 * a01
        return [(a11 * b[0] - a01 * b[1]) / det, (a00 * b[1] - a01 * b[0]) / det]

    def neg_log_posterior(w, sigma2, gamma2):
        sse = w[0] * (A[0][0] * w[0] + A[0][1] * w[1]) + w[1] * (A[1][0] * w[0] + A[1][1] * w[1]) - 2 * (b[0] * w[0] + b[1] * w[1]) + yy
        return sse / (2 * sigma2) + (w[0] ** 2 + w[1] ** 2) / (2 * gamma2)

    for sigma2, gamma2 in ((1.0, 1.0), (1.0, 0.25), (4.0, 0.25), (1.0, 100.0)):
        lam = sigma2 / gamma2
        w = ridge(lam)
        here = neg_log_posterior(w, sigma2, gamma2)
        # the ridge solution should beat every nearby weight vector
        nearby = min(neg_log_posterior([w[0] + dx, w[1] + dy], sigma2, gamma2) for dx in (-0.05, 0.05) for dy in (-0.05, 0.05))
        print(f"sigma^2 = {sigma2}, gamma^2 = {gamma2}: lambda = {lam:g}, w_MAP = ({w[0]:.4f}, {w[1]:.4f}), -log posterior {here:.4f} < {nearby:.4f}")
  `, r`
    sigma^2 = 1.0, gamma^2 = 1.0: lambda = 1, w_MAP = (0.3750, 0.5833), -log posterior 0.3542 < 0.3629
    sigma^2 = 1.0, gamma^2 = 0.25: lambda = 4, w_MAP = (0.2667, 0.5222), -log posterior 0.9611 < 0.9774
    sigma^2 = 4.0, gamma^2 = 0.25: lambda = 16, w_MAP = (0.1573, 0.3352), -log posterior 0.5658 < 0.5773
    sigma^2 = 1.0, gamma^2 = 100.0: lambda = 0.01, w_MAP = (0.6564, 0.5040), -log posterior 0.0868 < 0.0931
  `, "For each noise variance and prior variance, λ = σ²/γ² is plugged into the ridge formula. The resulting weights give a lower negative log-posterior than any nearby weights, confirming that the ridge solution is the MAP estimate."),
  "mlt-zero-one-linear": py(r`
    import math

    data = [((2, 1), 1), ((1, 3), -1), ((3, 2), -1), ((0, 1), 1)]

    def predict(w, x):
        return 1 if w[0] * x[0] + w[1] * x[1] >= 0 else -1       # sign(0) = +1

    def zero_one(w):
        return sum(predict(w, x) != y for x, y in data) / len(data)

    w = (1, -1)
    for x, y in data:
        s = w[0] * x[0] + w[1] * x[1]
        print(f"x = {x}: w.x = {s:+d}, prediction {predict(w, x):+d}, label {y:+d}  {'ok' if predict(w, x) == y else 'MISTAKE'}")
    print("zero-one loss:", zero_one(w))
    print("after scaling w by 10:", zero_one((10, -10)))

    # the loss as w rotates: only a few distinct values, constant in between
    losses = [zero_one((math.cos(math.radians(a)), math.sin(math.radians(a)))) for a in range(0, 360, 5)]
    print("distinct loss values over all directions:", sorted(set(losses)))
    print("smallest achievable loss:", min(losses))
  `, r`
    x = (2, 1): w.x = +1, prediction +1, label +1  ok
    x = (1, 3): w.x = -2, prediction -1, label -1  ok
    x = (3, 2): w.x = +1, prediction +1, label -1  MISTAKE
    x = (0, 1): w.x = -1, prediction -1, label +1  MISTAKE
    zero-one loss: 0.5
    after scaling w by 10: 0.5
    distinct loss values over all directions: [0.25, 0.5, 0.75]
    smallest achievable loss: 0.25
  `, "The loss counts sign disagreements and is unchanged by rescaling w. Rotating w through a full circle produces only a handful of distinct values: the loss is a staircase, which is why it offers no gradient to follow."),
  "mlt-knn": py(r`
    import math

    train = [((1, 0), 1), ((0, 2), -1), ((-1, -1), 1), ((2, 2), -1), ((0, -3), -1)]
    query = (0, 0)

    by_distance = sorted(train, key=lambda p: math.dist(p[0], query))
    for x, y in by_distance:
        print(f"point {x}: distance {math.dist(x, query):.3f}, label {y:+d}")

    for k in (1, 3, 5):
        votes = [y for _, y in by_distance[:k]]
        pred = 1 if sum(votes) > 0 else -1
        print(f"k = {k}: votes {votes} -> prediction {pred:+d}")

    # with Manhattan distance the neighbours can change
    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    order = sorted(train, key=lambda p: manhattan(p[0], query))
    print("Manhattan order:", [x for x, _ in order])
  `, r`
    point (1, 0): distance 1.000, label +1
    point (-1, -1): distance 1.414, label +1
    point (0, 2): distance 2.000, label -1
    point (2, 2): distance 2.828, label -1
    point (0, -3): distance 3.000, label -1
    k = 1: votes [1] -> prediction +1
    k = 3: votes [1, 1, -1] -> prediction +1
    k = 5: votes [1, 1, -1, -1, -1] -> prediction -1
    Manhattan order: [(1, 0), (0, 2), (-1, -1), (0, -3), (2, 2)]
  `, "Training points are sorted by distance to the query and the nearest k vote. The prediction flips from +1 to −1 when k reaches 5, because all five points — and so the majority class of the dataset — then take part."),
  "mlt-decision-stumps": py(r`
    import math

    def H(p):                                   # binary entropy in bits
        return 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def entropy(labels):
        return H(sum(1 for y in labels if y == 1) / len(labels)) if labels else 0.0

    # ten points with one feature
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 1, 1, 1, -1, 1, -1, -1, -1, -1]

    parent = entropy(y)
    print(f"H(parent) = {parent:.4f}")

    best = None
    for i in range(len(x) - 1):
        theta = (x[i] + x[i + 1]) / 2                       # candidate thresholds between points
        yes = [yi for xi, yi in zip(x, y) if xi <= theta]
        no = [yi for xi, yi in zip(x, y) if xi > theta]
        g = len(yes) / len(x)
        ig = parent - (g * entropy(yes) + (1 - g) * entropy(no))
        print(f"x <= {theta:4.1f}: H(yes) = {entropy(yes):.4f}, H(no) = {entropy(no):.4f}, IG = {ig:.4f}")
        if best is None or ig > best[1]:
            best = (theta, ig)
    print(f"best stump: x <= {best[0]} with information gain {best[1]:.4f}")
  `, r`
    H(parent) = 1.0000
    x <=  1.5: H(yes) = 0.0000, H(no) = 0.9911, IG = 0.1080
    x <=  2.5: H(yes) = 0.0000, H(no) = 0.9544, IG = 0.2365
    x <=  3.5: H(yes) = 0.0000, H(no) = 0.8631, IG = 0.3958
    x <=  4.5: H(yes) = 0.0000, H(no) = 0.6500, IG = 0.6100
    x <=  5.5: H(yes) = 0.7219, H(no) = 0.7219, IG = 0.2781
    x <=  6.5: H(yes) = 0.6500, H(no) = 0.0000, IG = 0.6100
    x <=  7.5: H(yes) = 0.8631, H(no) = 0.0000, IG = 0.3958
    x <=  8.5: H(yes) = 0.9544, H(no) = 0.0000, IG = 0.2365
    x <=  9.5: H(yes) = 0.9911, H(no) = 0.0000, IG = 0.1080
    best stump: x <= 4.5 with information gain 0.6100
  `, "Every threshold between consecutive feature values is tried. The information gain is the parent's entropy minus the size-weighted entropies of the two children, and the stump keeps the threshold with the largest gain."),
  "mlt-tree-growing": py(r`
    import math

    def H(labels):
        if not labels:
            return 0.0
        p = labels.count("+") / len(labels)
        return 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def grow(points, depth=0, max_depth=3):
        labels = [y for _, y in points]
        indent = "    " * depth
        if H(labels) == 0 or depth == max_depth:               # stopping criteria
            majority = max(set(labels), key=labels.count)
            print(f"{indent}leaf: predict {majority}  ({len(points)} points)")
            return
        xs = sorted(x for x, _ in points)
        best = None
        for a, b in zip(xs, xs[1:]):                            # every threshold between neighbours
            t = (a + b) / 2
            yes = [y for x, y in points if x <= t]
            no = [y for x, y in points if x > t]
            ig = H(labels) - (len(yes) * H(yes) + len(no) * H(no)) / len(points)
            if best is None or ig > best[1] + 1e-12:
                best = (t, ig)
        t, ig = best
        print(f"{indent}x <= {t}?  (information gain {ig:.3f})")
        grow([p for p in points if p[0] <= t], depth + 1, max_depth)
        grow([p for p in points if p[0] > t], depth + 1, max_depth)

    data = list(zip([1, 2, 3, 4, 5, 6], "++--++"))
    grow(data)
  `, r`
    x <= 2.5?  (information gain 0.252)
        leaf: predict +  (2 points)
        x <= 4.5?  (information gain 1.000)
            leaf: predict -  (2 points)
            leaf: predict +  (2 points)
  `, "The same best-question search is applied recursively to each subset until a node is pure. Two questions are enough here; the first cannot separate the classes on its own, and the second finishes the job on the impure branch."),
  "mlt-generative-discriminative": py(r`
    prior = {1: 0.3, 0: 0.7}                 # P(y)
    likelihood = {1: 0.8, 0: 0.1}            # P(x | y) for the observed features x

    joint = {y: prior[y] * likelihood[y] for y in (0, 1)}      # P(x, y) = P(y) P(x | y)
    evidence = sum(joint.values())                              # P(x)
    posterior = {y: joint[y] / evidence for y in (0, 1)}
    print("joint:    ", {y: round(v, 4) for y, v in joint.items()})
    print("evidence: ", round(evidence, 4))
    print("posterior:", {y: round(v, 4) for y, v in posterior.items()})
    print("prediction:", max(posterior, key=posterior.get))

    # parameters of a generative model over d binary features and a binary label
    print("\n d   unrestricted   naive Bayes")
    for d in (2, 5, 10, 20, 30):
        print(f"{d:2d}   {1 + 2 * (2 ** d - 1):12,d}   {1 + 2 * d:11d}")
  `, r`
    joint:     {0: 0.07, 1: 0.24}
    evidence:  0.31
    posterior: {0: 0.2258, 1: 0.7742}
    prediction: 1

     d   unrestricted   naive Bayes
     2              7             5
     5             63            11
    10          2,047            21
    20      2,097,151            41
    30   2,147,483,647            61
  `, "Prior times class-conditional gives the joint, and normalising the joint over the classes gives the posterior. The table shows why an unrestricted class-conditional distribution is hopeless: its parameter count doubles with every added feature."),
  "mlt-naive-bayes": py(r`
    # each row: (offer, free, meeting) word indicators, label 1 = spam
    data = [
        ((1, 1, 0), 1), ((1, 0, 0), 1), ((1, 1, 0), 1), ((0, 1, 1), 1),
        ((0, 0, 1), 0), ((0, 1, 1), 0), ((0, 0, 1), 0), ((0, 0, 1), 0), ((0, 1, 0), 0), ((0, 0, 0), 0),
    ]                                   # note: 'offer' never appears in a ham email
    d = 3

    def fit(smooth):
        n1 = sum(y for _, y in data)
        counts = {1: n1, 0: len(data) - n1}
        prior = n1 / len(data)
        p = {y: [(sum(x[j] for x, lab in data if lab == y) + smooth) / (counts[y] + 2 * smooth) for j in range(d)] for y in (0, 1)}
        return prior, p

    def posterior(x, prior, p):
        score = {1: prior, 0: 1 - prior}
        for y in (0, 1):
            for j in range(d):
                score[y] *= p[y][j] if x[j] else 1 - p[y][j]
        total = score[0] + score[1]
        return score[1] / total if total else float("nan")

    for smooth in (0, 1):
        prior, p = fit(smooth)
        print(f"smoothing = {smooth}: P(spam) = {prior}")
        print("  P(word | spam):", [round(v, 3) for v in p[1]])
        print("  P(word | ham): ", [round(v, 3) for v in p[0]])
        for x in ((1, 1, 0), (0, 0, 1), (1, 0, 1)):
            print(f"  x = {x}: P(spam | x) = {posterior(x, prior, p):.4f}")
  `, r`
    smoothing = 0: P(spam) = 0.4
      P(word | spam): [0.75, 0.75, 0.25]
      P(word | ham):  [0.0, 0.333, 0.667]
      x = (1, 1, 0): P(spam | x) = 1.0000
      x = (0, 0, 1): P(spam | x) = 0.0229
      x = (1, 0, 1): P(spam | x) = 1.0000
    smoothing = 1: P(spam) = 0.4
      P(word | spam): [0.667, 0.667, 0.333]
      P(word | ham):  [0.125, 0.375, 0.625]
      x = (1, 1, 0): P(spam | x) = 0.9183
      x = (0, 0, 1): P(spam | x) = 0.0674
      x = (1, 0, 1): P(spam | x) = 0.5028
  `, "The parameters are fractions of training emails. Because 'offer' never occurs in a ham email its unsmoothed estimate is 0, so any email containing it is declared spam with probability exactly 1 — even (1, 0, 1), which also contains the ham-like word 'meeting'. With one pseudo-count added each way that email becomes a near coin flip."),
  "mlt-gaussian-naive-bayes": py(r`
    import math

    # one feature, two classes
    class1 = [1.2, 2.5, 1.8, 2.9, 1.6]
    class0 = [-0.4, 0.6, -1.1, 0.3, 0.9, -0.3]

    def fit(xs):
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)        # maximum-likelihood variance
        return mu, var

    (mu1, v1), (mu0, v0) = fit(class1), fit(class0)
    n1, n0 = len(class1), len(class0)
    p1 = n1 / (n1 + n0)
    print(f"class 1: mu = {mu1:.3f}, var = {v1:.3f}, prior = {p1:.3f}")
    print(f"class 0: mu = {mu0:.3f}, var = {v0:.3f}, prior = {1 - p1:.3f}")

    def log_joint(x, mu, var, prior):
        return math.log(prior) - 0.5 * math.log(2 * math.pi * var) - (x - mu) ** 2 / (2 * var)

    for x in (0.0, 1.0, 1.2, 2.0):
        g1, g0 = log_joint(x, mu1, v1, p1), log_joint(x, mu0, v0, 1 - p1)
        post = 1 / (1 + math.exp(g0 - g1))
        print(f"x = {x}: log-odds = {g1 - g0:+.3f}, P(class 1 | x) = {post:.3f} -> class {1 if g1 >= g0 else 0}")

    # shared variance: closed-form boundary
    pooled = (n1 * v1 + n0 * v0) / (n1 + n0)
    boundary = (mu1 + mu0) / 2 - pooled / (mu1 - mu0) * math.log(p1 / (1 - p1))
    print(f"boundary with a pooled variance {pooled:.3f}: x* = {boundary:.3f}")
  `, r`
    class 1: mu = 2.000, var = 0.380, prior = 0.455
    class 0: mu = -0.000, var = 0.453, prior = 0.545
    x = 0.0: log-odds = -5.357, P(class 1 | x) = 0.005 -> class 0
    x = 1.0: log-odds = -0.307, P(class 1 | x) = 0.424 -> class 0
    x = 1.2: log-odds = +0.652, P(class 1 | x) = 0.657 -> class 1
    x = 2.0: log-odds = +4.318, P(class 1 | x) = 0.987 -> class 1
    boundary with a pooled variance 0.420: x* = 1.038
  `, "Each class is summarised by a mean, a variance and a prior. Classification compares the log of prior times Gaussian density; with a shared variance the comparison reduces to a single threshold, shifted from the midpoint of the means by the prior."),
  "mlt-lda": py(r`
    import math

    mu1, mu0 = (2.0, 2.0), (0.0, 0.0)
    Sigma = [[2.0, 0.0], [0.0, 1.0]]          # shared covariance
    p = 0.5                                    # prior of class 1

    det = Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0]
    Sinv = [[Sigma[1][1] / det, -Sigma[0][1] / det], [-Sigma[1][0] / det, Sigma[0][0] / det]]

    def quad(u, v):                            # u^T Sigma^-1 v
        return sum(u[a] * Sinv[a][b] * v[b] for a in range(2) for b in range(2))

    diff = (mu1[0] - mu0[0], mu1[1] - mu0[1])
    w = [sum(Sinv[a][b] * diff[b] for b in range(2)) for a in range(2)]
    b = -0.5 * (quad(mu1, mu1) - quad(mu0, mu0)) + math.log(p / (1 - p))
    print(f"w = {w}, b = {b}")
    print(f"boundary: {w[0]:g} x1 + {w[1]:g} x2 = {-b:g}")

    for x in ((1, 1), (3, 0), (0, 2), (0.5, 0.5)):
        s = w[0] * x[0] + w[1] * x[1] + b
        post = 1 / (1 + math.exp(-s))
        print(f"x = {x}: w.x + b = {s:+.2f}, P(class 1 | x) = {post:.3f}")
  `, r`
    w = [1.0, 2.0], b = -3.0
    boundary: 1 x1 + 2 x2 = 3
    x = (1, 1): w.x + b = +0.00, P(class 1 | x) = 0.500
    x = (3, 0): w.x + b = +0.00, P(class 1 | x) = 0.500
    x = (0, 2): w.x + b = +1.00, P(class 1 | x) = 0.731
    x = (0.5, 0.5): w.x + b = -1.50, P(class 1 | x) = 0.182
  `, "The weight vector is the difference of the class means multiplied by the inverse shared covariance, so the noisier first feature gets the smaller weight. The log-odds wᵀx + b pass through a sigmoid to give the posterior, exactly the functional form of logistic regression."),
  "mlt-separability-margin": py(r`
    import math

    data = [((2, 1), 1), ((1, 3), 1), ((-1, -1), -1), ((-2, 0), -1)]

    def margin(w):
        norm = math.hypot(*w)
        return min(y * (w[0] * x[0] + w[1] * x[1]) / norm for x, y in data)

    for w in ((1, 1), (1, 0), (0, 1), (1, -1)):
        g = margin(w)
        print(f"w = {w}: margin = {g:+.4f}  {'separates' if g > 0 else 'does not separate'}")

    # search all directions for the largest margin
    best = max(((math.cos(math.radians(a / 10)), math.sin(math.radians(a / 10))) for a in range(3600)), key=margin)
    print(f"largest margin {margin(best):.4f} at w = ({best[0]:.3f}, {best[1]:.3f})")

    R = max(math.hypot(*x) for x, _ in data)
    print(f"R = {R:.4f}, R^2 / gamma^2 = {R ** 2 / margin(best) ** 2:.3f}")
  `, r`
    w = (1, 1): margin = +1.4142  separates
    w = (1, 0): margin = +1.0000  separates
    w = (0, 1): margin = +0.0000  does not separate
    w = (1, -1): margin = -1.4142  does not separate
    largest margin 1.4142 at w = (0.707, 0.707)
    R = 3.1623, R^2 / gamma^2 = 5.000
  `, "The margin of a direction is the smallest signed distance of any point to its hyperplane; a positive value means the direction separates the classes. A scan over directions finds the largest margin, the γ that enters the perceptron bound R²/γ²."),
  "mlt-perceptron": py(r`
    data = [((1, 3), 1), ((2, -1), -1), ((-1, 2), 1), ((1, -2), -1), ((3, 1), 1)]

    def predict(w, x):
        return 1 if w[0] * x[0] + w[1] * x[1] >= 0 else -1       # sign(0) = +1

    w = [0, 0]
    updates, clean, i = 0, 0, 0
    while clean < len(data):                    # stop after a full pass without mistakes
        x, y = data[i]
        if predict(w, x) != y:
            before = y * (w[0] * x[0] + w[1] * x[1])
            w = [w[0] + y * x[0], w[1] + y * x[1]]               # w <- w + y x
            after = y * (w[0] * x[0] + w[1] * x[1])
            updates += 1
            clean = 0
            print(f"update {updates}: mistake on x{i + 1} = {x}, y = {y:+d};  y w.x: {before:+d} -> {after:+d};  w = {w}")
        else:
            clean += 1
        i = (i + 1) % len(data)

    print("final w:", w)
    print("scores:", [w[0] * x[0] + w[1] * x[1] for x, _ in data])
    print("all correct:", all(predict(w, x) == y for x, y in data))
  `, r`
    update 1: mistake on x2 = (2, -1), y = -1;  y w.x: +0 -> +5;  w = [-2, 1]
    update 2: mistake on x5 = (3, 1), y = +1;  y w.x: -5 -> +5;  w = [1, 2]
    update 3: mistake on x2 = (2, -1), y = -1;  y w.x: +0 -> +5;  w = [-1, 3]
    final w: [-1, 3]
    scores: [8, -5, 7, -7, 0]
    all correct: True
  `, "The points are visited cyclically and only mistakes change w. Each update raises the offending point's signed score by ‖x‖², although it may still be non-positive; after three updates a complete pass produces no mistake."),
  "mlt-perceptron-convergence": py(r`
    import math

    data = [((1, 3), 1), ((2, -1), -1), ((-1, 2), 1), ((1, -2), -1), ((3, 1), 1)]

    def margin(w):
        return min(y * (w[0] * x[0] + w[1] * x[1]) for x, y in data) / math.hypot(*w)

    # a unit separator w* with its margin gamma (best direction on a fine grid)
    w_star = max(((math.cos(math.radians(a / 10)), math.sin(math.radians(a / 10))) for a in range(3600)), key=margin)
    gamma = margin(w_star)
    R = max(math.hypot(*x) for x, _ in data)
    print(f"gamma = {gamma:.4f}, R = {R:.4f}, bound R^2/gamma^2 = {R ** 2 / gamma ** 2:.2f}")

    w, l, clean, i = [0.0, 0.0], 0, 0, 0
    while clean < len(data):
        x, y = data[i]
        if (1 if w[0] * x[0] + w[1] * x[1] >= 0 else -1) != y:
            w = [w[0] + y * x[0], w[1] + y * x[1]]
            l += 1
            clean = 0
            norm2 = w[0] ** 2 + w[1] ** 2
            align = w[0] * w_star[0] + w[1] * w_star[1]
            print(f"mistake {l}: |w|^2 = {norm2:5.1f} <= l R^2 = {l * R * R:5.1f};   w.w* = {align:6.3f} >= l gamma = {l * gamma:6.3f}")
        else:
            clean += 1
        i = (i + 1) % len(data)
    print(f"mistakes made: {l}  (bound: {R ** 2 / gamma ** 2:.2f})")
  `, r`
    gamma = 1.0000, R = 3.1623, bound R^2/gamma^2 = 10.00
    mistake 1: |w|^2 =   5.0 <= l R^2 =  10.0;   w.w* =  1.000 >= l gamma =  1.000
    mistake 2: |w|^2 =   5.0 <= l R^2 =  20.0;   w.w* =  2.000 >= l gamma =  2.000
    mistake 3: |w|^2 =  10.0 <= l R^2 =  30.0;   w.w* =  3.000 >= l gamma =  3.000
    mistakes made: 3  (bound: 10.00)
  `, "After every mistake the two inequalities of the proof are checked on the actual weights: the squared norm stays below lR² and the alignment with w* stays above lγ. Together they force the number of mistakes below R²/γ²."),
  "mlt-logistic-sigmoid": py(r`
    import math

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    print("   z   sigmoid   1-sigmoid(-z)   slope s(1-s)")
    for z in (-4, -2, 0, 1.5, 4):
        s = sigmoid(z)
        print(f"{z:5.1f}  {s:.4f}    {1 - sigmoid(-z):.4f}         {s * (1 - s):.4f}")

    w, x = (2, -1), (1, 0.5)
    z = w[0] * x[0] + w[1] * x[1]
    p = sigmoid(z)
    print(f"\nscore z = {z}, P(y = 1 | x) = {p:.4f}")
    print(f"odds = {p / (1 - p):.4f} = e^z = {math.exp(z):.4f}")
    print(f"log-odds = {math.log(p / (1 - p)):.4f}")
    print("prediction:", 1 if p >= 0.5 else 0)

    # one more unit of x1 adds w1 to the log-odds
    p2 = sigmoid(z + w[0])
    print(f"after x1 += 1: P = {p2:.4f}, log-odds = {math.log(p2 / (1 - p2)):.4f}")
  `, r`
       z   sigmoid   1-sigmoid(-z)   slope s(1-s)
     -4.0  0.0180    0.0180         0.0177
     -2.0  0.1192    0.1192         0.1050
      0.0  0.5000    0.5000         0.2500
      1.5  0.8176    0.8176         0.1491
      4.0  0.9820    0.9820         0.0177

    score z = 1.5, P(y = 1 | x) = 0.8176
    odds = 4.4817 = e^z = 4.4817
    log-odds = 1.5000
    prediction: 1
    after x1 += 1: P = 0.9707, log-odds = 3.5000
  `, "The table verifies the identity σ(−z) = 1 − σ(z) and shows the slope σ(1 − σ) peaking at ¼ when z = 0. For the example point the odds equal e^z, so the log-odds recover the linear score exactly."),
  "mlt-logistic-mle": py(r`
    import math

    data = [((1, 2), 1), ((2, 1), 0), ((0, 3), 1), ((3, 0), 0), ((2, 2), 1), ((2, 3), 0)]

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    def loglik(w):
        total = 0.0
        for x, y in data:
            p = sigmoid(w[0] * x[0] + w[1] * x[1])
            total += math.log(p) if y == 1 else math.log(1 - p)
        return total

    def gradient(w):                        # sum over the data of (y - sigma) x
        g = [0.0, 0.0]
        for x, y in data:
            e = y - sigmoid(w[0] * x[0] + w[1] * x[1])
            g[0] += e * x[0]
            g[1] += e * x[1]
        return g

    w, eta = [0.0, 0.0], 0.1
    print(f"step   0: w = ({w[0]:+.4f}, {w[1]:+.4f}), log-likelihood = {loglik(w):.4f}")
    for t in range(1, 2001):
        g = gradient(w)
        w = [w[0] + eta * g[0], w[1] + eta * g[1]]          # ascent: move along the gradient
        if t in (1, 10, 100, 2000):
            print(f"step {t:3d}: w = ({w[0]:+.4f}, {w[1]:+.4f}), log-likelihood = {loglik(w):.4f}, |gradient| = {math.hypot(*gradient(w)):.5f}")
    acc = sum((sigmoid(w[0] * x[0] + w[1] * x[1]) >= 0.5) == (y == 1) for x, y in data) / len(data)
    print(f"training accuracy: {acc:.3f}")
  `, r`
    step   0: w = (+0.0000, +0.0000), log-likelihood = -4.1589
    step   1: w = (-0.2000, +0.1500), log-likelihood = -3.6139, |gradient| = 1.86883
    step  10: w = (-0.8649, +0.6860), log-likelihood = -2.7239, |gradient| = 0.42976
    step 100: w = (-1.3826, +1.0864), log-likelihood = -2.6015, |gradient| = 0.00426
    step 2000: w = (-1.3908, +1.0927), log-likelihood = -2.6015, |gradient| = 0.00000
    training accuracy: 0.667
  `, "Each step adds the gradient Σ(yᵢ − σᵢ)xᵢ, scaled by η, to the weights. The log-likelihood climbs monotonically and the gradient shrinks towards zero; because this dataset is not separable the weights settle at finite values."),
  "mlt-svm-margins": py(r`
    import math

    data = [((1, 1), 1), ((3, 2), 1), ((-1, -1), -1), ((-2, 1), -1)]

    def margins(w):
        functional = min(y * (w[0] * x[0] + w[1] * x[1]) for x, y in data)
        return functional, functional / math.hypot(*w)

    print("w                functional   geometric   feasible   (1/2)|w|^2")
    for w in ((1, 1), (2, 2), (0.5, 0.5), (1, 0), (3, 2)):
        f, g = margins(w)
        print(f"{str(w):15s}  {f:8.3f}     {g:7.4f}     {str(f >= 1):8s}   {0.5 * (w[0] ** 2 + w[1] ** 2):.3f}")

    # rescale a separating w so that its functional margin is exactly 1: then margin = 1/|w|
    for w in ((1, 1), (3, 2)):
        f, g = margins(w)
        w1 = (w[0] / f, w[1] / f)
        print(f"rescaled {w} -> ({w1[0]:.3f}, {w1[1]:.3f}): (1/2)|w|^2 = {0.5 * (w1[0] ** 2 + w1[1] ** 2):.4f}, margin 1/|w| = {1 / math.hypot(*w1):.4f}")

    # the best direction over a fine grid: largest geometric margin = smallest feasible |w|
    best = max(((math.cos(math.radians(a / 10)), math.sin(math.radians(a / 10))) for a in range(3600)), key=lambda w: margins(w)[1])
    g = margins(best)[1]
    print(f"maximum margin {g:.4f} in direction ({best[0]:.3f}, {best[1]:.3f}); optimal (1/2)|w|^2 = {0.5 / g ** 2:.4f}")
  `, r`
    w                functional   geometric   feasible   (1/2)|w|^2
    (1, 1)              1.000      0.7071     True       1.000
    (2, 2)              2.000      0.7071     True       4.000
    (0.5, 0.5)          0.500      0.7071     False      0.250
    (1, 0)              1.000      1.0000     True       0.500
    (3, 2)              4.000      1.1094     True       6.500
    rescaled (1, 1) -> (1.000, 1.000): (1/2)|w|^2 = 1.0000, margin 1/|w| = 0.7071
    rescaled (3, 2) -> (0.750, 0.500): (1/2)|w|^2 = 0.4062, margin 1/|w| = 1.1094
    maximum margin 1.3411 in direction (0.895, 0.446); optimal (1/2)|w|^2 = 0.2780
  `, "Doubling or halving w changes the functional margin but leaves the geometric margin at 0.7071. Rescaling each direction so that its functional margin is 1 makes the margin equal to 1/‖w‖, so the direction with the widest margin is the one with the smallest ½‖w‖² — the quantity the hard-margin SVM minimises."),
  "mlt-svm-dual": py(r`
    data = [((1, 1), 1), ((3, 2), 1), ((-1, -1), -1), ((-2, 1), -1), ((2, -1), 1)]
    n = len(data)
    alpha = [0.0] * n
    w = [0.0, 0.0]

    def dual():
        return sum(alpha) - 0.5 * (w[0] ** 2 + w[1] ** 2)

    def primal_of_rescaled():
        m = min(y * (w[0] * x[0] + w[1] * x[1]) for x, y in data)
        return 0.5 * (w[0] ** 2 + w[1] ** 2) / m ** 2 if m > 0 else float("inf")

    for sweep in range(1, 201):
        for i, (x, y) in enumerate(data):
            # exact maximisation over alpha_i, keeping alpha_i >= 0
            g = 1 - y * (w[0] * x[0] + w[1] * x[1])
            new = max(0.0, alpha[i] + g / (x[0] ** 2 + x[1] ** 2))
            step = new - alpha[i]
            alpha[i] = new
            w = [w[0] + step * y * x[0], w[1] + step * y * x[1]]      # keep w = sum alpha_i y_i x_i
        if sweep in (1, 2, 5, 20, 200):
            print(f"sweep {sweep:3d}: dual = {dual():.5f}, primal (rescaled w) = {primal_of_rescaled():.5f}")

    print("alpha =", [round(a, 4) for a in alpha])
    print(f"w = ({w[0]:.4f}, {w[1]:.4f}),  sum(alpha) = {sum(alpha):.4f},  |w|^2 = {w[0] ** 2 + w[1] ** 2:.4f}")
  `, r`
    sweep   1: dual = 0.27500, primal (rescaled w) = 0.32500
    sweep   2: dual = 0.27775, primal (rescaled w) = 0.28225
    sweep   5: dual = 0.27778, primal (rescaled w) = 0.27778
    sweep  20: dual = 0.27778, primal (rescaled w) = 0.27778
    sweep 200: dual = 0.27778, primal (rescaled w) = 0.27778
    alpha = [0.4444, 0.0, 0.0, 0.1111, 0.0]
    w = (0.6667, 0.3333),  sum(alpha) = 0.5556,  |w|^2 = 0.5556
  `, "The dual is maximised one coordinate at a time while w is kept equal to Σαᵢyᵢxᵢ. The dual value rises, the primal value of a feasible rescaling of w falls, and they meet; at the optimum Σαᵢ equals ‖w‖², a consequence of strong duality."),
  "mlt-svm-support-vectors": py(r`
    data = [((2, 1), 1), ((1, 2), 1), ((3, 3), 1), ((-3, -1), -1), ((-2, -4), -1)]
    n = len(data)
    alpha, w = [0.0] * n, [0.0, 0.0]

    for _ in range(2000):                      # dual coordinate ascent (hard margin, no bias)
        for i, (x, y) in enumerate(data):
            g = 1 - y * (w[0] * x[0] + w[1] * x[1])
            new = max(0.0, alpha[i] + g / (x[0] ** 2 + x[1] ** 2))
            step = new - alpha[i]
            alpha[i] = new
            w = [w[0] + step * y * x[0], w[1] + step * y * x[1]]

    print(f"w* = ({w[0]:.4f}, {w[1]:.4f})")
    print(" i   y   y w.x    alpha    alpha (1 - y w.x)")
    for i, (x, y) in enumerate(data):
        m = y * (w[0] * x[0] + w[1] * x[1])
        tag = "  <- support vector" if alpha[i] > 1e-9 else ""
        print(f"{i + 1:2d}  {y:+d}  {m:6.3f}   {alpha[i]:.4f}   {abs(alpha[i] * (1 - m)):.6f}{tag}")

    # rebuild w from the support vectors only
    sv = [i for i in range(n) if alpha[i] > 1e-9]
    w_sv = [sum(alpha[i] * data[i][1] * data[i][0][c] for i in sv) for c in range(2)]
    print("w from support vectors only:", [round(v, 4) for v in w_sv])
  `, r`
    w* = (0.3333, 0.3333)
     i   y   y w.x    alpha    alpha (1 - y w.x)
     1  +1   1.000   0.1111   0.000000  <- support vector
     2  +1   1.000   0.1111   0.000000  <- support vector
     3  +1   2.000   0.0000   0.000000
     4  -1   1.333   0.0000   0.000000
     5  -1   2.000   0.0000   0.000000
    w from support vectors only: [0.3333, 0.3333]
  `, "After solving the dual, only the two points on the margin — (2, 1) and (1, 2) — have non-zero multipliers. Every product αᵢ(1 − yᵢwᵀxᵢ) is zero, and summing αᵢyᵢxᵢ over the support vectors alone reproduces w*."),
  "mlt-kernel-svm": py(r`
    def k(a, b):                      # polynomial kernel of degree 2 on one-dimensional inputs
        return (1 + a * b) ** 2

    # not linearly separable on the line: the negative point sits between the positives
    xs = [-2.0, 0.0, 2.0, -3.0, 0.5]
    ys = [1, -1, 1, 1, -1]
    n = len(xs)
    K = [[k(a, b) for b in xs] for a in xs]
    alpha = [0.0] * n

    def f(i):                         # sum_j alpha_j y_j K_ij  (the score of training point i)
        return sum(alpha[j] * ys[j] * K[j][i] for j in range(n))

    for _ in range(20000):            # dual coordinate ascent with the kernel matrix
        for i in range(n):
            alpha[i] = max(0.0, alpha[i] + (1 - ys[i] * f(i)) / K[i][i])

    print("alpha =", [round(a, 4) for a in alpha])
    print("training scores y f(x):", [round(ys[i] * f(i), 3) for i in range(n)])

    def predict(t):
        s = sum(alpha[j] * ys[j] * k(xs[j], t) for j in range(n) if alpha[j] > 1e-9)
        return s, (1 if s >= 0 else -1)

    for t in (-2.5, -1.0, 0.2, 1.0, 3.0):
        s, label = predict(t)
        print(f"x = {t:+.1f}: score = {s:+.3f} -> class {label:+d}")
  `, r`
    alpha = [0.0, 0.0, 0.2412, 0.0, 1.2575]
    training scores y f(x): [2.171, 1.016, 1.0, 5.715, 1.0]
    x = -2.5: score = +3.780 -> class +1
    x = -1.0: score = -0.073 -> class -1
    x = +0.2: score = -1.049 -> class -1
    x = +1.0: score = -0.659 -> class -1
    x = +3.0: score = +3.959 -> class +1
  `, "With the degree-2 kernel the classes become separable in feature space, so the same dual coordinate ascent — now driven by the kernel matrix — finds multipliers with every training score at least 1. Predictions sum kernel values against the support vectors and are positive on both outer sides of the line."),
  "mlt-soft-margin": py(r`
    data = [((2, 0), 1), ((0.5, 1), 1), ((-0.5, 2), 1), ((-3, 0), -1)]
    w, C = (1.0, 0.0), 1.0

    total = 0.0
    for x, y in data:
        u = y * (w[0] * x[0] + w[1] * x[1])          # signed score
        xi = max(0.0, 1 - u)                         # slack = hinge loss
        where = "outside the margin" if xi == 0 else "inside the margin, correct side" if xi <= 1 else "misclassified"
        total += xi
        print(f"x = {x}, y = {y:+d}: y w.x = {u:+.1f}, slack = {xi:.1f}  ({where})")

    reg = 0.5 * (w[0] ** 2 + w[1] ** 2)
    print(f"objective = {reg} + {C} * {total} = {reg + C * total}")

    # a larger dataset: three candidate lengths of w judged under three penalties C
    more = data + [((0.4, -1), 1), ((-0.6, 1), -1)]
    print("\nC      w = (0.5, 0)   w = (1, 0)   w = (2, 0)   best")
    for C in (0.1, 1.0, 10.0):
        vals = []
        for s in (0.5, 1.0, 2.0):
            slack = sum(max(0.0, 1 - y * s * x[0]) for x, y in more)
            vals.append(0.5 * s * s + C * slack)
        best = ("(0.5, 0)", "(1, 0)", "(2, 0)")[vals.index(min(vals))]
        print(f"{C:4.1f}   {vals[0]:9.3f}      {vals[1]:7.3f}      {vals[2]:7.3f}      {best}")
  `, r`
    x = (2, 0), y = +1: y w.x = +2.0, slack = 0.0  (outside the margin)
    x = (0.5, 1), y = +1: y w.x = +0.5, slack = 0.5  (inside the margin, correct side)
    x = (-0.5, 2), y = +1: y w.x = -0.5, slack = 1.5  (misclassified)
    x = (-3, 0), y = -1: y w.x = +3.0, slack = 0.0  (outside the margin)
    objective = 0.5 + 1.0 * 2.0 = 2.5

    C      w = (0.5, 0)   w = (1, 0)   w = (2, 0)   best
     0.1       0.475        0.800        2.220      (0.5, 0)
     1.0       3.625        3.500        4.200      (1, 0)
    10.0      35.125       30.500       24.000      (2, 0)
  `, "Each slack is the hinge loss of its point, and the objective adds C times the total slack to ½‖w‖². In the second experiment a longer w narrows the margin and reduces the slack: a small C prefers the shortest w, a large C the longest."),
  "mlt-soft-margin-dual": py(r`
    data = [((2, 1), 1), ((1, 2), 1), ((0.5, 0.2), 1), ((-0.6, 0.4), 1),
            ((-2, -1), -1), ((-1, -2), -1), ((-0.3, -0.6), -1), ((0.7, -0.2), -1)]
    n, C = len(data), 1.0
    alpha, w = [0.0] * n, [0.0, 0.0]

    for _ in range(2000):                      # dual coordinate ascent with the box constraint
        for i, (x, y) in enumerate(data):
            g = 1 - y * (w[0] * x[0] + w[1] * x[1])
            new = min(C, max(0.0, alpha[i] + g / (x[0] ** 2 + x[1] ** 2)))
            step = new - alpha[i]
            alpha[i] = new
            w = [w[0] + step * y * x[0], w[1] + step * y * x[1]]

    print(f"w = ({w[0]:.4f}, {w[1]:.4f}), C = {C}")
    print(" i   y w.x    slack   alpha   category")
    for i, (x, y) in enumerate(data):
        u = y * (w[0] * x[0] + w[1] * x[1])
        xi = max(0.0, 1 - u)
        a = alpha[i]
        cat = "alpha = 0      (outside margin)" if a < 1e-6 else "alpha = C      (violates margin)" if a > C - 1e-6 else "0 < alpha < C  (on the margin)"
        print(f"{i + 1:2d}  {u:+.3f}   {xi:.3f}   {a:.3f}   {cat}")
    print("support vectors:", sum(1 for a in alpha if a > 1e-6), "of", n)
  `, r`
    w = (-0.2600, 1.5200), C = 1.0
     i   y w.x    slack   alpha   category
     1  +1.000   0.000   0.088   0 < alpha < C  (on the margin)
     2  +2.780   0.000   0.000   alpha = 0      (outside margin)
     3  +0.174   0.826   1.000   alpha = C      (violates margin)
     4  +0.764   0.236   1.000   alpha = C      (violates margin)
     5  +1.000   0.000   0.032   0 < alpha < C  (on the margin)
     6  +2.780   0.000   0.000   alpha = 0      (outside margin)
     7  +0.834   0.166   1.000   alpha = C      (violates margin)
     8  +0.486   0.514   1.000   alpha = C      (violates margin)
    support vectors: 6 of 8
  `, "The only change from the hard-margin solver is clipping each multiplier at C. The table reproduces the three scenarios: zero multipliers for points beyond the margin, multipliers at the cap for points that violate it, and intermediate values for the two points sitting exactly on it."),
  "mlt-bagging": py(r`
    import math

    print("   n   P(point not in bag)   expected distinct fraction")
    for n in (2, 5, 10, 100, 1000):
        miss = (1 - 1 / n) ** n
        print(f"{n:4d}   {miss:.4f}                {1 - miss:.4f}")
    print(f"limit: 1/e = {math.exp(-1):.4f}, 1 - 1/e = {1 - math.exp(-1):.4f}")

    # draw bags with a small deterministic generator and count distinct points
    def lcg(seed):
        while True:
            seed = (seed * 1103515245 + 12345) % 2 ** 31
            yield seed

    rng = lcg(7)
    n, B = 100, 200
    fractions = []
    for _ in range(B):
        bag = [next(rng) % n for _ in range(n)]          # n draws with replacement
        fractions.append(len(set(bag)) / n)
    print(f"simulation, n = {n}, {B} bags: mean distinct fraction = {sum(fractions) / B:.4f}")
    first = [next(rng) % 5 + 1 for _ in range(5)]
    print("one bag from {1,...,5}:", first, "-> out-of-bag:", sorted(set(range(1, 6)) - set(first)))
  `, r`
       n   P(point not in bag)   expected distinct fraction
       2   0.2500                0.7500
       5   0.3277                0.6723
      10   0.3487                0.6513
     100   0.3660                0.6340
    1000   0.3677                0.6323
    limit: 1/e = 0.3679, 1 - 1/e = 0.6321
    simulation, n = 100, 200 bags: mean distinct fraction = 0.6370
    one bag from {1,...,5}: [5, 3, 4, 3, 3] -> out-of-bag: [1, 2]
  `, "The exact probability that a point is missing from a bag, (1 − 1/n)ⁿ, approaches 1/e quickly. Drawing 200 bags of 100 points confirms that about 63% of the points appear in each bag."),
  "mlt-random-forests": py(r`
    import math

    def ensemble_variance(sigma2, rho, B):
        return rho * sigma2 + (1 - rho) * sigma2 / B

    print("   B    rho = 0.6 (bagging)   rho = 0.2 (random forest)   rho = 0 (independent)")
    for B in (1, 10, 100, 1000):
        print(f"{B:4d}    {ensemble_variance(1, 0.6, B):.4f}                {ensemble_variance(1, 0.2, B):.4f}                      {ensemble_variance(1, 0.0, B):.4f}")
    print("floor as B grows: rho * sigma^2 =", 0.6, "and", 0.2)

    # features considered at each split
    for d in (4, 16, 100, 784):
        print(f"d = {d:3d}: m = sqrt(d) = {round(math.sqrt(d))} features per split")

    # probability that a particular strong feature is even available at a split
    d, m = 100, 10
    print(f"P(a given feature is among the m candidates) = m/d = {m / d}")
  `, r`
       B    rho = 0.6 (bagging)   rho = 0.2 (random forest)   rho = 0 (independent)
       1    1.0000                1.0000                      1.0000
      10    0.6400                0.2800                      0.1000
     100    0.6040                0.2080                      0.0100
    1000    0.6004                0.2008                      0.0010
    floor as B grows: rho * sigma^2 = 0.6 and 0.2
    d =   4: m = sqrt(d) = 2 features per split
    d =  16: m = sqrt(d) = 4 features per split
    d = 100: m = sqrt(d) = 10 features per split
    d = 784: m = sqrt(d) = 28 features per split
    P(a given feature is among the m candidates) = m/d = 0.1
  `, "The variance of an average of B equally correlated predictors is ρσ² + (1 − ρ)σ²/B. More trees only shrink the second term, so lowering the correlation ρ — what random feature selection does — is what moves the floor from 0.6 to 0.2."),
  "mlt-adaboost": py(r`
    import math

    x = list(range(10))
    y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    n = len(x)
    D = [1 / n] * n
    learners = []

    def stump(t, s, xi):                 # predicts s for xi < t and -s otherwise
        return s if xi < t else -s

    for rnd in range(1, 4):
        # weak learner: the threshold stump with the smallest weighted error
        best = min(((sum(D[i] for i in range(n) if stump(t, s, x[i]) != y[i]), t, s)
                    for t in [v + 0.5 for v in range(-1, 10)] for s in (1, -1)), key=lambda c: c[0])
        eps, t, s = best
        a = 0.5 * math.log((1 - eps) / eps)
        learners.append((a, t, s))
        D = [D[i] * math.exp(-a * y[i] * stump(t, s, x[i])) for i in range(n)]
        Z = sum(D)
        D = [d / Z for d in D]
        H = [1 if sum(al * stump(tt, ss, xi) for al, tt, ss in learners) >= 0 else -1 for xi in x]
        err = sum(h != yi for h, yi in zip(H, y)) / n
        side = "<" if s == 1 else ">="
        print(f"round {rnd}: stump +1 if x {side} {t}; eps = {eps:.4f}, alpha = {a:.4f}, Z = {Z:.4f}, training error of the vote = {err:.1f}")
        print("         new weights:", [round(d, 3) for d in D])
  `, r`
    round 1: stump +1 if x < 2.5; eps = 0.3000, alpha = 0.4236, Z = 0.9165, training error of the vote = 0.3
             new weights: [0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.167, 0.167, 0.167, 0.071]
    round 2: stump +1 if x < 8.5; eps = 0.2143, alpha = 0.6496, Z = 0.8207, training error of the vote = 0.3
             new weights: [0.045, 0.045, 0.045, 0.167, 0.167, 0.167, 0.106, 0.106, 0.106, 0.045]
    round 3: stump +1 if x >= 5.5; eps = 0.1818, alpha = 0.7520, Z = 0.7714, training error of the vote = 0.0
             new weights: [0.125, 0.125, 0.125, 0.102, 0.102, 0.102, 0.065, 0.065, 0.065, 0.125]
  `, "Each round picks the threshold stump with the smallest weighted error, gives it the coefficient α = ½ ln((1 − ε)/ε) and re-weights the examples. No single stump classifies this dataset correctly, but the weighted vote of three does: the training error reaches zero in the third round."),
  "mlt-surrogate-losses": py(r`
    import math

    losses = {
        "0-1":        lambda u: 1.0 if u < 0 else 0.0,
        "squared":    lambda u: (1 - u) ** 2,
        "hinge":      lambda u: max(0.0, 1 - u),
        "logistic":   lambda u: math.log(1 + math.exp(-u)),
        "perceptron": lambda u: max(0.0, -u),
    }

    us = [-2, -0.5, 0, 0.5, 1, 2]
    print("u = y w.x   " + "".join(f"{u:>8}" for u in us))
    for name, f in losses.items():
        print(f"{name:11s} " + "".join(f"{f(u):8.3f}" for u in us))

    # which surrogates are upper bounds on the zero-one loss?
    grid = [i / 100 for i in range(-400, 401)]
    for name, f in losses.items():
        if name != "0-1":
            print(f"{name:11s} >= 0-1 everywhere: {all(f(u) >= losses['0-1'](u) for u in grid)}")
    scaled = lambda u: losses["logistic"](u) / math.log(2)
    print("logistic / ln 2 >= 0-1 everywhere:", all(scaled(u) >= losses["0-1"](u) for u in grid))

    # the squared loss written with the label equals (1 - u)^2
    w, x, y = (0.5, -1.0), (2.0, 3.0), -1
    s = w[0] * x[0] + w[1] * x[1]
    print(f"(w.x - y)^2 = {(s - y) ** 2}, (1 - u)^2 = {(1 - y * s) ** 2}")
  `, r`
    u = y w.x         -2    -0.5       0     0.5       1       2
    0-1            1.000   1.000   0.000   0.000   0.000   0.000
    squared        9.000   2.250   1.000   0.250   0.000   1.000
    hinge          3.000   1.500   1.000   0.500   0.000   0.000
    logistic       2.127   0.974   0.693   0.474   0.313   0.127
    perceptron     2.000   0.500   0.000   0.000   0.000   0.000
    squared     >= 0-1 everywhere: True
    hinge       >= 0-1 everywhere: True
    logistic    >= 0-1 everywhere: False
    perceptron  >= 0-1 everywhere: False
    logistic / ln 2 >= 0-1 everywhere: True
    (w.x - y)^2 = 1.0, (1 - u)^2 = 1.0
  `, "All five losses are functions of the margin u = y·wᵀx. The table shows the squared loss charging 1 for a confidently correct u = 2, and the checks confirm that hinge, squared and rescaled logistic losses lie above the zero-one loss while the perceptron loss does not."),
  "mlt-neural-architecture": py(r`
    import math

    def relu(z):
        return max(0.0, z)

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    w, b = [0.5, -1.0], 0.2
    for x in ([2, 1], [1, 2], [0, 0]):
        z = sum(wi * xi for wi, xi in zip(w, x)) + b          # pre-activation
        print(f"x = {x}: z = {z:+.2f}, ReLU(z) = {relu(z):.3f}, sigmoid(z) = {sigmoid(z):.3f}")

    # number of parameters of a fully connected network
    def count(sizes):
        return sum(n_out * n_in + n_out for n_in, n_out in zip(sizes, sizes[1:]))

    for sizes in ([3, 4, 2, 1], [2, 3, 1], [784, 128, 64, 10]):
        print(f"layers {sizes}: {count(sizes):,} parameters")

    # without a non-linearity two layers collapse into one linear map
    W1, W2 = [[1, 2], [0, 1]], [[1, -1]]
    W = [[sum(W2[i][k] * W1[k][j] for k in range(2)) for j in range(2)] for i in range(1)]
    print("W2 W1 =", W, "-> a single linear layer")
  `, r`
    x = [2, 1]: z = +0.20, ReLU(z) = 0.200, sigmoid(z) = 0.550
    x = [1, 2]: z = -1.30, ReLU(z) = 0.000, sigmoid(z) = 0.214
    x = [0, 0]: z = +0.20, ReLU(z) = 0.200, sigmoid(z) = 0.550
    layers [3, 4, 2, 1]: 29 parameters
    layers [2, 3, 1]: 13 parameters
    layers [784, 128, 64, 10]: 109,386 parameters
    W2 W1 = [[1, 1]] -> a single linear layer
  `, "A neuron computes a weighted sum plus a bias and passes it through an activation; a negative pre-activation switches a ReLU unit off. Parameters are counted layer by layer, and multiplying two weight matrices shows why stacked linear layers gain nothing without a non-linearity."),
  "mlt-forward-propagation": py(r`
    import math

    x = [2.0, 1.0]
    W1 = [[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]]     # 3 hidden units x 2 inputs
    b1 = [0.2, -0.3, 0.1]
    W2 = [1.0, -1.0, 0.5]                           # 1 output x 3 hidden units
    b2 = 0.1
    y = 1.0

    z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(W1, b1)]
    a1 = [max(0.0, z) for z in z1]                  # ReLU
    z2 = sum(w * a for w, a in zip(W2, a1)) + b2
    print("z1 =", [round(z, 4) for z in z1])
    print("a1 =", [round(a, 4) for a in a1])
    print("z2 =", round(z2, 4))

    # regression: linear output, squared loss
    print(f"regression:     y_hat = {z2:.4f}, loss = {0.5 * (z2 - y) ** 2:.4f}")

    # binary classification: sigmoid output, cross-entropy loss
    p = 1 / (1 + math.exp(-z2))
    print(f"classification: y_hat = {p:.4f}, loss = {-(y * math.log(p) + (1 - y) * math.log(1 - p)):.4f}")
  `, r`
    z1 = [0.2, 2.2, 0.1]
    a1 = [0.2, 2.2, 0.1]
    z2 = -1.85
    regression:     y_hat = -1.8500, loss = 4.0613
    classification: y_hat = 0.1359, loss = 1.9960
  `, "The input is transformed layer by layer: an affine map, then ReLU, then another affine map. The same final pre-activation is read as a real-valued prediction for regression or squashed by a sigmoid into a probability for classification, each with its own loss."),
  "mlt-backpropagation": py(r`
    x = [2.0, 1.0]
    W1 = [[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]]
    b1 = [0.2, -0.3, 0.1]
    W2 = [1.0, -1.0, 0.5]
    b2 = 0.1
    y = 1.0

    def forward(W1, b1, W2, b2):
        z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(W1, b1)]
        a1 = [max(0.0, z) for z in z1]
        yhat = sum(w * a for w, a in zip(W2, a1)) + b2            # linear output (regression)
        return z1, a1, yhat, 0.5 * (yhat - y) ** 2

    z1, a1, yhat, loss = forward(W1, b1, W2, b2)
    delta2 = yhat - y                                             # dL/dz2
    gW2 = [delta2 * a for a in a1]                                # delta2 * a1^T
    delta1 = [W2[j] * delta2 * (1.0 if z1[j] > 0 else 0.0) for j in range(3)]   # (W2^T delta2) * relu'(z1)
    gW1 = [[d * xi for xi in x] for d in delta1]                  # delta1 * x^T
    print(f"loss = {loss:.4f}, delta2 = {delta2:.4f}")
    print("dL/dW2 =", [round(g, 4) for g in gW2])
    print("delta1 =", [round(d, 4) for d in delta1])
    print("dL/dW1 =", [[round(g, 4) for g in row] for row in gW1])

    # numerical check of one entry by central differences
    h = 1e-6
    Wp = [row[:] for row in W1]; Wp[1][0] += h
    Wm = [row[:] for row in W1]; Wm[1][0] -= h
    numeric = (forward(Wp, b1, W2, b2)[3] - forward(Wm, b1, W2, b2)[3]) / (2 * h)
    print(f"check dL/dW1[1][0]: backprop {gW1[1][0]:.4f}, finite difference {numeric:.4f}")

    # one gradient-descent step lowers the loss
    eta = 0.01
    W1n = [[w - eta * g for w, g in zip(rw, rg)] for rw, rg in zip(W1, gW1)]
    W2n = [w - eta * g for w, g in zip(W2, gW2)]
    print(f"loss after one step with eta = {eta}: {forward(W1n, [b - eta * d for b, d in zip(b1, delta1)], W2n, b2 - eta * delta2)[3]:.4f}")
  `, r`
    loss = 4.0613, delta2 = -2.8500
    dL/dW2 = [-0.57, -6.27, -0.285]
    delta1 = [-2.85, 2.85, -1.425]
    dL/dW1 = [[-5.7, -2.85], [5.7, 2.85], [-2.85, -1.425]]
    check dL/dW1[1][0]: backprop 5.7000, finite difference 5.7000
    loss after one step with eta = 0.01: 2.6609
  `, "The output error ŷ − y is propagated back through W₂ and the ReLU derivative to give δ⁽¹⁾; every weight gradient is then an error times an activation. A finite-difference check agrees with the backpropagated value, and one gradient step reduces the loss."),
};
