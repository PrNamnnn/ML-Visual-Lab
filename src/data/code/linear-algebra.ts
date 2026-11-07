import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

export const linearAlgebraCode: Record<string, CodeExample> = {
  vectors: ex(r`
    # a house described by three features: (area m², bedrooms, age in years)
    house = [120, 3, 15]
    print("dimension:", len(house))
    print("area:", house[0], "bedrooms:", house[1], "age:", house[2])
  `, r`
    fn main() {
        // a house described by three features: (area m², bedrooms, age in years)
        let house = [120, 3, 15];
        println!("dimension: {}", house.len());
        println!("area: {} bedrooms: {} age: {}", house[0], house[1], house[2]);
    }
  `, r`
    dimension: 3
    area: 120 bedrooms: 3 age: 15
  `, "A vector is just an ordered list of numbers; each position (component) means something. len gives the dimension n of the space ℝⁿ the vector lives in."),

  "vector-addition": ex(r`
    u = [3, 1]
    v = [1, 2]
    s = [u[i] + v[i] for i in range(2)]   # component-wise sum
    d = [u[i] - v[i] for i in range(2)]   # component-wise difference
    print("u + v =", s)
    print("u - v =", d)
  `, r`
    fn main() {
        let u = [3, 1];
        let v = [1, 2];
        let s = [u[0] + v[0], u[1] + v[1]]; // component-wise sum
        let d = [u[0] - v[0], u[1] - v[1]]; // component-wise difference
        println!("u + v = {:?}", s);
        println!("u - v = {:?}", d);
    }
  `, r`
    u + v = [4, 3]
    u - v = [2, -1]
  `, "Each output component is (uᵢ + vᵢ) or (uᵢ − vᵢ), exactly the formula u + v = (u₁+v₁, u₂+v₂). Gradient-descent updates are this operation on millions of components."),

  "vector-addition-3d": ex(r`
    u = [2.0, 0.5, 0.5]
    v = [0.5, 2.0, 1.0]
    s = [a + b for a, b in zip(u, v)]
    length = sum(x * x for x in s) ** 0.5
    print("u + v =", s)
    print(f"|u + v| = {length:.4f}")
  `, r`
    fn main() {
        let u = [2.0, 0.5, 0.5];
        let v = [0.5, 2.0, 1.0];
        let s: Vec<f64> = u.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
        let length = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("u + v = [{}, {}, {}]", s[0], s[1], s[2]);
        println!("|u + v| = {:.4}", length);
    }
  `, r`
    u + v = [2.5, 2.5, 1.5]
    |u + v| = 3.8406
  `, "zip pairs matching components; the length is √(2.5² + 2.5² + 1.5²) = √14.75. The same code works for any dimension."),

  "scalar-multiplication": ex(r`
    v = [3, 1]
    for c in [2, -1, 0.5]:
        print(c, "* v =", [c * x for x in v])
  `, r`
    fn main() {
        let v = [3.0, 1.0];
        for c in [2.0, -1.0, 0.5] {
            println!("{} * v = [{}, {}]", c, c * v[0], c * v[1]);
        }
    }
  `, r`
    2 * v = [6, 2]
    -1 * v = [-3, -1]
    0.5 * v = [1.5, 0.5]
  `, "Every component is multiplied by c. c = −1 reverses the direction, 0.5 halves the length. This is what the learning rate does to a gradient."),

  "magnitude-and-unit-vectors": ex(r`
    v = [3.0, 4.0]
    norm = (v[0] ** 2 + v[1] ** 2) ** 0.5
    unit = [x / norm for x in v]
    print(f"|v| = {norm:.1f}")
    print(f"unit = [{unit[0]:.1f}, {unit[1]:.1f}]")
  `, r`
    fn main() {
        let v = [3.0_f64, 4.0];
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let unit = [v[0] / norm, v[1] / norm];
        println!("|v| = {:.1}", norm);
        println!("unit = [{:.1}, {:.1}]", unit[0], unit[1]);
    }
  `, r`
    |v| = 5.0
    unit = [0.6, 0.8]
  `, "‖v‖ = √(3² + 4²) = 5 (Pythagoras). Dividing by the length keeps the direction and makes the length 1 — how embeddings are normalised before cosine search."),

  "dot-product": ex(r`
    u = [2, 1]
    v = [1, 3]
    d = sum(a * b for a, b in zip(u, v))
    print("u . v =", d)
  `, r`
    fn main() {
        let u = [2, 1];
        let v = [1, 3];
        let d: i32 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        println!("u . v = {}", d);
    }
  `, r`
    u . v = 5
  `, "Multiply matching components and add: 2·1 + 1·3 = 5. A neuron's pre-activation w·x + b and an attention score q·k are exactly this."),

  "cross-product": ex(r`
    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    u, v = [1, 0, 0], [0, 1, 0]
    print("e1 x e2 =", cross(u, v))
    print("e2 x e1 =", cross(v, u))
  `, r`
    fn cross(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
        [a[1] * b[2] - a[2] * b[1],
         a[2] * b[0] - a[0] * b[2],
         a[0] * b[1] - a[1] * b[0]]
    }

    fn main() {
        let (u, v) = ([1, 0, 0], [0, 1, 0]);
        println!("e1 x e2 = {:?}", cross(u, v));
        println!("e2 x e1 = {:?}", cross(v, u));
    }
  `, r`
    e1 x e2 = [0, 0, 1]
    e2 x e1 = [0, 0, -1]
  `, "Each component is a 2×2 determinant of the other two coordinates. Swapping the order flips the sign (anti-commutativity)."),

  "angle-between-vectors": ex(r`
    import math
    u, v = [2.0, 1.0], [1.0, 3.0]
    dot = u[0] * v[0] + u[1] * v[1]
    cos_t = dot / (math.hypot(*u) * math.hypot(*v))
    cos_t = max(-1.0, min(1.0, cos_t))      # guard against rounding
    print(f"cos(theta) = {cos_t:.4f}")
    print(f"theta = {math.degrees(math.acos(cos_t)):.1f} degrees")
  `, r`
    fn main() {
        let (u, v) = ([2.0_f64, 1.0], [1.0_f64, 3.0]);
        let dot = u[0] * v[0] + u[1] * v[1];
        let mut cos_t = dot / (u[0].hypot(u[1]) * v[0].hypot(v[1]));
        cos_t = cos_t.clamp(-1.0, 1.0); // guard against rounding
        println!("cos(theta) = {:.4}", cos_t);
        println!("theta = {:.1} degrees", cos_t.acos().to_degrees());
    }
  `, r`
    cos(theta) = 0.7071
    theta = 45.0 degrees
  `, "cos θ = u·v / (‖u‖‖v‖) = 5 / (√5·√10) = 1/√2, so θ = 45°. Clamping protects acos from values like 1.0000000002."),

  projection: ex(r`
    u, v = [3.0, 4.0], [1.0, 0.0]
    coef = (u[0] * v[0] + u[1] * v[1]) / (v[0] ** 2 + v[1] ** 2)
    proj = [coef * x for x in v]
    resid = [u[i] - proj[i] for i in range(2)]
    print("proj =", proj)
    print("residual =", resid)
    print("residual . v =", resid[0] * v[0] + resid[1] * v[1])
  `, r`
    fn main() {
        let (u, v) = ([3.0_f64, 4.0], [1.0_f64, 0.0]);
        let coef = (u[0] * v[0] + u[1] * v[1]) / (v[0] * v[0] + v[1] * v[1]);
        let proj = [coef * v[0], coef * v[1]];
        let resid = [u[0] - proj[0], u[1] - proj[1]];
        println!("proj = [{:.1}, {:.1}]", proj[0], proj[1]);
        println!("residual = [{:.1}, {:.1}]", resid[0], resid[1]);
        println!("residual . v = {:.1}", resid[0] * v[0] + resid[1] * v[1]);
    }
  `, r`
    proj = [3.0, 0.0]
    residual = [0.0, 4.0]
    residual . v = 0.0
  `, "proj_v(u) = (u·v / v·v)·v. The residual u − proj is perpendicular to v (dot product 0) — the same fact makes least-squares residuals orthogonal to every feature."),

  "orthogonal-vectors": ex(r`
    u, v = [2, 1], [-1, 2]
    d = u[0] * v[0] + u[1] * v[1]
    print("u . v =", d, "-> orthogonal" if d == 0 else "-> not orthogonal")
    s = [u[0] + v[0], u[1] + v[1]]
    print("|u+v|^2 =", s[0] ** 2 + s[1] ** 2, "=", (u[0]**2 + u[1]**2) + (v[0]**2 + v[1]**2))
  `, r`
    fn main() {
        let (u, v) = ([2, 1], [-1, 2]);
        let d = u[0] * v[0] + u[1] * v[1];
        println!("u . v = {} {}", d, if d == 0 { "-> orthogonal" } else { "-> not orthogonal" });
        let s = [u[0] + v[0], u[1] + v[1]];
        println!("|u+v|^2 = {} = {}", s[0] * s[0] + s[1] * s[1], (u[0] * u[0] + u[1] * u[1]) + (v[0] * v[0] + v[1] * v[1]));
    }
  `, r`
    u . v = 0 -> orthogonal
    |u+v|^2 = 10 = 10
  `, "A zero dot product means perpendicular, and then Pythagoras holds: ‖u+v‖² = ‖u‖² + ‖v‖² (here 10 = 5 + 5)."),

  "linear-combinations": ex(r`
    a, b = [2.0, 0.5], [0.5, 1.5]
    c1, c2 = 1.2, 0.8
    w = [c1 * a[i] + c2 * b[i] for i in range(2)]
    print(f"w = [{w[0]:.2f}, {w[1]:.2f}]")
  `, r`
    fn main() {
        let (a, b) = ([2.0, 0.5], [0.5, 1.5]);
        let (c1, c2) = (1.2, 0.8);
        let w = [c1 * a[0] + c2 * b[0], c1 * a[1] + c2 * b[1]];
        println!("w = [{:.2}, {:.2}]", w[0], w[1]);
    }
  `, r`
    w = [2.80, 1.80]
  `, "w = 1.2·(2, 0.5) + 0.8·(0.5, 1.5) = (2.4 + 0.4, 0.6 + 1.2). A matrix–vector product Ax is a linear combination of A's columns with weights x."),

  "linear-independence": ex(r`
    def det2(a, b):            # columns a and b
        return a[0] * b[1] - a[1] * b[0]

    print("(2,1),(1,3):", det2([2, 1], [1, 3]))   # non-zero -> independent
    print("(2,1),(4,2):", det2([2, 1], [4, 2]))   # zero -> dependent
  `, r`
    fn det2(a: [i32; 2], b: [i32; 2]) -> i32 {
        a[0] * b[1] - a[1] * b[0]
    }

    fn main() {
        println!("(2,1),(1,3): {}", det2([2, 1], [1, 3])); // non-zero -> independent
        println!("(2,1),(4,2): {}", det2([2, 1], [4, 2])); // zero -> dependent
    }
  `, r`
    (2,1),(1,3): 5
    (2,1),(4,2): 0
  `, "Two vectors in ℝ² are independent exactly when the matrix with them as columns has non-zero determinant. (4, 2) = 2·(2, 1), so that pair is dependent."),

  "span-and-basis": ex(r`
    # coordinates of x in the basis {a, b}: solve c1*a + c2*b = x (Cramer's rule)
    a, b, x = [2.0, 0.5], [0.5, 1.5], [2.5, 2.5]
    det = a[0] * b[1] - a[1] * b[0]
    c1 = (x[0] * b[1] - x[1] * b[0]) / det
    c2 = (a[0] * x[1] - a[1] * x[0]) / det
    print(f"c1 = {c1:.4f}, c2 = {c2:.4f}")
  `, r`
    fn main() {
        // coordinates of x in the basis {a, b}: solve c1*a + c2*b = x (Cramer's rule)
        let (a, b, x) = ([2.0, 0.5], [0.5, 1.5], [2.5, 2.5]);
        let det = a[0] * b[1] - a[1] * b[0];
        let c1 = (x[0] * b[1] - x[1] * b[0]) / det;
        let c2 = (a[0] * x[1] - a[1] * x[0]) / det;
        println!("c1 = {:.4}, c2 = {:.4}", c1, c2);
    }
  `, r`
    c1 = 0.9091, c2 = 1.3636
  `, "Because a and b are independent (det = 2.75 ≠ 0) every x has unique coordinates in that basis: x = 0.9091·a + 1.3636·b."),

  "vector-spaces": ex(r`
    def on_line(p, c):        # the set y = 2x + c
        return abs(p[1] - (2 * p[0] + c)) < 1e-12

    for c in [0, 1]:
        p, q = [1, 2 + c], [3, 6 + c]           # two members
        s = [p[0] + q[0], p[1] + q[1]]
        print(f"c={c}: zero in set={on_line([0, 0], c)}, p+q in set={on_line(s, c)}")
  `, r`
    fn on_line(p: [f64; 2], c: f64) -> bool {
        // the set y = 2x + c
        (p[1] - (2.0 * p[0] + c)).abs() < 1e-12
    }

    fn main() {
        for c in [0.0, 1.0] {
            let p = [1.0, 2.0 + c];
            let q = [3.0, 6.0 + c];
            let s = [p[0] + q[0], p[1] + q[1]];
            let cap = |b: bool| if b { "True" } else { "False" };
            println!("c={}: zero in set={}, p+q in set={}", c, cap(on_line([0.0, 0.0], c)), cap(on_line(s, c)));
        }
    }
  `, r`
    c=0: zero in set=True, p+q in set=True
    c=1: zero in set=False, p+q in set=False
  `, "The line y = 2x passes the subspace tests; shifting it by c = 1 breaks both (no zero vector, not closed under addition)."),

  norms: ex(r`
    x = [3.0, -4.0]
    l1 = sum(abs(v) for v in x)
    l2 = sum(v * v for v in x) ** 0.5
    linf = max(abs(v) for v in x)
    print(f"L1 = {l1}, L2 = {l2}, Linf = {linf}")
  `, r`
    fn main() {
        let x = [3.0_f64, -4.0];
        let l1: f64 = x.iter().map(|v| v.abs()).sum();
        let l2 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let linf = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("L1 = {:.1}, L2 = {:.1}, Linf = {:.1}", l1, l2, linf);
    }
  `, r`
    L1 = 7.0, L2 = 5.0, Linf = 4.0
  `, "L1 adds absolute values, L2 is Euclidean length, L∞ takes the largest |component|; note L∞ ≤ L2 ≤ L1. L1 and L2 penalties are Lasso and Ridge regularisation."),

  "distance-metrics": ex(r`
    a, b = [0.0, 0.0], [3.0, 4.0]
    diff = [abs(a[i] - b[i]) for i in range(2)]
    print("manhattan =", sum(diff))
    print("euclidean =", sum(d * d for d in diff) ** 0.5)
    print("chebyshev =", max(diff))
  `, r`
    fn main() {
        let (a, b) = ([0.0_f64, 0.0], [3.0_f64, 4.0]);
        let diff = [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()];
        println!("manhattan = {:.1}", diff[0] + diff[1]);
        println!("euclidean = {:.1}", (diff[0] * diff[0] + diff[1] * diff[1]).sqrt());
        println!("chebyshev = {:.1}", diff[0].max(diff[1]));
    }
  `, r`
    manhattan = 7.0
    euclidean = 5.0
    chebyshev = 4.0
  `, "All three are norms of the difference vector a − b. KNN and k-means results depend on which one you choose — and on feature scaling."),

  "gram-schmidt": ex(r`
    import math
    a1, a2 = [3.0, 1.0], [2.0, 2.0]
    n1 = math.hypot(*a1)
    q1 = [x / n1 for x in a1]
    c = a2[0] * q1[0] + a2[1] * q1[1]               # component of a2 along q1
    u2 = [a2[i] - c * q1[i] for i in range(2)]       # remove it
    n2 = math.hypot(*u2)
    q2 = [x / n2 for x in u2]
    print(f"q1 = [{q1[0]:.4f}, {q1[1]:.4f}]")
    print(f"q2 = [{q2[0]:.4f}, {q2[1]:.4f}]")
    print(f"q1 . q2 = {abs(q1[0] * q2[0] + q1[1] * q2[1]):.4f}")
  `, r`
    fn main() {
        let (a1, a2) = ([3.0_f64, 1.0], [2.0_f64, 2.0]);
        let n1 = a1[0].hypot(a1[1]);
        let q1 = [a1[0] / n1, a1[1] / n1];
        let c = a2[0] * q1[0] + a2[1] * q1[1]; // component of a2 along q1
        let u2 = [a2[0] - c * q1[0], a2[1] - c * q1[1]]; // remove it
        let n2 = u2[0].hypot(u2[1]);
        let q2 = [u2[0] / n2, u2[1] / n2];
        println!("q1 = [{:.4}, {:.4}]", q1[0], q1[1]);
        println!("q2 = [{:.4}, {:.4}]", q2[0], q2[1]);
        println!("q1 . q2 = {:.4}", (q1[0] * q2[0] + q1[1] * q2[1]).abs());
    }
  `, r`
    q1 = [0.9487, 0.3162]
    q2 = [-0.3162, 0.9487]
    q1 . q2 = 0.0000
  `, "q₁ = a₁/‖a₁‖; then subtract a₂'s projection onto q₁ and normalise. The result is an orthonormal basis — the Q in A = QR."),

  matrices: ex(r`
    A = [[1, 2, 3],
         [4, 5, 6]]
    print("shape:", len(A), "x", len(A[0]))
    print("row 2:", A[1])
    print("column 3:", [row[2] for row in A])
    print("a_12 =", A[0][1])
  `, r`
    fn main() {
        let a = [[1, 2, 3],
                 [4, 5, 6]];
        println!("shape: {} x {}", a.len(), a[0].len());
        println!("row 2: {:?}", a[1]);
        let col: Vec<i32> = a.iter().map(|row| row[2]).collect();
        println!("column 3: {:?}", col);
        println!("a_12 = {}", a[0][1]);
    }
  `, r`
    shape: 2 x 3
    row 2: [4, 5, 6]
    column 3: [3, 6]
    a_12 = 2
  `, "A matrix is a list of rows. Indexing is A[row][column] (0-based in code, 1-based in maths), so a₁₂ is A[0][1]."),

  "matrix-addition": ex(r`
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    c = 2
    C = [[A[i][j] + c * B[i][j] for j in range(2)] for i in range(2)]
    print("A + 2B =", C)
  `, r`
    fn main() {
        let a = [[1, 2], [3, 4]];
        let b = [[5, 6], [7, 8]];
        let c = 2;
        let mut out = [[0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] = a[i][j] + c * b[i][j];
            }
        }
        println!("A + 2B = {:?}", out);
    }
  `, r`
    A + 2B = [[11, 14], [17, 20]]
  `, "Entry (i, j) of the result only uses entry (i, j) of each input: (A + cB)ᵢⱼ = aᵢⱼ + c·bᵢⱼ. Both matrices must have the same shape."),

  "matrix-multiplication": ex(r`
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    n, m, p = len(A), len(B), len(B[0])
    C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
    print("AB =", C)
  `, r`
    fn main() {
        let a = [[1, 2], [3, 4]];
        let b = [[5, 6], [7, 8]];
        let mut c = [[0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    c[i][j] += a[i][k] * b[k][j]; // row i of A · column j of B
                }
            }
        }
        println!("AB = {:?}", c);
    }
  `, r`
    AB = [[19, 22], [43, 50]]
  `, "cᵢⱼ = Σₖ aᵢₖ bₖⱼ: row i of A dotted with column j of B (e.g. 1·5 + 2·7 = 19). Three nested loops = the 2mnp operation count of a dense layer."),

  "matrix-vector-multiplication": ex(r`
    A = [[1, 2, 0], [3, -1, 2]]
    x = [2, -1, 1]
    y = [sum(a * b for a, b in zip(row, x)) for row in A]     # row view
    print("Ax =", y)
    cols = [[A[0][j], A[1][j]] for j in range(3)]             # column view
    z = [sum(x[j] * cols[j][i] for j in range(3)) for i in range(2)]
    print("x1*a1 + x2*a2 + x3*a3 =", z)
  `, r`
    fn main() {
        let a = [[1, 2, 0], [3, -1, 2]];
        let x = [2, -1, 1];
        // row view: each output is a dot product
        let y: Vec<i32> = a.iter().map(|row| row.iter().zip(x.iter()).map(|(p, q)| p * q).sum()).collect();
        println!("Ax = {:?}", y);
        // column view: weighted sum of columns
        let mut z = vec![0; 2];
        for j in 0..3 {
            for i in 0..2 {
                z[i] += x[j] * a[i][j];
            }
        }
        println!("x1*a1 + x2*a2 + x3*a3 = {:?}", z);
    }
  `, r`
    Ax = [0, 9]
    x1*a1 + x2*a2 + x3*a3 = [0, 9]
  `, "Both views give the same answer: row i · x, or the columns of A combined with weights x. A linear layer computes Wx this way."),

  "linear-transformations": ex(r`
    import math
    t = math.radians(30)
    R = [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]   # rotation by 30 degrees
    e1 = [1.0, 0.0]
    Re1 = [R[0][0] * e1[0] + R[0][1] * e1[1], R[1][0] * e1[0] + R[1][1] * e1[1]]
    print(f"R e1 = [{Re1[0]:.4f}, {Re1[1]:.4f}]   (first column of R)")
    print(f"length = {math.hypot(*Re1):.4f}")
  `, r`
    fn main() {
        let t = 30.0_f64.to_radians();
        let r = [[t.cos(), -t.sin()], [t.sin(), t.cos()]]; // rotation by 30 degrees
        let e1 = [1.0, 0.0];
        let re1 = [r[0][0] * e1[0] + r[0][1] * e1[1], r[1][0] * e1[0] + r[1][1] * e1[1]];
        println!("R e1 = [{:.4}, {:.4}]   (first column of R)", re1[0], re1[1]);
        println!("length = {:.4}", re1[0].hypot(re1[1]));
    }
  `, r`
    R e1 = [0.8660, 0.5000]   (first column of R)
    length = 1.0000
  `, "A matrix sends e₁ to its first column. A rotation keeps lengths (1.0000), as orthogonal matrices always do."),

  transpose: ex(r`
    A = [[1, 2, 3], [4, 5, 6]]
    T = [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
    print("A^T =", T)
  `, r`
    fn main() {
        let a = [[1, 2, 3], [4, 5, 6]];
        let mut t = vec![vec![0; 2]; 3];
        for i in 0..2 {
            for j in 0..3 {
                t[j][i] = a[i][j]; // swap row and column index
            }
        }
        println!("A^T = {:?}", t);
    }
  `, r`
    A^T = [[1, 4], [2, 5], [3, 6]]
  `, "(Aᵀ)ⱼᵢ = aᵢⱼ: rows become columns and a 2×3 matrix becomes 3×2. The backward pass of y = Wx uses Wᵀ."),

  "special-matrices": ex(r`
    D = [[2, 0], [0, -1]]
    x = [3, 4]
    print("D x =", [D[0][0] * x[0], D[1][1] * x[1]])     # diagonal: scale each axis
    U = [[2, 1, 4], [0, 3, 5], [0, 0, 6]]
    print("det(U) = product of diagonal =", U[0][0] * U[1][1] * U[2][2])
    S = [[4, 1], [1, 3]]
    print("symmetric:", S[0][1] == S[1][0])
  `, r`
    fn main() {
        let d = [[2, 0], [0, -1]];
        let x = [3, 4];
        println!("D x = {:?}", [d[0][0] * x[0], d[1][1] * x[1]]); // diagonal: scale each axis
        let u = [[2, 1, 4], [0, 3, 5], [0, 0, 6]];
        println!("det(U) = product of diagonal = {}", u[0][0] * u[1][1] * u[2][2]);
        let s = [[4, 1], [1, 3]];
        println!("symmetric: {}", if s[0][1] == s[1][0] { "True" } else { "False" });
    }
  `, r`
    D x = [6, -4]
    det(U) = product of diagonal = 36
    symmetric: True
  `, "Diagonal matrices scale coordinates independently, triangular determinants are diagonal products, and symmetric means aᵢⱼ = aⱼᵢ (like every covariance matrix)."),

  "orthogonal-matrices": ex(r`
    import math
    t = math.radians(40)
    Q = [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
    QtQ = [[sum(Q[k][i] * Q[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    print("Q^T Q =", [[round(v, 10) + 0.0 for v in row] for row in QtQ])
    print(f"det Q = {Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]:.4f}")
  `, r`
    fn main() {
        let t = 40.0_f64.to_radians();
        let q = [[t.cos(), -t.sin()], [t.sin(), t.cos()]];
        let mut qtq = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    qtq[i][j] += q[k][i] * q[k][j];
                }
            }
        }
        let r10 = |v: f64| (v * 1e10).round() / 1e10 + 0.0; // + 0.0 turns -0.0 into 0.0
        println!("Q^T Q = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", r10(qtq[0][0]), r10(qtq[0][1]), r10(qtq[1][0]), r10(qtq[1][1]));
        println!("det Q = {:.4}", q[0][0] * q[1][1] - q[0][1] * q[1][0]);
    }
  `, r`
    Q^T Q = [[1.0, 0.0], [0.0, 1.0]]
    det Q = 1.0000
  `, "QᵀQ = I (up to rounding, so we round to 10 decimals) means the columns are orthonormal; det = +1 identifies a rotation."),

  trace: ex(r`
    A = [[0.9, 0.4], [-0.3, 0.8]]
    print("trace =", round(A[0][0] + A[1][1], 4))
    P = [[1.0, 0.0], [0.0, 1.0]]
    for _ in range(3):        # P = A^3
        P = [[sum(P[i][k] * A[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    print("A^3 =", [[round(v, 4) for v in row] for row in P])
  `, r`
    fn main() {
        let a = [[0.9, 0.4], [-0.3, 0.8]];
        println!("trace = {}", ((a[0][0] + a[1][1]) * 1e4_f64).round() / 1e4);
        let mut p = [[1.0_f64, 0.0], [0.0, 1.0]];
        for _ in 0..3 {
            // P = P * A
            let mut n = [[0.0; 2]; 2];
            for i in 0..2 { for j in 0..2 { for k in 0..2 { n[i][j] += p[i][k] * a[k][j]; } } }
            p = n;
        }
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        println!("A^3 = [[{}, {}], [{}, {}]]", r4(p[0][0]), r4(p[0][1]), r4(p[1][0]), r4(p[1][1]));
    }
  `, r`
    trace = 1.7
    A^3 = [[0.417, 0.82], [-0.615, 0.212]]
  `, "The trace is the sum of the diagonal (= sum of eigenvalues). Repeated multiplication gives powers; here the entries shrink because the eigenvalues have |λ| < 1."),

  "hadamard-product": ex(r`
    A = [[1, 2], [3, 4]]
    M = [[1, 0], [0, 1]]          # a mask that keeps only the diagonal
    H = [[A[i][j] * M[i][j] for j in range(2)] for i in range(2)]
    print("A (.) M =", H)
  `, r`
    fn main() {
        let a = [[1, 2], [3, 4]];
        let m = [[1, 0], [0, 1]]; // a mask that keeps only the diagonal
        let mut h = [[0; 2]; 2];
        for i in 0..2 { for j in 0..2 { h[i][j] = a[i][j] * m[i][j]; } }
        println!("A (.) M = {:?}", h);
    }
  `, r`
    A (.) M = [[1, 0], [0, 4]]
  `, "Element-wise multiplication (A ⊙ M)ᵢⱼ = aᵢⱼmᵢⱼ. Multiplying by a 0/1 mask is how dropout and attention masks switch entries off."),

  broadcasting: ex(r`
    X = [[1, 2, 3], [4, 5, 6]]   # 2 samples x 3 features
    b = [10, 20, 30]             # one bias per feature
    Y = [[x + bj for x, bj in zip(row, b)] for row in X]   # b reused for every row
    print(Y)
  `, r`
    fn main() {
        let x = [[1, 2, 3], [4, 5, 6]]; // 2 samples x 3 features
        let b = [10, 20, 30]; // one bias per feature
        let y: Vec<Vec<i32>> = x.iter().map(|row| row.iter().zip(b.iter()).map(|(v, bj)| v + bj).collect()).collect();
        println!("{:?}", y);
    }
  `, r`
    [[11, 22, 33], [14, 25, 36]]
  `, "Broadcasting repeats the 1×3 vector b for every row: yᵢⱼ = xᵢⱼ + bⱼ. NumPy/PyTorch do this implicitly for X + b."),

  "sparse-matrices": ex(r`
    A = [[5, 0, 0], [0, 0, 3], [2, 0, 1]]
    values, col_idx, row_ptr = [], [], [0]
    for row in A:
        for j, v in enumerate(row):
            if v != 0:
                values.append(v)
                col_idx.append(j)
        row_ptr.append(len(values))
    print("values =", values, "col_idx =", col_idx, "row_ptr =", row_ptr)
    x = [1, 2, 3]
    y = [sum(values[k] * x[col_idx[k]] for k in range(row_ptr[i], row_ptr[i + 1])) for i in range(3)]
    print("A x =", y)
  `, r`
    fn main() {
        let a = [[5, 0, 0], [0, 0, 3], [2, 0, 1]];
        let (mut values, mut col_idx, mut row_ptr) = (vec![], vec![], vec![0]);
        for row in a.iter() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0 {
                    values.push(v);
                    col_idx.push(j);
                }
            }
            row_ptr.push(values.len());
        }
        println!("values = {:?} col_idx = {:?} row_ptr = {:?}", values, col_idx, row_ptr);
        let x = [1, 2, 3];
        let y: Vec<i32> = (0..3).map(|i| (row_ptr[i]..row_ptr[i + 1]).map(|k| values[k] * x[col_idx[k]]).sum()).collect();
        println!("A x = {:?}", y);
    }
  `, r`
    values = [5, 3, 2, 1] col_idx = [0, 2, 0, 2] row_ptr = [0, 1, 2, 4]
    A x = [5, 9, 5]
  `, "CSR keeps only non-zeros. Row i's entries live in values[row_ptr[i]..row_ptr[i+1]], so A·x costs nnz = 4 multiplications instead of 9."),

  determinants: ex(r`
    A = [[2.0, 1.0], [0.0, 1.5]]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    print("det(A) =", det, "-> the unit square becomes area", abs(det))
  `, r`
    fn main() {
        let a = [[2.0_f64, 1.0], [0.0, 1.5]];
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        println!("det(A) = {:.1} -> the unit square becomes area {:.1}", det, det.abs());
    }
  `, r`
    det(A) = 3.0 -> the unit square becomes area 3.0
  `, "det = ad − bc = 2·1.5 − 1·0 = 3: the transformation triples every area. A negative value would mean orientation flips."),

  "determinants-3d": ex(r`
    def det3(m):
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    print("det =", det3([[6, 1, 1], [4, -2, 5], [2, 8, 7]]))
  `, r`
    fn det3(m: [[i32; 3]; 3]) -> i32 {
        let [a, b, c] = m[0];
        let [d, e, f] = m[1];
        let [g, h, i] = m[2];
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    fn main() {
        println!("det = {}", det3([[6, 1, 1], [4, -2, 5], [2, 8, 7]]));
    }
  `, r`
    det = -306
  `, "Cofactor expansion along the first row: 6(−14−40) − 1(28−10) + 1(32+4) = −306. |det| is the volume of the transformed unit cube."),

  "singular-matrices": ex(r`
    for t in [0.0, 0.5, 1.0]:
        # column 2 slides toward 2 x column 1 as t -> 1
        A = [[1.0, (1 - t) * (-1) + t * 2], [2.0, (1 - t) * 1 + t * 4]]
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        print(f"t={t}: det={det:.2f}", "(singular)" if abs(det) < 1e-12 else "(invertible)")
  `, r`
    fn main() {
        for t in [0.0_f64, 0.5, 1.0] {
            // column 2 slides toward 2 x column 1 as t -> 1
            let a = [[1.0, (1.0 - t) * -1.0 + t * 2.0], [2.0, (1.0 - t) * 1.0 + t * 4.0]];
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            println!("t={:.1}: det={:.2} {}", t, det, if det.abs() < 1e-12 { "(singular)" } else { "(invertible)" });
        }
    }
  `, r`
    t=0.0: det=3.00 (invertible)
    t=0.5: det=1.50 (invertible)
    t=1.0: det=0.00 (singular)
  `, "When the second column becomes a multiple of the first, the plane collapses onto a line and det = 0: no inverse exists."),

  "matrix-inverse": ex(r`
    a, b, c, d = 4.0, 7.0, 2.0, 6.0
    det = a * d - b * c
    inv = [[d / det, -b / det], [-c / det, a / det]]
    print("det =", det)
    print("A^-1 =", inv)
    I = [[a * inv[0][0] + b * inv[1][0], a * inv[0][1] + b * inv[1][1]],
         [c * inv[0][0] + d * inv[1][0], c * inv[0][1] + d * inv[1][1]]]
    print("A A^-1 =", [[round(v, 12) + 0.0 for v in row] for row in I])   # + 0.0 turns -0.0 into 0.0
  `, r`
    fn main() {
        let (a, b, c, d) = (4.0_f64, 7.0, 2.0, 6.0);
        let det = a * d - b * c;
        let inv = [[d / det, -b / det], [-c / det, a / det]];
        println!("det = {:.1}", det);
        println!("A^-1 = [[{}, {}], [{}, {}]]", inv[0][0], inv[0][1], inv[1][0], inv[1][1]);
        let i = [[a * inv[0][0] + b * inv[1][0], a * inv[0][1] + b * inv[1][1]],
                 [c * inv[0][0] + d * inv[1][0], c * inv[0][1] + d * inv[1][1]]];
        let r = |v: f64| (v * 1e12).round() / 1e12 + 0.0;
        println!("A A^-1 = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", r(i[0][0]), r(i[0][1]), r(i[1][0]), r(i[1][1]));
    }
  `, r`
    det = 10.0
    A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
    A A^-1 = [[1.0, 0.0], [0.0, 1.0]]
  `, "Swap a and d, negate b and c, divide by det = 10. Multiplying back gives I up to floating-point rounding (hence round to 12 decimals)."),

  "gaussian-elimination": ex(r`
    # solve 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3 by row reduction
    M = [[2.0, 1.0, -1.0, 8.0], [-3.0, -1.0, 2.0, -11.0], [-2.0, 1.0, 2.0, -3.0]]
    n = 3
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))   # partial pivoting
        M[k], M[p] = M[p], M[k]
        M[k] = [v / M[k][k] for v in M[k]]                  # scale pivot row to 1
        for i in range(n):
            if i != k:
                f = M[i][k]
                M[i] = [M[i][j] - f * M[k][j] for j in range(n + 1)]
    print("x, y, z =", [round(row[n], 6) for row in M])
  `, r`
    fn main() {
        // solve 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3 by row reduction
        let mut m = vec![vec![2.0_f64, 1.0, -1.0, 8.0], vec![-3.0, -1.0, 2.0, -11.0], vec![-2.0, 1.0, 2.0, -3.0]];
        let n = 3;
        for k in 0..n {
            let p = (k..n).max_by(|&a, &b| m[a][k].abs().partial_cmp(&m[b][k].abs()).unwrap()).unwrap(); // partial pivoting
            m.swap(k, p);
            let piv = m[k][k];
            for j in 0..=n { m[k][j] /= piv; } // scale pivot row to 1
            for i in 0..n {
                if i != k {
                    let f = m[i][k];
                    for j in 0..=n { m[i][j] -= f * m[k][j]; }
                }
            }
        }
        let sol: Vec<f64> = m.iter().map(|row| (row[n] * 1e6).round() / 1e6).collect();
        println!("x, y, z = [{:.1}, {:.1}, {:.1}]", sol[0], sol[1], sol[2]);
    }
  `, r`
    x, y, z = [2.0, 3.0, -1.0]
  `, "Swap the largest pivot up, scale it to 1, then eliminate that column from every other row. The last column ends up holding the solution."),

  "linear-systems": ex(r`
    # x + y = 3 and 2x - y = 0 (two lines in the plane)
    a, b, e = 1.0, 1.0, 3.0
    c, d, f = 2.0, -1.0, 0.0
    det = a * d - b * c
    if det == 0:
        print("no unique solution (parallel or identical lines)")
    else:
        print("x =", (e * d - b * f) / det, " y =", (a * f - e * c) / det)
  `, r`
    fn main() {
        // x + y = 3 and 2x - y = 0 (two lines in the plane)
        let (a, b, e) = (1.0_f64, 1.0, 3.0);
        let (c, d, f) = (2.0_f64, -1.0, 0.0);
        let det = a * d - b * c;
        if det == 0.0 {
            println!("no unique solution (parallel or identical lines)");
        } else {
            println!("x = {:.1}  y = {:.1}", (e * d - b * f) / det, (a * f - e * c) / det);
        }
    }
  `, r`
    x = 1.0  y = 2.0
  `, "Cramer's rule for a 2×2 system: the lines meet at (1, 2). det = 0 would mean parallel (no solution) or identical lines (infinitely many)."),

  rank: ex(r`
    def rank(A, eps=1e-10):
        M = [row[:] for row in A]
        r = 0
        for c in range(len(M[0])):
            p = max(range(r, len(M)), key=lambda i: abs(M[i][c]), default=None)
            if p is None or abs(M[p][c]) < eps:
                continue
            M[r], M[p] = M[p], M[r]
            for i in range(r + 1, len(M)):
                f = M[i][c] / M[r][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(len(M[0]))]
            r += 1
            if r == len(M):
                break
        return r

    print("rank =", rank([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]]))
  `, r`
    fn rank(a: &[Vec<f64>]) -> usize {
        let mut m = a.to_vec();
        let (rows, cols) = (m.len(), m[0].len());
        let mut r = 0;
        for c in 0..cols {
            if r == rows { break; }
            let p = (r..rows).max_by(|&x, &y| m[x][c].abs().partial_cmp(&m[y][c].abs()).unwrap()).unwrap();
            if m[p][c].abs() < 1e-10 { continue; }
            m.swap(r, p);
            for i in r + 1..rows {
                let f = m[i][c] / m[r][c];
                for j in 0..cols { m[i][j] -= f * m[r][j]; }
            }
            r += 1;
        }
        r
    }

    fn main() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![1.0, 0.0, 1.0]];
        println!("rank = {}", rank(&a));
    }
  `, r`
    rank = 2
  `, "Row 2 is 2 × row 1, so elimination leaves only two pivots: rank 2, and by rank–nullity the null space is 3 − 2 = 1-dimensional."),

  "lu-decomposition": ex(r`
    A = [[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]
    n = 3
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [row[:] for row in A]
    for k in range(n):                       # no pivoting needed for this A
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]       # store the multiplier
            U[i] = [U[i][j] - L[i][k] * U[k][j] for j in range(n)]
    print("L =", L)
    print("U =", U)
  `, r`
    fn main() {
        let a = [[2.0_f64, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]];
        let n = 3;
        let mut l = [[0.0_f64; 3]; 3];
        for i in 0..n { l[i][i] = 1.0; }
        let mut u = a;
        for k in 0..n { // no pivoting needed for this A
            for i in k + 1..n {
                l[i][k] = u[i][k] / u[k][k]; // store the multiplier
                for j in 0..n { u[i][j] -= l[i][k] * u[k][j]; }
            }
        }
        let show = |m: [[f64; 3]; 3]| format!("[[{:.1}, {:.1}, {:.1}], [{:.1}, {:.1}, {:.1}], [{:.1}, {:.1}, {:.1}]]", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
        println!("L = {}", show(l));
        println!("U = {}", show(u));
    }
  `, r`
    L = [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, -1.0, 1.0]]
    U = [[2.0, 1.0, 1.0], [0.0, -8.0, -2.0], [0.0, 0.0, 1.0]]
  `, "Gaussian elimination, but each multiplier is saved in L. U is the echelon form; L·U reproduces A, and det(A) = 2·(−8)·1 = −16."),

  "eigenvalues-eigenvectors": ex(r`
    import math
    a, b, c, d = 2.0, 1.0, 1.0, 2.0
    tr, det = a + d, a * d - b * c
    disc = math.sqrt(tr * tr - 4 * det)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    print("eigenvalues:", l1, l2)
    v = [1.0, 1.0]            # check A v = 3 v
    print("A v =", [a * v[0] + b * v[1], c * v[0] + d * v[1]])
  `, r`
    fn main() {
        let (a, b, c, d) = (2.0_f64, 1.0, 1.0, 2.0);
        let (tr, det) = (a + d, a * d - b * c);
        let disc = (tr * tr - 4.0 * det).sqrt();
        let (l1, l2) = ((tr + disc) / 2.0, (tr - disc) / 2.0);
        println!("eigenvalues: {:.1} {:.1}", l1, l2);
        let v = [1.0, 1.0]; // check A v = 3 v
        println!("A v = [{:.1}, {:.1}]", a * v[0] + b * v[1], c * v[0] + d * v[1]);
    }
  `, r`
    eigenvalues: 3.0 1.0
    A v = [3.0, 3.0]
  `, "Solve λ² − tr·λ + det = 0. For λ = 3 the vector (1, 1) is only stretched: A(1, 1) = (3, 3) = 3·(1, 1)."),

  diagonalization: ex(r`
    import math
    s = 1 / math.sqrt(2)
    Q = [[s, s], [s, -s]]           # eigenvectors of [[2,1],[1,2]] as columns
    L = [3.0, 1.0]                  # eigenvalues
    # A = Q diag(L) Q^T
    A = [[sum(Q[i][k] * L[k] * Q[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
    print("Q L Q^T =", [[round(v, 10) for v in row] for row in A])
  `, r`
    fn main() {
        let s = 1.0 / 2.0_f64.sqrt();
        let q = [[s, s], [s, -s]]; // eigenvectors of [[2,1],[1,2]] as columns
        let l = [3.0, 1.0]; // eigenvalues
        let mut a = [[0.0_f64; 2]; 2];
        for i in 0..2 { for j in 0..2 { for k in 0..2 { a[i][j] += q[i][k] * l[k] * q[j][k]; } } }
        let r = |v: f64| (v * 1e10).round() / 1e10;
        println!("Q L Q^T = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", r(a[0][0]), r(a[0][1]), r(a[1][0]), r(a[1][1]));
    }
  `, r`
    Q L Q^T = [[2.0, 1.0], [1.0, 2.0]]
  `, "For a symmetric matrix A = QΛQᵀ with orthonormal eigenvectors in Q. Rebuilding A from its eigen-pieces confirms the decomposition."),

  "positive-definite": ex(r`
    import math
    def classify(a, b, d):          # symmetric [[a, b], [b, d]]
        tr, det = a + d, a * d - b * b
        disc = math.sqrt(tr * tr - 4 * det)
        l1, l2 = (tr + disc) / 2, (tr - disc) / 2
        if l2 > 0: return "positive definite"
        if l2 >= 0: return "positive semidefinite"
        if l1 < 0: return "negative definite"
        return "indefinite (saddle)"

    print(classify(2, 0.5, 1))
    print(classify(1, 1, 1))
    print(classify(1, 0, -1))
  `, r`
    fn classify(a: f64, b: f64, d: f64) -> &'static str {
        // symmetric [[a, b], [b, d]]
        let (tr, det) = (a + d, a * d - b * b);
        let disc = (tr * tr - 4.0 * det).sqrt();
        let (l1, l2) = ((tr + disc) / 2.0, (tr - disc) / 2.0);
        if l2 > 0.0 { return "positive definite"; }
        if l2 >= 0.0 { return "positive semidefinite"; }
        if l1 < 0.0 { return "negative definite"; }
        "indefinite (saddle)"
    }

    fn main() {
        println!("{}", classify(2.0, 0.5, 1.0));
        println!("{}", classify(1.0, 1.0, 1.0));
        println!("{}", classify(1.0, 0.0, -1.0));
    }
  `, r`
    positive definite
    positive semidefinite
    indefinite (saddle)
  `, "The signs of the eigenvalues decide the shape of xᵀAx: all positive → bowl, a zero → flat valley, mixed → saddle."),

  svd: ex(r`
    import math
    A = [[3.0, 0.0], [4.0, 5.0]]
    # singular values are square roots of the eigenvalues of A^T A
    AtA = [[sum(A[k][i] * A[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    tr = AtA[0][0] + AtA[1][1]
    det = AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0]
    disc = math.sqrt(tr * tr - 4 * det)
    s1, s2 = math.sqrt((tr + disc) / 2), math.sqrt((tr - disc) / 2)
    print(f"sigma1 = {s1:.4f}, sigma2 = {s2:.4f}")
    print(f"sigma1 * sigma2 = {s1 * s2:.4f} = |det A| = {abs(A[0][0] * A[1][1] - A[0][1] * A[1][0]):.4f}")
  `, r`
    fn main() {
        let a = [[3.0_f64, 0.0], [4.0, 5.0]];
        // singular values are square roots of the eigenvalues of A^T A
        let mut ata = [[0.0_f64; 2]; 2];
        for i in 0..2 { for j in 0..2 { for k in 0..2 { ata[i][j] += a[k][i] * a[k][j]; } } }
        let tr = ata[0][0] + ata[1][1];
        let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
        let disc = (tr * tr - 4.0 * det).sqrt();
        let (s1, s2) = (((tr + disc) / 2.0).sqrt(), ((tr - disc) / 2.0).sqrt());
        println!("sigma1 = {:.4}, sigma2 = {:.4}", s1, s2);
        println!("sigma1 * sigma2 = {:.4} = |det A| = {:.4}", s1 * s2, (a[0][0] * a[1][1] - a[0][1] * a[1][0]).abs());
    }
  `, r`
    sigma1 = 6.7082, sigma2 = 2.2361
    sigma1 * sigma2 = 15.0000 = |det A| = 15.0000
  `, "AᵀA = [[25, 20], [20, 25]] has eigenvalues 45 and 5, so σ = √45 and √5. Their product equals |det A| because U and V are rotations."),

  "low-rank-approximation": ex(r`
    # A = 6 * u1 v1^T + 0.5 * u2 v2^T with orthonormal u's and v's (an SVD written out)
    u1, v1, s1 = [0.6, 0.8], [1.0, 0.0], 6.0
    u2, v2, s2 = [-0.8, 0.6], [0.0, 1.0], 0.5
    A = [[s1 * u1[i] * v1[j] + s2 * u2[i] * v2[j] for j in range(2)] for i in range(2)]
    A1 = [[s1 * u1[i] * v1[j] for j in range(2)] for i in range(2)]   # keep the largest only
    err = sum((A[i][j] - A1[i][j]) ** 2 for i in range(2) for j in range(2)) ** 0.5
    show = lambda M: "[[" + "], [".join(", ".join(f"{v:.1f}" for v in row) for row in M) + "]]"
    print("A   =", show(A))
    print("A_1 =", show(A1))
    print("||A - A_1||_F =", round(err, 4))
  `, r`
    fn main() {
        // A = 6 * u1 v1^T + 0.5 * u2 v2^T with orthonormal u's and v's (an SVD written out)
        let (u1, v1, s1) = ([0.6_f64, 0.8], [1.0, 0.0], 6.0);
        let (u2, v2, s2) = ([-0.8_f64, 0.6], [0.0, 1.0], 0.5);
        let mut a = [[0.0_f64; 2]; 2];
        let mut a1 = [[0.0_f64; 2]; 2];
        let mut err = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                a[i][j] = s1 * u1[i] * v1[j] + s2 * u2[i] * v2[j];
                a1[i][j] = s1 * u1[i] * v1[j]; // keep the largest only
                err += (a[i][j] - a1[i][j]).powi(2);
            }
        }
        println!("A   = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", a[0][0], a[0][1], a[1][0], a[1][1]);
        println!("A_1 = [[{:.1}, {:.1}], [{:.1}, {:.1}]]", a1[0][0], a1[0][1], a1[1][0], a1[1][1]);
        println!("||A - A_1||_F = {}", (err.sqrt() * 1e4).round() / 1e4);
    }
  `, r`
    A   = [[3.6, -0.4], [4.8, 0.3]]
    A_1 = [[3.6, 0.0], [4.8, 0.0]]
    ||A - A_1||_F = 0.5
  `, "Dropping the smaller singular component leaves an error of exactly σ₂ = 0.5 in Frobenius norm — the Eckart–Young theorem."),

  "change-of-basis": ex(r`
    # basis vectors as columns of B; coordinates c solve B c = x
    B = [[2.0, 1.0], [0.0, 1.0]]
    x = [5.0, 3.0]
    det = B[0][0] * B[1][1] - B[0][1] * B[1][0]
    c = [(x[0] * B[1][1] - B[0][1] * x[1]) / det, (B[0][0] * x[1] - B[1][0] * x[0]) / det]
    print("coordinates in basis B:", c)
    back = [B[0][0] * c[0] + B[0][1] * c[1], B[1][0] * c[0] + B[1][1] * c[1]]
    print("B c =", back)
  `, r`
    fn main() {
        // basis vectors as columns of B; coordinates c solve B c = x
        let b = [[2.0_f64, 1.0], [0.0, 1.0]];
        let x = [5.0, 3.0];
        let det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
        let c = [(x[0] * b[1][1] - b[0][1] * x[1]) / det, (b[0][0] * x[1] - b[1][0] * x[0]) / det];
        println!("coordinates in basis B: [{:.1}, {:.1}]", c[0], c[1]);
        let back = [b[0][0] * c[0] + b[0][1] * c[1], b[1][0] * c[0] + b[1][1] * c[1]];
        println!("B c = [{:.1}, {:.1}]", back[0], back[1]);
    }
  `, r`
    coordinates in basis B: [1.0, 3.0]
    B c = [5.0, 3.0]
  `, "The same point (5, 3) has coordinates (1, 3) in the basis {(2, 0), (1, 1)}: 1·(2, 0) + 3·(1, 1) = (5, 3)."),

  "covariance-matrix": ex(r`
    X = [[2.0, 1.0], [4.0, 3.0], [6.0, 2.0], [8.0, 6.0]]   # 4 samples, 2 features
    n = len(X)
    mean = [sum(r[j] for r in X) / n for j in range(2)]
    C = [[sum((r[i] - mean[i]) * (r[j] - mean[j]) for r in X) / (n - 1) for j in range(2)] for i in range(2)]
    print("mean =", mean)
    print("cov  =", [[round(v, 4) for v in row] for row in C])
  `, r`
    fn main() {
        let x = [[2.0_f64, 1.0], [4.0, 3.0], [6.0, 2.0], [8.0, 6.0]]; // 4 samples, 2 features
        let n = x.len() as f64;
        let mean = [x.iter().map(|r| r[0]).sum::<f64>() / n, x.iter().map(|r| r[1]).sum::<f64>() / n];
        let mut c = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                c[i][j] = x.iter().map(|r| (r[i] - mean[i]) * (r[j] - mean[j])).sum::<f64>() / (n - 1.0);
            }
        }
        let r = |v: f64| (v * 1e4).round() / 1e4;
        println!("mean = [{:.1}, {:.1}]", mean[0], mean[1]);
        println!("cov  = [[{}, {}], [{}, {}]]", r(c[0][0]), r(c[0][1]), r(c[1][0]), r(c[1][1]));
    }
  `, r`
    mean = [5.0, 3.0]
    cov  = [[6.6667, 4.6667], [4.6667, 4.6667]]
  `, "Centre each feature, multiply deviations pairwise and divide by n − 1. The diagonal holds variances, the off-diagonal the covariance (positive: both features rise together)."),

  "least-squares": ex(r`
    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 3.0, 5.0]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    w = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    b = my - w * mx
    resid = [y - (w * x + b) for x, y in zip(xs, ys)]
    print(f"w = {w:.4f}, b = {b:.4f}")
    print(f"sum of residuals * x = {abs(sum(r * x for r, x in zip(resid, xs))):.4f}")
  `, r`
    fn main() {
        let xs = [1.0_f64, 2.0, 3.0];
        let ys = [2.0_f64, 3.0, 5.0];
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let num: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
        let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let w = num / den;
        let b = my - w * mx;
        let dotx: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (y - (w * x + b)) * x).sum();
        println!("w = {:.4}, b = {:.4}", w, b);
        println!("sum of residuals * x = {:.4}", dotx.abs());
    }
  `, r`
    w = 1.5000, b = 0.3333
    sum of residuals * x = 0.0000
  `, "The least-squares line makes the residual vector orthogonal to the feature column (residual·x = 0) — the normal equations XᵀXw = Xᵀy in action."),

  "condition-number": ex(r`
    import math
    def cond(a, b, c, d):
        ata = [[a*a + c*c, a*b + c*d], [a*b + c*d, b*b + d*d]]
        tr = ata[0][0] + ata[1][1]
        det = ata[0][0] * ata[1][1] - ata[0][1] ** 2
        disc = math.sqrt(tr * tr - 4 * det)
        return math.sqrt((tr + disc) / (tr - disc))       # sigma_max / sigma_min

    print(f"well-conditioned:  {cond(1, 0, 0, 2):.2f}")
    print(f"ill-conditioned:   {cond(1, 1, 1, 1.0001):.0f}")
  `, r`
    fn cond(a: f64, b: f64, c: f64, d: f64) -> f64 {
        let ata = [[a * a + c * c, a * b + c * d], [a * b + c * d, b * b + d * d]];
        let tr = ata[0][0] + ata[1][1];
        let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[0][1];
        let disc = (tr * tr - 4.0 * det).sqrt();
        ((tr + disc) / (tr - disc)).sqrt() // sigma_max / sigma_min
    }

    fn main() {
        println!("well-conditioned:  {:.2}", cond(1.0, 0.0, 0.0, 2.0));
        println!("ill-conditioned:   {:.0}", cond(1.0, 1.0, 1.0, 1.0001));
    }
  `, r`
    well-conditioned:  2.00
    ill-conditioned:   40002
  `, "κ = σ_max/σ_min. Nearly parallel columns give κ ≈ 4·10⁴: relative input errors can be amplified about 40,000 times (≈ 4–5 lost digits)."),

  "inner-product-spaces": ex(r`
    W = [[2.0, 1.0], [1.0, 3.0]]        # symmetric positive definite weight matrix
    def ip(u, v):
        return sum(u[i] * W[i][j] * v[j] for i in range(2) for j in range(2))

    u, v = [1.0, 0.0], [0.0, 1.0]
    print("u . v   =", u[0] * v[0] + u[1] * v[1])
    print("<u,v>_W =", ip(u, v))
    print("||u||_W =", round(ip(u, u) ** 0.5, 4))
  `, r`
    fn main() {
        let w = [[2.0_f64, 1.0], [1.0, 3.0]]; // symmetric positive definite weight matrix
        let ip = |u: [f64; 2], v: [f64; 2]| -> f64 {
            let mut s = 0.0;
            for i in 0..2 { for j in 0..2 { s += u[i] * w[i][j] * v[j]; } }
            s
        };
        let (u, v) = ([1.0, 0.0], [0.0, 1.0]);
        println!("u . v   = {:.1}", u[0] * v[0] + u[1] * v[1]);
        println!("<u,v>_W = {:.1}", ip(u, v));
        println!("||u||_W = {}", (ip(u, u).sqrt() * 1e4).round() / 1e4);
    }
  `, r`
    u . v   = 0.0
    <u,v>_W = 1.0
    ||u||_W = 1.4142
  `, "uᵀWv changes the geometry: e₁ and e₂ are orthogonal for the dot product but not for W, and ‖e₁‖_W = √2. With W = Σ⁻¹ this is Mahalanobis distance."),
};
