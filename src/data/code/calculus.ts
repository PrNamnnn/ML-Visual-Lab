import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

export const calculusCode: Record<string, CodeExample> = {
  functions: ex(r`
    import math
    f = lambda x: math.exp(x)
    g = lambda x: math.log(x)        # inverse of exp, defined only for x > 0
    for x in [0.5, 1.0, 2.0]:
        print(f"x={x}: g(f(x)) = {g(f(x)):.4f}")
    try:
        g(0)
    except ValueError:
        print("log(0) is undefined")
  `, r`
    fn main() {
        let f = |x: f64| x.exp();
        let g = |x: f64| x.ln(); // inverse of exp, defined only for x > 0
        for x in [0.5, 1.0, 2.0] {
            println!("x={:.1}: g(f(x)) = {:.4}", x, g(f(x)));
        }
        // Rust returns -inf instead of raising an error, so we check the domain ourselves
        let x = 0.0_f64;
        if x <= 0.0 { println!("log(0) is undefined"); }
    }
  `, r`
    x=0.5: g(f(x)) = 0.5000
    x=1.0: g(f(x)) = 1.0000
    x=2.0: g(f(x)) = 2.0000
    log(0) is undefined
  `, "Composing a function with its inverse returns the input: ln(eˣ) = x. The domain matters — ln is only defined for positive inputs, which is why ML code adds a small ε before taking logs."),

  limits: ex(r`
    import math
    f = math.sin
    x = 1.0
    for h in [1.0, 0.1, 0.01, 0.001]:
        slope = (f(x + h) - f(x)) / h          # secant slope
        print(f"h={h:g}: slope={slope:.6f}")   # :g prints 1.0 as 1, like Rust
    print(f"limit (cos 1) = {math.cos(1.0):.6f}")
  `, r`
    fn main() {
        let f = |x: f64| x.sin();
        let x = 1.0_f64;
        for h in [1.0, 0.1, 0.01, 0.001] {
            let slope = (f(x + h) - f(x)) / h; // secant slope
            println!("h={}: slope={:.6}", h, slope);
        }
        println!("limit (cos 1) = {:.6}", 1.0_f64.cos());
    }
  `, r`
    h=1: slope=0.067826
    h=0.1: slope=0.497364
    h=0.01: slope=0.536086
    h=0.001: slope=0.539881
    limit (cos 1) = 0.540302
  `, "As h shrinks the secant slope approaches the limit cos(1) = 0.5403 — the definition of the derivative. Very small h eventually loses accuracy to floating-point round-off."),

  derivatives: ex(r`
    f = lambda x: 0.25 * x ** 3 - x
    df = lambda x: 0.75 * x ** 2 - 1          # power rule
    x = 2.0
    h = 1e-5
    numeric = (f(x + h) - f(x - h)) / (2 * h)  # central difference
    print(f"f'(2) exact   = {df(x):.6f}")
    print(f"f'(2) numeric = {numeric:.6f}")
    print(f"tangent: y = {f(x):.1f} + {df(x):.1f}(x - 2)")
  `, r`
    fn main() {
        let f = |x: f64| 0.25 * x.powi(3) - x;
        let df = |x: f64| 0.75 * x * x - 1.0; // power rule
        let x = 2.0;
        let h = 1e-5;
        let numeric = (f(x + h) - f(x - h)) / (2.0 * h); // central difference
        println!("f'(2) exact   = {:.6}", df(x));
        println!("f'(2) numeric = {:.6}", numeric);
        println!("tangent: y = {:.1} + {:.1}(x - 2)", f(x), df(x));
    }
  `, r`
    f'(2) exact   = 2.000000
    f'(2) numeric = 2.000000
    tangent: y = 0.0 + 2.0(x - 2)
  `, "The power rule gives f′(x) = 0.75x² − 1, so f′(2) = 2. A central difference (f(x+h) − f(x−h))/2h confirms it numerically — the same trick used for gradient checking."),

  "chain-rule": ex(r`
    import math
    x = 1.2
    u = x ** 2                 # inner function
    du_dx = 2 * x
    dy_du = math.cos(u)        # derivative of sin at u
    print(f"dy/dx = cos(x^2) * 2x = {dy_du * du_dx:.4f}")
    h = 1e-6
    print(f"numeric check       = {(math.sin((x + h) ** 2) - math.sin((x - h) ** 2)) / (2 * h):.4f}")
  `, r`
    fn main() {
        let x = 1.2_f64;
        let u = x * x; // inner function
        let du_dx = 2.0 * x;
        let dy_du = u.cos(); // derivative of sin at u
        println!("dy/dx = cos(x^2) * 2x = {:.4}", dy_du * du_dx);
        let h = 1e-6;
        println!("numeric check       = {:.4}", (((x + h).powi(2)).sin() - ((x - h).powi(2)).sin()) / (2.0 * h));
    }
  `, r`
    dy/dx = cos(x^2) * 2x = 0.3130
    numeric check       = 0.3130
  `, "Outer derivative at the inner value times inner derivative: cos(1.44)·2.4 ≈ 0.313. Backpropagation applies this rule layer after layer."),

  "second-derivative": ex(r`
    f = lambda x: 0.25 * x ** 3 - x
    d1 = lambda x: 0.75 * x ** 2 - 1
    d2 = lambda x: 1.5 * x
    for x in [-1.1547, 0.0, 1.1547]:
        kind = "min" if d2(x) > 0 else "max" if d2(x) < 0 else "inflection"
        print(f"x={x:+.4f}  f'={d1(x):+.4f}  f''={d2(x):+.4f}  -> {kind}")
  `, r`
    fn main() {
        let d1 = |x: f64| 0.75 * x * x - 1.0;
        let d2 = |x: f64| 1.5 * x;
        for x in [-1.1547_f64, 0.0, 1.1547] {
            let kind = if d2(x) > 0.0 { "min" } else if d2(x) < 0.0 { "max" } else { "inflection" };
            println!("x={:+.4}  f'={:+.4}  f''={:+.4}  -> {}", x, d1(x), d2(x), kind);
        }
    }
  `, r`
    x=-1.1547  f'=-0.0000  f''=-1.7321  -> max
    x=+0.0000  f'=-1.0000  f''=+0.0000  -> inflection
    x=+1.1547  f'=-0.0000  f''=+1.7321  -> min
  `, "At x = ±2/√3 ≈ ±1.1547 the slope is (almost exactly) 0; the sign of f″ decides max vs min. f″ = 0 at x = 0 marks the inflection point where curvature flips."),

  "implicit-differentiation": ex(r`
    import math
    a, b = 2.2, 1.4
    t = 0.9
    x, y = a * math.cos(t), b * math.sin(t)          # a point on x^2/a^2 + y^2/b^2 = 1
    implicit = -(b * b * x) / (a * a * y)
    explicit = -(b * x) / (a * math.sqrt(a * a - x * x))   # upper branch y = b sqrt(1 - x^2/a^2)
    print(f"dy/dx implicit = {implicit:.6f}")
    print(f"dy/dx explicit = {explicit:.6f}")
  `, r`
    fn main() {
        let (a, b) = (2.2_f64, 1.4_f64);
        let t = 0.9_f64;
        let (x, y) = (a * t.cos(), b * t.sin()); // a point on x^2/a^2 + y^2/b^2 = 1
        let implicit = -(b * b * x) / (a * a * y);
        let explicit = -(b * x) / (a * (a * a - x * x).sqrt()); // upper branch
        println!("dy/dx implicit = {:.6}", implicit);
        println!("dy/dx explicit = {:.6}", explicit);
    }
  `, r`
    dy/dx implicit = -0.504987
    dy/dx explicit = -0.504987
  `, "Differentiating x²/a² + y²/b² = 1 implicitly gives dy/dx = −b²x/(a²y) without solving for y; it agrees with differentiating the explicit upper branch."),

  "partial-derivatives": ex(r`
    f = lambda x, y: x ** 2 + 2 * y ** 2
    x, y, h = 1.2, 0.8, 1e-6
    fx = (f(x + h, y) - f(x - h, y)) / (2 * h)    # hold y fixed
    fy = (f(x, y + h) - f(x, y - h)) / (2 * h)    # hold x fixed
    print(f"df/dx = {fx:.4f} (exact {2 * x:.4f})")
    print(f"df/dy = {fy:.4f} (exact {4 * y:.4f})")
  `, r`
    fn main() {
        let f = |x: f64, y: f64| x * x + 2.0 * y * y;
        let (x, y, h) = (1.2, 0.8, 1e-6);
        let fx = (f(x + h, y) - f(x - h, y)) / (2.0 * h); // hold y fixed
        let fy = (f(x, y + h) - f(x, y - h)) / (2.0 * h); // hold x fixed
        println!("df/dx = {:.4} (exact {:.4})", fx, 2.0 * x);
        println!("df/dy = {:.4} (exact {:.4})", fy, 4.0 * y);
    }
  `, r`
    df/dx = 2.4000 (exact 2.4000)
    df/dy = 3.2000 (exact 3.2000)
  `, "A partial derivative nudges one input and freezes the others. For f = x² + 2y² that gives 2x and 4y."),

  gradient: ex(r`
    f = lambda x, y: x ** 2 + 2 * y ** 2
    grad = lambda x, y: (2 * x, 4 * y)
    x, y, lr = 1.0, 1.0, 0.1
    for step in range(3):
        gx, gy = grad(x, y)
        x, y = x - lr * gx, y - lr * gy          # move against the gradient
        print(f"step {step + 1}: ({x:.4f}, {y:.4f})  f = {f(x, y):.4f}")
  `, r`
    fn main() {
        let f = |x: f64, y: f64| x * x + 2.0 * y * y;
        let grad = |x: f64, y: f64| (2.0 * x, 4.0 * y);
        let (mut x, mut y, lr) = (1.0, 1.0, 0.1);
        for step in 0..3 {
            let (gx, gy) = grad(x, y);
            x -= lr * gx; // move against the gradient
            y -= lr * gy;
            println!("step {}: ({:.4}, {:.4})  f = {:.4}", step + 1, x, y, f(x, y));
        }
    }
  `, r`
    step 1: (0.8000, 0.6000)  f = 1.3600
    step 2: (0.6400, 0.3600)  f = 0.6688
    step 3: (0.5120, 0.2160)  f = 0.3555
  `, "∇f = (2x, 4y) points uphill; stepping along −∇f lowers f every step. The steeper y-direction shrinks faster."),

  "directional-derivative": ex(r`
    import math
    gx, gy = 2.4, 3.2                       # gradient of x^2 + 2y^2 at (1.2, 0.8)
    for deg in [0, 53.13, 90, 180]:
        ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        print(f"theta={deg:>6}: D_u f = {gx * ux + gy * uy:+.4f}")
    print(f"max = |grad| = {math.hypot(gx, gy):.4f}")
  `, r`
    fn main() {
        let (gx, gy) = (2.4_f64, 3.2_f64); // gradient of x^2 + 2y^2 at (1.2, 0.8)
        let degs = [("0", 0.0_f64), ("53.13", 53.13), ("90", 90.0), ("180", 180.0)];
        for (label, deg) in degs {
            let (ux, uy) = (deg.to_radians().cos(), deg.to_radians().sin());
            println!("theta={:>6}: D_u f = {:+.4}", label, gx * ux + gy * uy);
        }
        println!("max = |grad| = {:.4}", gx.hypot(gy));
    }
  `, r`
    theta=     0: D_u f = +2.4000
    theta= 53.13: D_u f = +4.0000
    theta=    90: D_u f = +3.2000
    theta=   180: D_u f = -2.4000
    max = |grad| = 4.0000
  `, "D_u f = ∇f·u. It peaks (= ‖∇f‖ = 4) when u points along the gradient (θ ≈ 53.13°) and is negative when moving the opposite way."),

  "level-curves": ex(r`
    import math
    # level curve x^2 + 2y^2 = 2 is an ellipse; gradient (2x, 4y) is perpendicular to it
    for t in [0.3, 1.0, 2.0]:
        x, y = math.sqrt(2) * math.cos(t), math.sin(t)
        tx, ty = -math.sqrt(2) * math.sin(t), math.cos(t)      # tangent direction
        gx, gy = 2 * x, 4 * y
        print(f"t={t:g}: f={x * x + 2 * y * y:.4f}, grad . tangent = {abs(gx * tx + gy * ty):.4f}")
  `, r`
    fn main() {
        // level curve x^2 + 2y^2 = 2 is an ellipse; gradient (2x, 4y) is perpendicular to it
        for t in [0.3_f64, 1.0, 2.0] {
            let (x, y) = (2.0_f64.sqrt() * t.cos(), t.sin());
            let (tx, ty) = (-(2.0_f64.sqrt()) * t.sin(), t.cos()); // tangent direction
            let (gx, gy) = (2.0 * x, 4.0 * y);
            println!("t={}: f={:.4}, grad . tangent = {:.4}", t, x * x + 2.0 * y * y, (gx * tx + gy * ty).abs());
        }
    }
  `, r`
    t=0.3: f=2.0000, grad . tangent = 0.0000
    t=1: f=2.0000, grad . tangent = 0.0000
    t=2: f=2.0000, grad . tangent = 0.0000
  `, "Every point of the parametrised ellipse has f = 2 (same level), and the gradient is orthogonal to the curve's tangent (dot product 0)."),

  surfaces: ex(r`
    import math
    f = lambda x, y: math.sin(x) * math.cos(y)
    for y in [0.0, 1.0]:
        row = [f"{f(x, y):+.3f}" for x in [-1.5, 0.0, 1.5]]
        print(f"y={y:.1f}:", " ".join(row))
  `, r`
    fn main() {
        let f = |x: f64, y: f64| x.sin() * y.cos();
        for y in [0.0_f64, 1.0] {
            let row: Vec<String> = [-1.5_f64, 0.0, 1.5].iter().map(|&x| format!("{:+.3}", f(x, y))).collect();
            println!("y={:.1}: {}", y, row.join(" "));
        }
    }
  `, r`
    y=0.0: -0.997 +0.000 +0.997
    y=1.0: -0.539 +0.000 +0.539
  `, "Evaluating z = f(x, y) on a grid gives the heights of the surface — exactly what a 3-D surface or contour plot draws."),

  hessian: ex(r`
    import math
    # f = x^2 - y^2 + 0.1 y^4 ; Hessian at (0, 0)
    fxx, fxy, fyy = 2.0, 0.0, -2.0
    tr, det = fxx + fyy, fxx * fyy - fxy * fxy
    disc = math.sqrt(tr * tr - 4 * det)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    print("eigenvalues:", l1, l2)
    print("saddle point" if l1 > 0 > l2 else "not a saddle")
  `, r`
    fn main() {
        // f = x^2 - y^2 + 0.1 y^4 ; Hessian at (0, 0)
        let (fxx, fxy, fyy) = (2.0_f64, 0.0_f64, -2.0_f64);
        let (tr, det) = (fxx + fyy, fxx * fyy - fxy * fxy);
        let disc = (tr * tr - 4.0 * det).sqrt();
        let (l1, l2) = ((tr + disc) / 2.0, (tr - disc) / 2.0);
        println!("eigenvalues: {:.1} {:.1}", l1, l2);
        println!("{}", if l1 > 0.0 && 0.0 > l2 { "saddle point" } else { "not a saddle" });
    }
  `, r`
    eigenvalues: 2.0 -2.0
    saddle point
  `, "The Hessian's eigenvalues are curvatures along its eigen-directions. One positive and one negative means the origin is a saddle: up in x, down in y."),

  jacobian: ex(r`
    import math
    k, x, y = 0.5, 0.6, 0.4
    f = lambda x, y: (x + k * math.sin(y), y + k * math.sin(x))
    J = [[1.0, k * math.cos(y)], [k * math.cos(x), 1.0]]
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    h = 1e-6     # numeric column 1: d f / d x
    c1 = [(f(x + h, y)[i] - f(x - h, y)[i]) / (2 * h) for i in range(2)]
    print(f"J = [[{J[0][0]:.4f}, {J[0][1]:.4f}], [{J[1][0]:.4f}, {J[1][1]:.4f}]]")
    print(f"numeric column 1 = [{c1[0]:.4f}, {c1[1]:.4f}]")
    print(f"det J = {det:.4f}")
  `, r`
    fn main() {
        let (k, x, y) = (0.5_f64, 0.6_f64, 0.4_f64);
        let f = |x: f64, y: f64| [x + k * y.sin(), y + k * x.sin()];
        let j = [[1.0, k * y.cos()], [k * x.cos(), 1.0]];
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let h = 1e-6; // numeric column 1: d f / d x
        let c1: Vec<f64> = (0..2).map(|i| (f(x + h, y)[i] - f(x - h, y)[i]) / (2.0 * h)).collect();
        println!("J = [[{:.4}, {:.4}], [{:.4}, {:.4}]]", j[0][0], j[0][1], j[1][0], j[1][1]);
        println!("numeric column 1 = [{:.4}, {:.4}]", c1[0], c1[1]);
        println!("det J = {:.4}", det);
    }
  `, r`
    J = [[1.0000, 0.4605], [0.4127, 1.0000]]
    numeric column 1 = [1.0000, 0.4127]
    det J = 0.8100
  `, "Column j of the Jacobian is how all outputs change when input j moves. det J ≈ 0.81 means a tiny square near (0.6, 0.4) shrinks to 81% of its area."),

  "multivariable-chain-rule": ex(r`
    import math
    t = 1.2
    x, y = t ** 2, math.sin(t)
    dz_dt = y * 2 * t + x * math.cos(t)      # path through x  +  path through y
    z = lambda t: t ** 2 * math.sin(t)
    h = 1e-6
    print(f"chain rule: {dz_dt:.4f}")
    print(f"numeric:    {(z(t + h) - z(t - h)) / (2 * h):.4f}")
  `, r`
    fn main() {
        let t = 1.2_f64;
        let (x, y) = (t * t, t.sin());
        let dz_dt = y * 2.0 * t + x * t.cos(); // path through x  +  path through y
        let z = |t: f64| t * t * t.sin();
        let h = 1e-6;
        println!("chain rule: {:.4}", dz_dt);
        println!("numeric:    {:.4}", (z(t + h) - z(t - h)) / (2.0 * h));
    }
  `, r`
    chain rule: 2.7587
    numeric:    2.7587
  `, "z = x·y with x = t², y = sin t: t affects z through two paths, so dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt) = y·2t + x·cos t."),

  "total-differential": ex(r`
    f = lambda x, y: x ** 2 + 2 * y ** 2
    x, y = 1.0, 1.0
    for step in [0.1, 0.01]:
        dx, dy = step, -step
        df = 2 * x * dx + 4 * y * dy                 # linear prediction
        actual = f(x + dx, y + dy) - f(x, y)
        print(f"step={step}: df={df:.4f} actual={actual:.4f} error={abs(actual - df):.4f}")
  `, r`
    fn main() {
        let f = |x: f64, y: f64| x * x + 2.0 * y * y;
        let (x, y) = (1.0, 1.0);
        for step in [0.1_f64, 0.01] {
            let (dx, dy) = (step, -step);
            let df = 2.0 * x * dx + 4.0 * y * dy; // linear prediction
            let actual = f(x + dx, y + dy) - f(x, y);
            println!("step={}: df={:.4} actual={:.4} error={:.4}", step, df, actual, (actual - df).abs());
        }
    }
  `, r`
    step=0.1: df=-0.2000 actual=-0.1700 error=0.0300
    step=0.01: df=-0.0200 actual=-0.0197 error=0.0003
  `, "df = f_x dx + f_y dy predicts the change; the error falls 100× when the step falls 10× (it is second-order)."),

  "riemann-sums": ex(r`
    f = lambda x: x ** 2 - 2 * x
    a, b = 0.0, 3.0
    exact = (b ** 3 / 3 - b ** 2) - (a ** 3 / 3 - a ** 2)
    for n in [4, 40, 400]:
        w = (b - a) / n
        mid = sum(f(a + (i + 0.5) * w) for i in range(n)) * w     # midpoint rule
        print(f"n={n}: {mid:.6f}")
    print(f"exact: {exact:.6f}")
  `, r`
    fn main() {
        let f = |x: f64| x * x - 2.0 * x;
        let (a, b) = (0.0_f64, 3.0_f64);
        let exact = (b.powi(3) / 3.0 - b * b) - (a.powi(3) / 3.0 - a * a);
        for n in [4, 40, 400] {
            let w = (b - a) / n as f64;
            let mid: f64 = (0..n).map(|i| f(a + (i as f64 + 0.5) * w)).sum::<f64>() * w; // midpoint rule
            println!("n={}: {:.6}", n, mid);
        }
        println!("exact: {:.6}", exact);
    }
  `, r`
    n=4: -0.140625
    n=40: -0.001406
    n=400: -0.000014
    exact: 0.000000
  `, "Midpoint rectangles converge to the exact integral (0 here — positive and negative areas cancel). The error shrinks 100× when n grows 10×."),

  "fundamental-theorem": ex(r`
    import math
    F = lambda x: -math.cos(x)                 # antiderivative of sin
    a, b = 0.0, math.pi
    print(f"F(pi) - F(0) = {F(b) - F(a):.6f}")
    n = 1000
    w = (b - a) / n
    print(f"numeric area = {sum(math.sin(a + (i + 0.5) * w) for i in range(n)) * w:.6f}")
  `, r`
    fn main() {
        let big_f = |x: f64| -x.cos(); // antiderivative of sin
        let (a, b) = (0.0_f64, std::f64::consts::PI);
        println!("F(pi) - F(0) = {:.6}", big_f(b) - big_f(a));
        let n = 1000;
        let w = (b - a) / n as f64;
        let area: f64 = (0..n).map(|i| (a + (i as f64 + 0.5) * w).sin()).sum::<f64>() * w;
        println!("numeric area = {:.6}", area);
    }
  `, r`
    F(pi) - F(0) = 2.000000
    numeric area = 2.000001
  `, "∫₀^π sin x dx = F(π) − F(0) with F = −cos: exactly 2. The midpoint sum with 1000 rectangles is accurate to 6 decimals."),

  "integration-probability": ex(r`
    import math
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))     # standard normal CDF
    for k in [1, 2, 3]:
        print(f"P(|Z| < {k}) = {Phi(k) - Phi(-k):.4f}")
  `, r`
    // Rust's std has no erf, so we integrate the density numerically (Simpson's rule)
    fn pdf(z: f64) -> f64 { (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt() }
    fn integrate(a: f64, b: f64) -> f64 {
        let n = 2000;
        let h = (b - a) / n as f64;
        let mut s = pdf(a) + pdf(b);
        for i in 1..n { s += pdf(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }; }
        s * h / 3.0
    }
    fn main() {
        for k in [1, 2, 3] {
            println!("P(|Z| < {}) = {:.4}", k, integrate(-(k as f64), k as f64));
        }
    }
  `, r`
    P(|Z| < 1) = 0.6827
    P(|Z| < 2) = 0.9545
    P(|Z| < 3) = 0.9973
  `, "Probabilities are areas under the density: the 68–95–99.7 rule. Python uses the closed-form erf; Rust integrates the density with Simpson's rule — both give the same 4 decimals."),

  "double-integrals": ex(r`
    f = lambda x, y: x ** 2 + y
    a = 2.0
    for n in [5, 50, 500]:
        w = a / n
        s = sum(f((i + 0.5) * w, (j + 0.5) * w) for i in range(n) for j in range(n)) * w * w
        print(f"n={n}: {s:.6f}")
    print(f"exact: {a ** 4 / 3 + a ** 3 / 2:.6f}")
  `, r`
    fn main() {
        let f = |x: f64, y: f64| x * x + y;
        let a = 2.0_f64;
        for n in [5, 50, 500] {
            let w = a / n as f64;
            let mut s = 0.0;
            for i in 0..n { for j in 0..n { s += f((i as f64 + 0.5) * w, (j as f64 + 0.5) * w); } }
            println!("n={}: {:.6}", n, s * w * w);
        }
        println!("exact: {:.6}", a.powi(4) / 3.0 + a.powi(3) / 2.0);
    }
  `, r`
    n=5: 9.280000
    n=50: 9.332800
    n=500: 9.333328
    exact: 9.333333
  `, "A 2-D midpoint sum over an n×n grid converges to ∬(x² + y) dA = 28/3. Notice the cost grows like n² — the curse of dimensionality."),

  "computational-graphs": ex(r`
    import math
    w, x, b, t = 0.8, 1.5, -0.4, 1.0
    # forward pass
    z = w * x + b
    a = 1 / (1 + math.exp(-z))
    L = (a - t) ** 2
    # backward pass: upstream gradient x local derivative
    dL_da = 2 * (a - t)
    dL_dz = dL_da * a * (1 - a)
    print(f"L = {L:.6f}")
    print(f"dL/dw = {dL_dz * x:.6f}, dL/db = {dL_dz:.6f}, dL/dx = {dL_dz * w:.6f}")
  `, r`
    fn main() {
        let (w, x, b, t) = (0.8_f64, 1.5_f64, -0.4_f64, 1.0_f64);
        // forward pass
        let z = w * x + b;
        let a = 1.0 / (1.0 + (-z).exp());
        let l = (a - t).powi(2);
        // backward pass: upstream gradient x local derivative
        let dl_da = 2.0 * (a - t);
        let dl_dz = dl_da * a * (1.0 - a);
        println!("L = {:.6}", l);
        println!("dL/dw = {:.6}, dL/db = {:.6}, dL/dx = {:.6}", dl_dz * x, dl_dz, dl_dz * w);
    }
  `, r`
    L = 0.096116
    dL/dw = -0.198952, dL/db = -0.132635, dL/dx = -0.106108
  `, "Forward pass stores z, a, L; backward pass multiplies the upstream gradient by each node's local derivative. One sweep gives all three gradients — reverse-mode autodiff."),

  "taylor-approximation": ex(r`
    import math
    x = 1.0
    approx, term = 0.0, 1.0
    for k in range(8):
        approx += term                 # add x^k / k!
        print(f"order {k}: {approx:.6f}")
        term *= x / (k + 1)
    print(f"e^1 = {math.exp(1):.6f}")
  `, r`
    fn main() {
        let x = 1.0_f64;
        let (mut approx, mut term) = (0.0_f64, 1.0_f64);
        for k in 0..8 {
            approx += term; // add x^k / k!
            println!("order {}: {:.6}", k, approx);
            term *= x / (k as f64 + 1.0);
        }
        println!("e^1 = {:.6}", 1.0_f64.exp());
    }
  `, r`
    order 0: 1.000000
    order 1: 2.000000
    order 2: 2.500000
    order 3: 2.666667
    order 4: 2.708333
    order 5: 2.716667
    order 6: 2.718056
    order 7: 2.718254
    e^1 = 2.718282
  `, "Each term xᵏ/k! of the Maclaurin series of eˣ brings the polynomial closer to e. Gradient descent uses the order-1 model, Newton's method the order-2 model."),

  "loss-surface-gd": ex(r`
    f = lambda x, y: (1 - x) ** 2 + 100 * (y - x * x) ** 2          # Rosenbrock
    grad = lambda x, y: (-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x))
    x, y, lr = -1.5, 2.2, 0.0015
    for it in range(1, 5001):
        gx, gy = grad(x, y)
        x, y = x - lr * gx, y - lr * gy
        if it in (1, 100, 1000, 5000):
            print(f"iter {it}: ({x:.4f}, {y:.4f}) f={f(x, y):.6f}")
  `, r`
    fn main() {
        let f = |x: f64, y: f64| (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2); // Rosenbrock
        let grad = |x: f64, y: f64| (-2.0 * (1.0 - x) - 400.0 * x * (y - x * x), 200.0 * (y - x * x));
        let (mut x, mut y, lr) = (-1.5_f64, 2.2_f64, 0.0015);
        for it in 1..=5000 {
            let (gx, gy) = grad(x, y);
            x -= lr * gx;
            y -= lr * gy;
            if [1, 100, 1000, 5000].contains(&it) {
                println!("iter {}: ({:.4}, {:.4}) f={:.6}", it, x, y, f(x, y));
            }
        }
    }
  `, r`
    iter 1: (-1.4475, 2.2150) f=7.424113
    iter 100: (-1.0771, 1.1680) f=4.320497
    iter 1000: (0.5160, 0.2639) f=0.234824
    iter 5000: (0.9769, 0.9541) f=0.000537
  `, "Plain gradient descent quickly finds Rosenbrock's curved valley but crawls along it toward the minimum at (1, 1) — the motivation for momentum and Adam."),

  "saddle-points": ex(r`
    grad = lambda x, y: (2 * x, -2 * y + 0.4 * y ** 3)      # f = x^2 - y^2 + 0.1 y^4
    x, y, lr = -2.5, 0.02, 0.05
    for it in range(1, 201):
        gx, gy = grad(x, y)
        x, y = x - lr * gx, y - lr * gy
        if it in (10, 50, 100, 200):
            print(f"iter {it}: x={x:+.4f} y={y:+.4f}")
  `, r`
    fn main() {
        let grad = |x: f64, y: f64| (2.0 * x, -2.0 * y + 0.4 * y.powi(3)); // f = x^2 - y^2 + 0.1 y^4
        let (mut x, mut y, lr) = (-2.5_f64, 0.02_f64, 0.05);
        for it in 1..=200 {
            let (gx, gy) = grad(x, y);
            x -= lr * gx;
            y -= lr * gy;
            if [10, 50, 100, 200].contains(&it) {
                println!("iter {}: x={:+.4} y={:+.4}", it, x, y);
            }
        }
    }
  `, r`
    iter 10: x=-0.8717 y=+0.0519
    iter 50: x=-0.0129 y=+1.6610
    iter 100: x=-0.0001 y=+2.2361
    iter 200: x=-0.0000 y=+2.2361
  `, "Starting almost on the ridge (y = 0.02), the iterates first slide toward the saddle at the origin, then the tiny y-component grows and they escape to the true minimum y = √5 ≈ 2.2361."),

  "convex-functions": ex(r`
    import math
    def jensen_violations(f, lo=-2.5, hi=2.5, n=60):
        count = 0
        xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
        for x1 in xs:
            for x2 in xs:
                m = 0.5 * (x1 + x2)
                if f(m) > 0.5 * (f(x1) + f(x2)) + 1e-12:   # chord below the curve
                    count += 1
        return count

    print("x^2:", jensen_violations(lambda x: x * x))
    print("x^2 - 2cos(3x):", jensen_violations(lambda x: x * x - 2 * math.cos(3 * x)) > 0)
  `, r`
    fn jensen_violations(f: impl Fn(f64) -> f64) -> usize {
        let (lo, hi, n) = (-2.5_f64, 2.5_f64, 60);
        let xs: Vec<f64> = (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect();
        let mut count = 0;
        for &x1 in &xs {
            for &x2 in &xs {
                let m = 0.5 * (x1 + x2);
                if f(m) > 0.5 * (f(x1) + f(x2)) + 1e-12 { count += 1; } // chord below the curve
            }
        }
        count
    }

    fn main() {
        println!("x^2: {}", jensen_violations(|x| x * x));
        let cap = |b: bool| if b { "True" } else { "False" };
        println!("x^2 - 2cos(3x): {}", cap(jensen_violations(|x| x * x - 2.0 * (3.0 * x).cos()) > 0));
    }
  `, r`
    x^2: 0
    x^2 - 2cos(3x): True
  `, "Check the midpoint form of Jensen's inequality f((x₁+x₂)/2) ≤ (f(x₁)+f(x₂))/2 on a grid of pairs: never violated for x² (convex), violated for the wavy function (non-convex)."),
};
