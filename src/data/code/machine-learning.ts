import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";
import { dedent } from "@/lib/utilities/highlight";

/* Shared helpers, repeated inside snippets so every program runs on its own. */
const PY_LCG = `seed = 42
def rand():                      # tiny linear congruential generator (same in the Rust version)
    global seed
    seed = (seed * 1103515245 + 12345) % 2**31
    return seed / 2**31`;
const RS_LCG = `static mut SEED: u64 = 42;
fn rand() -> f64 {
    // tiny linear congruential generator (same in the Python version)
    unsafe {
        SEED = (SEED * 1103515245 + 12345) % (1u64 << 31);
        SEED as f64 / (1u64 << 31) as f64
    }
}`;
const join = (...parts: string[]) => parts.map(dedent).join("\n\n");

export const machineLearningCode: Record<string, CodeExample> = {
  "learning-paradigms": ex(r`
    xs = [0, 1, 2, 3, 10, 11, 12, 13]
    f = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]   # only the two end points are labelled
    for _ in range(50):                           # label propagation along nearest neighbours
        new = f[:]
        for i in range(1, len(xs) - 1):
            nb = sorted(range(len(xs)), key=lambda j: abs(xs[j] - xs[i]))[1:3]
            new[i] = sum(f[j] for j in nb) / 2
        f = new
    print("scores:", ", ".join(f"{v:.2f}" for v in f))
    print("labels:", [1 if v > 0.5 else 0 for v in f])
  `, r`
    fn main() {
        let xs = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0_f64];
        let mut f = vec![0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]; // only the two end points are labelled
        for _ in 0..50 {
            // label propagation along nearest neighbours
            let mut new = f.clone();
            for i in 1..xs.len() - 1 {
                let mut idx: Vec<usize> = (0..xs.len()).collect();
                idx.sort_by(|&a, &b| (xs[a] - xs[i]).abs().partial_cmp(&(xs[b] - xs[i]).abs()).unwrap());
                new[i] = (f[idx[1]] + f[idx[2]]) / 2.0;
            }
            f = new;
        }
        println!("scores: {}", f.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join(", "));
        println!("labels: {:?}", f.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect::<Vec<_>>());
    }
  `, r`
    scores: 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00
    labels: [0, 0, 0, 0, 1, 1, 1, 1]
  `, "Semi-supervised learning: only the two end points are labelled, yet averaging over nearest neighbours spreads each label through its own cluster. The unlabelled points 1–3 take label 0 and 10–12 take label 1."),

  "train-validation-test": ex(join(PY_LCG, r`
    idx = list(range(10))
    for i in range(len(idx) - 1, 0, -1):          # Fisher-Yates shuffle with the LCG
        j = int(rand() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    train, val, test = idx[:6], idx[6:8], idx[8:]
    print("train:", train)
    print("val  :", val)
    print("test :", test)
  `), join(RS_LCG, r`
    fn main() {
        let mut idx: Vec<usize> = (0..10).collect();
        for i in (1..idx.len()).rev() {
            // Fisher-Yates shuffle with the LCG
            let j = (rand() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        println!("train: {:?}", &idx[..6]);
        println!("val  : {:?}", &idx[6..8]);
        println!("test : {:?}", &idx[8..]);
    }
  `), r`
    train: [8, 7, 6, 1, 0, 2]
    val  : [9, 3]
    test : [4, 5]
  `, "Shuffle once, then cut: 60 % for fitting, 20 % for choosing hyperparameters, 20 % kept untouched for the final report. Fixing the seed makes the split reproducible."),

  "bias-variance": ex(join(PY_LCG, r`
    def fit_line(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        return my - b * mx, b
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    truth = lambda x: x * x
    x0 = 0.9
    preds = {0: [], 1: []}
    for _ in range(500):                           # many training sets with fresh noise
        ys = [truth(x) + (rand() - 0.5) * 0.4 for x in xs]
        preds[0].append(sum(ys) / len(ys))         # degree 0: predict the mean
        a, b = fit_line(xs, ys)
        preds[1].append(a + b * x0)                # degree 1: straight line
    for d, p in preds.items():
        m = sum(p) / len(p)
        var = sum((v - m) ** 2 for v in p) / len(p)
        print(f"degree {d}: bias^2={(m - truth(x0)) ** 2:.4f} variance={var:.4f}")
  `), join(RS_LCG, r`
    fn fit_line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let b = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
        (my - b * mx, b)
    }

    fn main() {
        let xs = [0.0, 0.25, 0.5, 0.75, 1.0_f64];
        let truth = |x: f64| x * x;
        let x0 = 0.9;
        let (mut p0, mut p1) = (Vec::new(), Vec::new());
        for _ in 0..500 {
            // many training sets with fresh noise
            let ys: Vec<f64> = xs.iter().map(|&x| truth(x) + (rand() - 0.5) * 0.4).collect();
            p0.push(ys.iter().sum::<f64>() / ys.len() as f64); // degree 0: predict the mean
            let (a, b) = fit_line(&xs, &ys);
            p1.push(a + b * x0); // degree 1: straight line
        }
        for (d, p) in [(0, &p0), (1, &p1)] {
            let m = p.iter().sum::<f64>() / p.len() as f64;
            let var = p.iter().map(|v| (v - m).powi(2)).sum::<f64>() / p.len() as f64;
            println!("degree {}: bias^2={:.4} variance={:.4}", d, (m - truth(x0)).powi(2), var);
        }
    }
  `), r`
    degree 0: bias^2=0.1887 variance=0.0024
    degree 1: bias^2=0.0014 variance=0.0058
  `, "Refitting on 500 noisy training sets shows the trade-off at x = 0.9: the constant model is very stable (low variance) but systematically wrong (high bias²); the line is closer on average but varies more from dataset to dataset."),

  "data-leakage": ex(r`
    train = [10.0, 12.0, 11.0, 13.0]
    test = [30.0, 32.0]                            # the future looks different
    mean_all = sum(train + test) / 6
    mean_train = sum(train) / 4
    print(f"scaler mean fitted on all data : {mean_all:.2f}  <- uses test information")
    print(f"scaler mean fitted on train only: {mean_train:.2f}")
    print("centred test (leaky):", ", ".join(f"{v - mean_all:.2f}" for v in test))
    print("centred test (clean):", ", ".join(f"{v - mean_train:.2f}" for v in test))
  `, r`
    fn main() {
        let train = [10.0, 12.0, 11.0, 13.0_f64];
        let test = [30.0, 32.0_f64]; // the future looks different
        let mean_all = (train.iter().sum::<f64>() + test.iter().sum::<f64>()) / 6.0;
        let mean_train = train.iter().sum::<f64>() / 4.0;
        println!("scaler mean fitted on all data : {:.2}  <- uses test information", mean_all);
        println!("scaler mean fitted on train only: {:.2}", mean_train);
        let show = |m: f64| test.iter().map(|v| format!("{:.2}", v - m)).collect::<Vec<_>>().join(", ");
        println!("centred test (leaky): {}", show(mean_all));
        println!("centred test (clean): {}", show(mean_train));
    }
  `, r`
    scaler mean fitted on all data : 18.00  <- uses test information
    scaler mean fitted on train only: 11.50
    centred test (leaky): 12.00, 14.00
    centred test (clean): 18.50, 20.50
  `, "Fitting even a simple scaler on all the data lets test statistics leak into training: the leaky version makes the shifted test data look less unusual than it really is. Fit every preprocessing step on the training split only."),

  regularization: ex(r`
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.1, 3.9, 6.2, 7.8]
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    for lam in [0.0, 1.0, 10.0, 100.0]:
        w = sxy / (sxx + lam)                      # ridge solution for y = w x (no intercept)
        print(f"lambda={lam:6.1f}  w={w:.4f}")
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0_f64];
        let ys = [2.1, 3.9, 6.2, 7.8_f64];
        let sxx: f64 = xs.iter().map(|x| x * x).sum();
        let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        for lam in [0.0, 1.0, 10.0, 100.0] {
            let w = sxy / (sxx + lam); // ridge solution for y = w x (no intercept)
            println!("lambda={:6.1}  w={:.4}", lam, w);
        }
    }
  `, r`
    lambda=   0.0  w=1.9900
    lambda=   1.0  w=1.9258
    lambda=  10.0  w=1.4925
    lambda= 100.0  w=0.4592
  `, "The penalty λw² shrinks the weight towards zero as λ grows: λ trades a slightly worse fit on the training data for a simpler, more stable model."),

  "linear-regression": ex(r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.2, 4.1, 6.3, 7.9, 10.1]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    print(f"y = {a:.3f} + {b:.3f} x")
    print(f"R^2 = {1 - ss_res / ss_tot:.4f}")
    print(f"prediction at x=6: {a + b * 6:.3f}")
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0_f64];
        let ys = [2.2, 4.1, 6.3, 7.9, 10.1_f64];
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let b = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
        let a = my - b * mx;
        let ss_res: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (y - (a + b * x)).powi(2)).sum();
        let ss_tot: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
        println!("y = {:.3} + {:.3} x", a, b);
        println!("R^2 = {:.4}", 1.0 - ss_res / ss_tot);
        println!("prediction at x=6: {:.3}", a + b * 6.0);
    }
  `, r`
    y = 0.240 + 1.960 x
    R^2 = 0.9981
    prediction at x=6: 12.000
  `, "The least-squares slope is cov(x, y) / var(x) and the intercept makes the line pass through the means. R² close to 1 says the line explains almost all the variation."),

  "polynomial-regression": ex(r`
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    ys = [x * x * 0.5 + 2 * x + 1 for x in xs]    # exact quadratic, no noise
    # normal equations for [1, x, x^2]: (X^T X) w = X^T y, solved by Gaussian elimination
    A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
    for c in range(3):
        for r_ in range(c + 1, 3):
            m = A[r_][c] / A[c][c]
            A[r_] = [A[r_][k] - m * A[c][k] for k in range(3)]
            b[r_] -= m * b[c]
    w = [0.0] * 3
    for i in (2, 1, 0):
        w[i] = (b[i] - sum(A[i][k] * w[k] for k in range(i + 1, 3))) / A[i][i]
    print("w0, w1, w2 =", ", ".join(f"{v + 0.0:.4f}" for v in w))
  `, r`
    fn main() {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0_f64];
        let ys: Vec<f64> = xs.iter().map(|x| x * x * 0.5 + 2.0 * x + 1.0).collect(); // exact quadratic, no noise
        // normal equations for [1, x, x^2]: (X^T X) w = X^T y, solved by Gaussian elimination
        let mut a = [[0.0_f64; 3]; 3];
        let mut b = [0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 { a[i][j] = xs.iter().map(|x| x.powi((i + j) as i32)).sum(); }
            b[i] = xs.iter().zip(ys.iter()).map(|(x, y)| y * x.powi(i as i32)).sum();
        }
        for c in 0..3 {
            for r in c + 1..3 {
                let m = a[r][c] / a[c][c];
                for k in 0..3 { a[r][k] -= m * a[c][k]; }
                b[r] -= m * b[c];
            }
        }
        let mut w = [0.0_f64; 3];
        for i in (0..3).rev() {
            let s: f64 = (i + 1..3).map(|k| a[i][k] * w[k]).sum();
            w[i] = (b[i] - s) / a[i][i];
        }
        println!("w0, w1, w2 = {}", w.iter().map(|v| format!("{:.4}", v + 0.0)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    w0, w1, w2 = 1.0000, 2.0000, 0.5000
  `, "Polynomial regression is linear regression on the features 1, x, x². Solving the normal equations recovers the generating curve 1 + 2x + 0.5x² exactly because the data has no noise."),

  "ridge-regression": ex(r`
    # two almost identical features: OLS weights blow up, ridge stays stable
    X = [[1.0, 1.01], [2.0, 1.98], [3.0, 3.02], [4.0, 3.99]]
    y = [2.0, 4.1, 5.9, 8.0]
    def solve2(lam):
        a = sum(r[0] * r[0] for r in X) + lam; b = sum(r[0] * r[1] for r in X)
        d = sum(r[1] * r[1] for r in X) + lam
        e = sum(r[0] * t for r, t in zip(X, y)); f = sum(r[1] * t for r, t in zip(X, y))
        det = a * d - b * b
        return (d * e - b * f) / det, (a * f - b * e) / det
    for lam in [0.0, 0.1, 1.0]:
        w1, w2 = solve2(lam)
        print(f"lambda={lam:.1f}: w=({w1:+.3f}, {w2:+.3f})")
  `, r`
    fn main() {
        // two almost identical features: OLS weights blow up, ridge stays stable
        let x = [[1.0, 1.01], [2.0, 1.98], [3.0, 3.02], [4.0, 3.99_f64]];
        let y = [2.0, 4.1, 5.9, 8.0_f64];
        let solve2 = |lam: f64| {
            let a: f64 = x.iter().map(|r| r[0] * r[0]).sum::<f64>() + lam;
            let b: f64 = x.iter().map(|r| r[0] * r[1]).sum();
            let d: f64 = x.iter().map(|r| r[1] * r[1]).sum::<f64>() + lam;
            let e: f64 = x.iter().zip(y.iter()).map(|(r, t)| r[0] * t).sum();
            let f: f64 = x.iter().zip(y.iter()).map(|(r, t)| r[1] * t).sum();
            let det = a * d - b * b;
            ((d * e - b * f) / det, (a * f - b * e) / det)
        };
        for lam in [0.0, 0.1, 1.0] {
            let (w1, w2) = solve2(lam);
            println!("lambda={:.1}: w=({:+.3}, {:+.3})", lam, w1, w2);
        }
    }
  `, r`
    lambda=0.0: w=(+6.042, -4.047)
    lambda=0.1: w=(+1.022, +0.972)
    lambda=1.0: w=(+0.985, +0.979)
  `, "With two nearly duplicate features, ordinary least squares (λ = 0) produces large opposite-signed weights that depend on tiny noise. Adding λI to XᵀX makes the system well conditioned and splits the weight evenly between the twins."),

  "lasso-regression": ex(r`
    rho, z = 3.0, 1.0                              # rho = x^T r / n, z = x^T x / n for one coordinate
    soft = lambda v, t: (abs(v) - t if abs(v) > t else 0.0) * (1 if v > 0 else -1)
    for lam in [0.0, 1.0, 2.5, 3.0, 4.0]:
        print(f"lambda={lam:.1f}: w={soft(rho, lam) / z + 0.0:.2f}")
  `, r`
    fn main() {
        let (rho, z) = (3.0_f64, 1.0_f64); // rho = x^T r / n, z = x^T x / n for one coordinate
        let soft = |v: f64, t: f64| if v.abs() > t { (v.abs() - t) * v.signum() } else { 0.0 };
        for lam in [0.0, 1.0, 2.5, 3.0, 4.0] {
            println!("lambda={:.1}: w={:.2}", lam, soft(rho, lam) / z + 0.0);
        }
    }
  `, r`
    lambda=0.0: w=3.00
    lambda=1.0: w=2.00
    lambda=2.5: w=0.50
    lambda=3.0: w=0.00
    lambda=4.0: w=0.00
  `, "The Lasso coordinate update is a soft threshold: the weight shrinks linearly with λ and becomes exactly 0 once λ ≥ |ρ| = 3. That exact zero is what makes the Lasso a feature selector."),

  "elastic-net": ex(r`
    rho, z, lam = 3.0, 1.0, 2.0
    soft = lambda v, t: (abs(v) - t if abs(v) > t else 0.0) * (1 if v > 0 else -1)
    for alpha in [1.0, 0.5, 0.0]:                  # 1 = pure lasso, 0 = pure ridge
        w = soft(rho, lam * alpha) / (z + lam * (1 - alpha))
        print(f"alpha={alpha:.1f}: w={w:.3f}")
  `, r`
    fn main() {
        let (rho, z, lam) = (3.0_f64, 1.0_f64, 2.0_f64);
        let soft = |v: f64, t: f64| if v.abs() > t { (v.abs() - t) * v.signum() } else { 0.0 };
        for alpha in [1.0, 0.5, 0.0] {
            // 1 = pure lasso, 0 = pure ridge
            let w = soft(rho, lam * alpha) / (z + lam * (1.0 - alpha));
            println!("alpha={:.1}: w={:.3}", alpha, w);
        }
    }
  `, r`
    alpha=1.0: w=1.000
    alpha=0.5: w=1.000
    alpha=0.0: w=1.000
  `, "Elastic Net mixes both penalties: the L1 part soft-thresholds (subtracting λα) and the L2 part divides by 1 + λ(1 − α). α slides smoothly between lasso-like sparsity and ridge-like shrinkage."),

  "logistic-regression": ex(r`
    import math
    hours = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0]
    passed = [0, 0, 0, 1, 0, 1, 1, 1]
    w, b, lr = 0.0, 0.0, 0.5
    sig = lambda z: 1 / (1 + math.exp(-z))
    for _ in range(2000):
        gw = sum((sig(w * x + b) - y) * x for x, y in zip(hours, passed)) / len(hours)
        gb = sum(sig(w * x + b) - y for x, y in zip(hours, passed)) / len(hours)
        w -= lr * gw; b -= lr * gb
    print(f"w={w:.3f} b={b:.3f}  decision boundary at {-b / w:.2f} hours")
    for x in [1.0, 2.5, 4.5]:
        print(f"P(pass | {x} h) = {sig(w * x + b):.3f}")
  `, r`
    fn main() {
        let hours = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0_f64];
        let passed = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0_f64];
        let (mut w, mut b, lr) = (0.0_f64, 0.0_f64, 0.5);
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let n = hours.len() as f64;
        for _ in 0..2000 {
            let gw: f64 = hours.iter().zip(passed.iter()).map(|(x, y)| (sig(w * x + b) - y) * x).sum::<f64>() / n;
            let gb: f64 = hours.iter().zip(passed.iter()).map(|(x, y)| sig(w * x + b) - y).sum::<f64>() / n;
            w -= lr * gw;
            b -= lr * gb;
        }
        println!("w={:.3} b={:.3}  decision boundary at {:.2} hours", w, b, -b / w);
        for x in [1.0, 2.5, 4.5] {
            println!("P(pass | {:.1} h) = {:.3}", x, sig(w * x + b));
        }
    }
  `, r`
    w=1.619 b=-4.069  decision boundary at 2.51 hours
    P(pass | 1.0 h) = 0.079
    P(pass | 2.5 h) = 0.494
    P(pass | 4.5 h) = 0.961
  `, "Gradient descent on the log-loss fits a sigmoid curve of pass probability against study hours. The decision boundary is where w·x + b = 0, i.e. probability 0.5."),

  "poisson-regression": ex(r`
    import math
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1, 2, 4, 7, 15]                          # counts that grow multiplicatively
    b0, b1 = 0.0, 0.0
    for _ in range(25):                            # Newton-Raphson on the Poisson log-likelihood
        mu = [math.exp(b0 + b1 * x) for x in xs]
        g0 = sum(y - m for y, m in zip(ys, mu)); g1 = sum((y - m) * x for x, y, m in zip(xs, ys, mu))
        h00 = sum(mu); h01 = sum(m * x for x, m in zip(xs, mu)); h11 = sum(m * x * x for x, m in zip(xs, mu))
        det = h00 * h11 - h01 * h01
        b0 += (h11 * g0 - h01 * g1) / det
        b1 += (h00 * g1 - h01 * g0) / det
    print(f"log mu = {b0:.4f} + {b1:.4f} x")
    print(f"rate ratio per unit x = {math.exp(b1):.4f}")
  `, r`
    fn main() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0_f64];
        let ys = [1.0, 2.0, 4.0, 7.0, 15.0_f64]; // counts that grow multiplicatively
        let (mut b0, mut b1) = (0.0_f64, 0.0_f64);
        for _ in 0..25 {
            // Newton-Raphson on the Poisson log-likelihood
            let mu: Vec<f64> = xs.iter().map(|x| (b0 + b1 * x).exp()).collect();
            let g0: f64 = ys.iter().zip(&mu).map(|(y, m)| y - m).sum();
            let g1: f64 = xs.iter().zip(ys.iter()).zip(&mu).map(|((x, y), m)| (y - m) * x).sum();
            let h00: f64 = mu.iter().sum();
            let h01: f64 = xs.iter().zip(&mu).map(|(x, m)| m * x).sum();
            let h11: f64 = xs.iter().zip(&mu).map(|(x, m)| m * x * x).sum();
            let det = h00 * h11 - h01 * h01;
            b0 += (h11 * g0 - h01 * g1) / det;
            b1 += (h00 * g1 - h01 * g0) / det;
        }
        println!("log mu = {:.4} + {:.4} x", b0, b1);
        println!("rate ratio per unit x = {:.4}", b1.exp());
    }
  `, r`
    log mu = -0.0045 + 0.6734 x
    rate ratio per unit x = 1.9609
  `, "Newton's method on the concave Poisson log-likelihood converges in a few steps. Each unit of x multiplies the expected count by e^{b₁} — here roughly doubling it."),

  "quantile-regression": ex(r`
    delivery = [22, 25, 25, 28, 30, 31, 35, 40, 48, 60]   # minutes
    def pinball(c, tau):
        return sum(tau * (y - c) if y >= c else (tau - 1) * (y - c) for y in delivery)
    for tau in [0.5, 0.9]:
        best = min(delivery, key=lambda c: pinball(c, tau))   # the optimum is one of the data points
        print(f"tau={tau}: prediction={best} min, loss={pinball(best, tau):.1f}")
  `, r`
    fn main() {
        let delivery = [22.0, 25.0, 25.0, 28.0, 30.0, 31.0, 35.0, 40.0, 48.0, 60.0_f64]; // minutes
        let pinball = |c: f64, tau: f64| -> f64 {
            delivery.iter().map(|&y| if y >= c { tau * (y - c) } else { (tau - 1.0) * (y - c) }).sum()
        };
        for tau in [0.5, 0.9] {
            // the optimum is one of the data points
            let mut best = delivery[0];
            for &c in delivery.iter() { if pinball(c, tau) < pinball(best, tau) { best = c; } }
            println!("tau={}: prediction={} min, loss={:.1}", tau, best, pinball(best, tau));
        }
    }
  `, r`
    tau=0.5: prediction=30 min, loss=42.0
    tau=0.9: prediction=60 min, loss=25.6
  `, "Minimising the pinball loss with τ = 0.5 gives a median; τ = 0.9 gives the value that 90 % of deliveries beat, which is much later because of the long right tail. With features, the same loss fits a whole quantile line."),

  "robust-regression": ex(r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.1, 2.0, 2.9, 4.2, 5.0, 20.0]           # the last point is an outlier
    def wls(w):
        sw = sum(w); mx = sum(a * x for a, x in zip(w, xs)) / sw; my = sum(a * y for a, y in zip(w, ys)) / sw
        b = sum(a * (x - mx) * (y - my) for a, x, y in zip(w, xs, ys)) / sum(a * (x - mx) ** 2 for a, x in zip(w, xs))
        return my - b * mx, b
    a, b = wls([1.0] * 6)
    print(f"OLS   slope={b:.3f}")
    delta = 1.0
    for _ in range(50):                            # Huber via iteratively reweighted least squares
        res = [abs(y - a - b * x) for x, y in zip(xs, ys)]
        a, b = wls([1.0 if r_ <= delta else delta / r_ for r_ in res])
    print(f"Huber slope={b:.3f}")
  `, r`
    fn wls(w: &[f64], xs: &[f64], ys: &[f64]) -> (f64, f64) {
        let sw: f64 = w.iter().sum();
        let mx = w.iter().zip(xs).map(|(a, x)| a * x).sum::<f64>() / sw;
        let my = w.iter().zip(ys).map(|(a, y)| a * y).sum::<f64>() / sw;
        let num: f64 = (0..xs.len()).map(|i| w[i] * (xs[i] - mx) * (ys[i] - my)).sum();
        let den: f64 = (0..xs.len()).map(|i| w[i] * (xs[i] - mx).powi(2)).sum();
        let b = num / den;
        (my - b * mx, b)
    }

    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
        let ys = [1.1, 2.0, 2.9, 4.2, 5.0, 20.0_f64]; // the last point is an outlier
        let (mut a, mut b) = wls(&[1.0; 6], &xs, &ys);
        println!("OLS   slope={:.3}", b);
        let delta = 1.0;
        for _ in 0..50 {
            // Huber via iteratively reweighted least squares
            let w: Vec<f64> = xs.iter().zip(ys.iter()).map(|(x, y)| { let r = (y - a - b * x).abs(); if r <= delta { 1.0 } else { delta / r } }).collect();
            let fit = wls(&w, &xs, &ys);
            a = fit.0;
            b = fit.1;
        }
        println!("Huber slope={:.3}", b);
    }
  `, r`
    OLS   slope=2.994
    Huber slope=1.300
  `, "The single outlier triples the OLS slope. Huber regression down-weights points with large residuals (weight δ/|r|), so its slope stays much closer to the underlying trend of about 1."),

  "k-nearest-neighbors": ex(r`
    import math
    pts = [((1.0, 1.0), "A"), ((1.5, 2.0), "A"), ((3.0, 4.0), "B"), ((5.0, 7.0), "B"), ((3.5, 5.0), "B"), ((2.0, 1.5), "A")]
    q = (2.5, 3.0)
    ranked = sorted(pts, key=lambda p: math.dist(p[0], q))
    for (xy, lab) in ranked[:3]:
        print(f"neighbour {xy} label={lab} dist={math.dist(xy, q):.3f}")
    votes = [lab for _, lab in ranked[:3]]
    print("prediction (k=3):", max(set(votes), key=votes.count))
  `, r`
    fn main() {
        let pts = [((1.0, 1.0), "A"), ((1.5, 2.0), "A"), ((3.0, 4.0), "B"), ((5.0, 7.0), "B"), ((3.5, 5.0), "B"), ((2.0, 1.5), "A")];
        let q = (2.5_f64, 3.0_f64);
        let dist = |p: (f64, f64)| (p.0 - q.0).hypot(p.1 - q.1);
        let mut ranked = pts.to_vec();
        ranked.sort_by(|a, b| dist(a.0).partial_cmp(&dist(b.0)).unwrap());
        for (xy, lab) in ranked.iter().take(3) {
            println!("neighbour ({:.1}, {:.1}) label={} dist={:.3}", xy.0, xy.1, lab, dist(*xy));
        }
        let a = ranked.iter().take(3).filter(|p| p.1 == "A").count();
        println!("prediction (k=3): {}", if a >= 2 { "A" } else { "B" });
    }
  `, r`
    neighbour (3.0, 4.0) label=B dist=1.118
    neighbour (1.5, 2.0) label=A dist=1.414
    neighbour (2.0, 1.5) label=A dist=1.581
    prediction (k=3): A
  `, "k-NN stores the training points and, for a query, takes a majority vote among the k closest. The single closest point is class B, but two of the three nearest are class A, so k = 3 predicts A — the choice of k matters."),

  "naive-bayes": ex(r`
    import math
    spam = [8.0, 9.5, 7.0, 10.0]                     # number of links in spam e-mails
    ham = [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]             # ... and in normal e-mails
    def fit(v):
        m = sum(v) / len(v)
        return m, sum((x - m) ** 2 for x in v) / len(v)
    def gauss(x, m, var):
        return math.exp(-(x - m) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
    (ms, vs), (mh, vh) = fit(spam), fit(ham)
    ps, ph = len(spam) / 10, len(ham) / 10
    for x in [2.0, 5.0, 6.0]:
        a, b = ps * gauss(x, ms, vs), ph * gauss(x, mh, vh)
        print(f"links={x}: P(spam | x) = {a / (a + b):.4f}")
  `, r`
    fn fit(v: &[f64]) -> (f64, f64) {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64)
    }

    fn gauss(x: f64, m: f64, var: f64) -> f64 {
        (-(x - m).powi(2) / (2.0 * var)).exp() / (2.0 * std::f64::consts::PI * var).sqrt()
    }

    fn main() {
        let spam = [8.0, 9.5, 7.0, 10.0]; // number of links in spam e-mails
        let ham = [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]; // ... and in normal e-mails
        let ((ms, vs), (mh, vh)) = (fit(&spam), fit(&ham));
        let (ps, ph) = (spam.len() as f64 / 10.0, ham.len() as f64 / 10.0);
        for x in [2.0, 5.0, 6.0] {
            let (a, b) = (ps * gauss(x, ms, vs), ph * gauss(x, mh, vh));
            println!("links={:.1}: P(spam | x) = {:.4}", x, a / (a + b));
        }
    }
  `, r`
    links=2.0: P(spam | x) = 0.0000
    links=5.0: P(spam | x) = 0.8679
    links=6.0: P(spam | x) = 0.9999
  `, "Gaussian Naive Bayes fits one normal distribution per class and combines it with the class prior through Bayes' rule. With several features it would simply multiply one such likelihood per feature (the 'naive' independence assumption)."),

  "decision-trees": ex(r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ys = [0, 0, 0, 1, 0, 1, 1, 1]
    def gini(labels):
        if not labels: return 0.0
        p = sum(labels) / len(labels)
        return 2 * p * (1 - p)
    best = None
    for t in [(a + b) / 2 for a, b in zip(xs, xs[1:])]:
        left = [y for x, y in zip(xs, ys) if x <= t]; right = [y for x, y in zip(xs, ys) if x > t]
        g = (len(left) * gini(left) + len(right) * gini(right)) / len(ys)
        print(f"split x <= {t}: weighted Gini = {g:.4f}")
        if best is None or g < best[1]: best = (t, g)
    print(f"best split: x <= {best[0]} (Gini {best[1]:.4f}, parent {gini(ys):.4f})")
  `, r`
    fn gini(labels: &[f64]) -> f64 {
        if labels.is_empty() { return 0.0; }
        let p = labels.iter().sum::<f64>() / labels.len() as f64;
        2.0 * p * (1.0 - p)
    }

    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64];
        let ys = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0_f64];
        let mut best = (0.0, f64::MAX);
        for k in 0..xs.len() - 1 {
            let t = (xs[k] + xs[k + 1]) / 2.0;
            let left: Vec<f64> = xs.iter().zip(ys.iter()).filter(|(x, _)| **x <= t).map(|(_, y)| *y).collect();
            let right: Vec<f64> = xs.iter().zip(ys.iter()).filter(|(x, _)| **x > t).map(|(_, y)| *y).collect();
            let g = (left.len() as f64 * gini(&left) + right.len() as f64 * gini(&right)) / ys.len() as f64;
            println!("split x <= {}: weighted Gini = {:.4}", t, g);
            if g < best.1 { best = (t, g); }
        }
        println!("best split: x <= {} (Gini {:.4}, parent {:.4})", best.0, best.1, gini(&ys));
    }
  `, r`
    split x <= 1.5: weighted Gini = 0.4286
    split x <= 2.5: weighted Gini = 0.3333
    split x <= 3.5: weighted Gini = 0.2000
    split x <= 4.5: weighted Gini = 0.3750
    split x <= 5.5: weighted Gini = 0.2000
    split x <= 6.5: weighted Gini = 0.3333
    split x <= 7.5: weighted Gini = 0.4286
    best split: x <= 3.5 (Gini 0.2000, parent 0.5000)
  `, "A tree chooses the threshold whose children are purest, measured by the size-weighted Gini impurity. Splitting at x ≤ 3.5 leaves one pure child and one mostly-1 child; the tree then recurses into each side."),

  "random-forests": ex(join(PY_LCG, r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ys = [0, 0, 1, 0, 1, 1, 0, 1]                  # noisy labels
    def stump(sx, sy):                             # best single threshold by misclassification
        best = (0, 1.0, 0, 1)
        for t in [x + 0.5 for x in range(0, 9)]:
            for lo, hi in ((0, 1), (1, 0)):
                err = sum((lo if x <= t else hi) != y for x, y in zip(sx, sy)) / len(sx)
                if err < best[1]: best = (t, err, lo, hi)
        return best
    trees = []
    for _ in range(25):                            # bootstrap sample for every tree
        idx = [int(rand() * len(xs)) for _ in xs]
        trees.append(stump([xs[i] for i in idx], [ys[i] for i in idx]))
    for q in [2.0, 4.5, 7.5]:
        votes = sum((t[2] if q <= t[0] else t[3]) for t in trees)
        print(f"x={q}: {votes}/25 trees vote 1 -> class {int(votes > 12)}")
  `), join(RS_LCG, r`
    fn stump(sx: &[f64], sy: &[i32]) -> (f64, f64, i32, i32) {
        // best single threshold by misclassification
        let mut best = (0.0, 1.0, 0, 1);
        for k in 0..9 {
            let t = k as f64 + 0.5;
            for (lo, hi) in [(0, 1), (1, 0)] {
                let err = sx.iter().zip(sy).filter(|(x, y)| (if **x <= t { lo } else { hi }) != **y).count() as f64 / sx.len() as f64;
                if err < best.1 { best = (t, err, lo, hi); }
            }
        }
        best
    }

    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64];
        let ys = [0, 0, 1, 0, 1, 1, 0, 1]; // noisy labels
        let mut trees = Vec::new();
        for _ in 0..25 {
            // bootstrap sample for every tree
            let idx: Vec<usize> = xs.iter().map(|_| (rand() * xs.len() as f64) as usize).collect();
            let sx: Vec<f64> = idx.iter().map(|&i| xs[i]).collect();
            let sy: Vec<i32> = idx.iter().map(|&i| ys[i]).collect();
            trees.push(stump(&sx, &sy));
        }
        for q in [2.0, 4.5, 7.5] {
            let votes: i32 = trees.iter().map(|t| if q <= t.0 { t.2 } else { t.3 }).sum();
            println!("x={:.1}: {}/25 trees vote 1 -> class {}", q, votes, if votes > 12 { 1 } else { 0 });
        }
    }
  `), r`
    x=2.0: 6/25 trees vote 1 -> class 0
    x=4.5: 18/25 trees vote 1 -> class 1
    x=7.5: 21/25 trees vote 1 -> class 1
  `, "Each tree is trained on a different bootstrap sample, so the trees disagree; the forest takes a majority vote. Averaging many decorrelated trees reduces variance compared with any single tree."),

  "support-vector-machines": ex(r`
    X = [(2.0, 3.0), (3.0, 3.5), (1.0, 1.0), (2.0, 0.5), (4.0, 4.5), (0.5, 2.0)]
    y = [1, 1, -1, -1, 1, -1]
    w, b, lam = [0.0, 0.0], 0.0, 0.01
    for t in range(1, 5001):                       # Pegasos: stochastic sub-gradient on the hinge loss
        i = t % len(X)
        eta = 1 / (lam * t)
        margin = y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b)
        w = [(1 - eta * lam) * wj for wj in w]
        if margin < 1:
            w = [w[0] + eta * y[i] * X[i][0], w[1] + eta * y[i] * X[i][1]]
            b += eta * y[i] * 0.01
    print(f"w=({w[0]:.3f}, {w[1]:.3f}) b={b:.3f}")
    print("margins y*f(x):", ", ".join(f"{yi * (w[0] * x0 + w[1] * x1 + b):.2f}" for (x0, x1), yi in zip(X, y)))
  `, r`
    fn main() {
        let x = [(2.0, 3.0), (3.0, 3.5), (1.0, 1.0), (2.0, 0.5), (4.0, 4.5), (0.5, 2.0_f64)];
        let y = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0_f64];
        let (mut w, mut b, lam) = ([0.0_f64, 0.0], 0.0_f64, 0.01);
        for t in 1..=5000 {
            // Pegasos: stochastic sub-gradient on the hinge loss
            let i = t % x.len();
            let eta = 1.0 / (lam * t as f64);
            let margin = y[i] * (w[0] * x[i].0 + w[1] * x[i].1 + b);
            w = [(1.0 - eta * lam) * w[0], (1.0 - eta * lam) * w[1]];
            if margin < 1.0 {
                w = [w[0] + eta * y[i] * x[i].0, w[1] + eta * y[i] * x[i].1];
                b += eta * y[i] * 0.01;
            }
        }
        println!("w=({:.3}, {:.3}) b={:.3}", w[0], w[1], b);
        let m: Vec<String> = x.iter().zip(y.iter()).map(|(p, yi)| format!("{:.2}", yi * (w[0] * p.0 + w[1] * p.1 + b))).collect();
        println!("margins y*f(x): {}", m.join(", "));
    }
  `, r`
    w=(0.090, 0.620) b=-1.531
    margins y*f(x): 0.51, 0.91, 0.82, 1.04, 1.62, 0.25
  `, "The linear SVM minimises the regularised hinge loss. All margins y·f(x) end up positive, so every point is classified correctly; points with margin near 1 are the support vectors that hold the boundary in place."),

  "multiclass-strategies": ex(r`
    # three trained one-vs-rest scorers s_k(x) = w_k . x + b_k
    W = {"cat": (1.0, -0.5, 0.2), "dog": (-0.3, 1.2, -0.1), "fox": (0.4, 0.4, -0.6)}
    for x in [(2.0, 0.5), (0.2, 1.5), (1.2, 1.1)]:
        scores = {k: w[0] * x[0] + w[1] * x[1] + w[2] for k, w in W.items()}
        best = max(scores, key=scores.get)
        print(x, " ".join(f"{k}={v:+.2f}" for k, v in scores.items()), "->", best)
  `, r`
    fn main() {
        // three trained one-vs-rest scorers s_k(x) = w_k . x + b_k
        let w = [("cat", (1.0, -0.5, 0.2)), ("dog", (-0.3, 1.2, -0.1)), ("fox", (0.4, 0.4, -0.6_f64))];
        for x in [(2.0, 0.5), (0.2, 1.5), (1.2, 1.1_f64)] {
            let scores: Vec<(&str, f64)> = w.iter().map(|(k, c)| (*k, c.0 * x.0 + c.1 * x.1 + c.2)).collect();
            let best = scores.iter().cloned().fold(("", f64::MIN), |a, s| if s.1 > a.1 { s } else { a });
            let shown: Vec<String> = scores.iter().map(|(k, v)| format!("{}={:+.2}", k, v)).collect();
            println!("({:.1}, {:.1}) {} -> {}", x.0, x.1, shown.join(" "), best.0);
        }
    }
  `, r`
    (2.0, 0.5) cat=+1.95 dog=-0.10 fox=+0.40 -> cat
    (0.2, 1.5) cat=-0.35 dog=+1.64 fox=+0.08 -> dog
    (1.2, 1.1) cat=+0.85 dog=+0.86 fox=+0.32 -> dog
  `, "One-vs-rest keeps one binary scorer per class and predicts the class with the highest score. One-vs-one would instead train a model for every pair of classes and count votes."),

  "gradient-boosting": ex(r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.0, 1.2, 1.9, 3.8, 4.1, 4.2]
    pred = [sum(ys) / len(ys)] * len(xs)           # start from the mean
    lr = 0.5
    def best_stump(res):
        best = None
        for t in [x + 0.5 for x in xs[:-1]]:
            l = [r_ for x, r_ in zip(xs, res) if x <= t]; r2 = [r_ for x, r_ in zip(xs, res) if x > t]
            lv, rv = sum(l) / len(l), sum(r2) / len(r2)
            sse = sum((r_ - (lv if x <= t else rv)) ** 2 for x, r_ in zip(xs, res))
            if best is None or sse < best[0]: best = (sse, t, lv, rv)
        return best
    for m in range(1, 5):
        res = [y - p for y, p in zip(ys, pred)]   # fit a stump to the residuals (negative gradient of MSE)
        _, t, lv, rv = best_stump(res)
        pred = [p + lr * (lv if x <= t else rv) for x, p in zip(xs, pred)]
        mse = sum((y - p) ** 2 for y, p in zip(ys, pred)) / len(ys)
        print(f"round {m}: split at {t}, train MSE = {mse:.4f}")
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
        let ys = [1.0, 1.2, 1.9, 3.8, 4.1, 4.2_f64];
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let mut pred = vec![mean; xs.len()]; // start from the mean
        let lr = 0.5;
        for m in 1..=4 {
            // fit a stump to the residuals (negative gradient of MSE)
            let res: Vec<f64> = ys.iter().zip(&pred).map(|(y, p)| y - p).collect();
            let mut best = (f64::MAX, 0.0, 0.0, 0.0);
            for k in 0..xs.len() - 1 {
                let t = xs[k] + 0.5;
                let l: Vec<f64> = (0..xs.len()).filter(|&i| xs[i] <= t).map(|i| res[i]).collect();
                let r: Vec<f64> = (0..xs.len()).filter(|&i| xs[i] > t).map(|i| res[i]).collect();
                let (lv, rv) = (l.iter().sum::<f64>() / l.len() as f64, r.iter().sum::<f64>() / r.len() as f64);
                let sse: f64 = (0..xs.len()).map(|i| (res[i] - if xs[i] <= t { lv } else { rv }).powi(2)).sum();
                if sse < best.0 { best = (sse, t, lv, rv); }
            }
            let (_, t, lv, rv) = best;
            for i in 0..xs.len() { pred[i] += lr * if xs[i] <= t { lv } else { rv }; }
            let mse = ys.iter().zip(&pred).map(|(y, p)| (y - p).powi(2)).sum::<f64>() / ys.len() as f64;
            println!("round {}: split at {}, train MSE = {:.4}", m, t, mse);
        }
    }
  `, r`
    round 1: split at 3.5, train MSE = 0.5333
    round 2: split at 3.5, train MSE = 0.2000
    round 3: split at 2.5, train MSE = 0.0650
    round 4: split at 4.5, train MSE = 0.0312
  `, "Each round fits a small tree (a stump) to the current residuals and adds a shrunken copy of it. Training error falls round after round; the learning rate controls how big each correction is."),

  bagging: ex(join(PY_LCG, r`
    import math
    sample = [2.0, 4.0, 4.0, 5.0, 7.0, 9.0, 10.0, 15.0]
    boot_means = []
    for _ in range(1000):                          # resample with replacement and recompute
        s = [sample[int(rand() * len(sample))] for _ in sample]
        boot_means.append(sum(s) / len(s))
    m = sum(boot_means) / len(boot_means)
    sd = math.sqrt(sum((v - m) ** 2 for v in boot_means) / len(boot_means))
    mean = sum(sample) / len(sample)
    se = math.sqrt(sum((v - mean) ** 2 for v in sample) / len(sample)) / math.sqrt(len(sample))
    print(f"sample mean = {mean:.3f}, bagged mean = {m:.3f}")
    print(f"bootstrap std of the mean = {sd:.3f} (formula: {se:.3f})")
  `), join(RS_LCG, r`
    fn main() {
        let sample = [2.0, 4.0, 4.0, 5.0, 7.0, 9.0, 10.0, 15.0_f64];
        let n = sample.len();
        let mut boot = Vec::new();
        for _ in 0..1000 {
            // resample with replacement and recompute
            let s: f64 = (0..n).map(|_| sample[(rand() * n as f64) as usize]).sum();
            boot.push(s / n as f64);
        }
        let m = boot.iter().sum::<f64>() / boot.len() as f64;
        let sd = (boot.iter().map(|v| (v - m).powi(2)).sum::<f64>() / boot.len() as f64).sqrt();
        let mean = sample.iter().sum::<f64>() / n as f64;
        let se = (sample.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt() / (n as f64).sqrt();
        println!("sample mean = {:.3}, bagged mean = {:.3}", mean, m);
        println!("bootstrap std of the mean = {:.3} (formula: {:.3})", sd, se);
    }
  `), r`
    sample mean = 7.000, bagged mean = 7.020
    bootstrap std of the mean = 1.418 (formula: 1.392)
  `, "Bagging = bootstrap aggregating: train (here: compute) the same estimator on many resamples and average. The spread across resamples matches the theoretical standard error, showing how much a single model would vary."),

  adaboost: ex(r`
    import math
    xs = [1, 2, 3, 4, 5, 6]
    ys = [1, 1, -1, -1, 1, 1]
    w = [1 / 6] * 6
    F = [0.0] * 6
    for rnd in range(1, 4):
        best = None                                # weak learner: the stump with the lowest weighted error
        for k in range(7):
            t = k + 0.5
            for s in (1, -1):                      # predict s if x <= t else -s
                h = [s if x <= t else -s for x in xs]
                err = sum(wi for wi, hi, yi in zip(w, h, ys) if hi != yi)
                if best is None or err < best[0] - 1e-12: best = (err, t, s, h)
        err, t, s, h = best
        alpha = 0.5 * math.log((1 - err) / err)
        w = [wi * math.exp(-alpha * yi * hi) for wi, hi, yi in zip(w, h, ys)]
        z = sum(w); w = [wi / z for wi in w]       # renormalise: misclassified points gain weight
        F = [f + alpha * hi for f, hi in zip(F, h)]
        print(f"round {rnd}: stump x<={t} sign={s:+d} error={err:.3f} alpha={alpha:.3f}")
    print("ensemble signs:", [1 if f > 0 else -1 for f in F], "truth:", ys)
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
        let ys = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0_f64];
        let mut w = vec![1.0 / 6.0; 6];
        let mut f = vec![0.0; 6];
        for rnd in 1..=3 {
            // weak learner: the stump with the lowest weighted error
            let mut best: (f64, f64, f64, Vec<f64>) = (f64::MAX, 0.0, 0.0, vec![]);
            for k in 0..7 {
                let t = k as f64 + 0.5;
                for s in [1.0, -1.0] {
                    // predict s if x <= t else -s
                    let h: Vec<f64> = xs.iter().map(|&x| if x <= t { s } else { -s }).collect();
                    let err: f64 = (0..6).filter(|&i| h[i] != ys[i]).map(|i| w[i]).sum();
                    if err < best.0 - 1e-12 { best = (err, t, s, h); }
                }
            }
            let (err, t, s, h) = best;
            let alpha = 0.5 * ((1.0 - err) / err).ln();
            for i in 0..6 { w[i] *= (-alpha * ys[i] * h[i]).exp(); }
            let z: f64 = w.iter().sum();
            for wi in w.iter_mut() { *wi /= z; } // renormalise: misclassified points gain weight
            for i in 0..6 { f[i] += alpha * h[i]; }
            println!("round {}: stump x<={} sign={:+} error={:.3} alpha={:.3}", rnd, t, s as i32, err, alpha);
        }
        let signs: Vec<i32> = f.iter().map(|&v| if v > 0.0 { 1 } else { -1 }).collect();
        println!("ensemble signs: {:?} truth: {:?}", signs, ys.iter().map(|&v| v as i32).collect::<Vec<_>>());
    }
  `, r`
    round 1: stump x<=0.5 sign=-1 error=0.333 alpha=0.347
    round 2: stump x<=2.5 sign=+1 error=0.250 alpha=0.549
    round 3: stump x<=4.5 sign=-1 error=0.167 alpha=0.805
    ensemble signs: [1, 1, -1, -1, 1, 1] truth: [1, 1, -1, -1, 1, 1]
  `, "Each round picks the stump with the lowest weighted error and gives it a vote α = ½ ln((1 − ε)/ε); misclassified points are up-weighted so the next stump focuses on them. No single threshold can label this pattern (+ + − − + +), but three weighted stumps get every point right."),

  stacking: ex(r`
    y = [3.0, 5.0, 7.0, 9.0]                     # validation targets
    p1 = [2.0, 5.5, 6.0, 10.0]                   # out-of-fold predictions of base model 1
    p2 = [3.5, 4.0, 7.5, 8.0]                    # ... and of base model 2
    # meta-model y ~ a*p1 + b*p2 by least squares (2x2 normal equations)
    s11 = sum(u * u for u in p1); s12 = sum(u * v for u, v in zip(p1, p2)); s22 = sum(v * v for v in p2)
    t1 = sum(u * t for u, t in zip(p1, y)); t2 = sum(v * t for v, t in zip(p2, y))
    det = s11 * s22 - s12 * s12
    a, b = (s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det
    mse = lambda p: sum((t - q) ** 2 for t, q in zip(y, p)) / len(y)
    print(f"meta weights a={a:.3f} b={b:.3f}")
    print(f"MSE model1={mse(p1):.3f} model2={mse(p2):.3f} stacked={mse([a * u + b * v for u, v in zip(p1, p2)]):.3f}")
  `, r`
    fn main() {
        let y = [3.0, 5.0, 7.0, 9.0_f64]; // validation targets
        let p1 = [2.0, 5.5, 6.0, 10.0_f64]; // out-of-fold predictions of base model 1
        let p2 = [3.5, 4.0, 7.5, 8.0_f64]; // ... and of base model 2
        // meta-model y ~ a*p1 + b*p2 by least squares (2x2 normal equations)
        let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
        let (s11, s12, s22) = (dot(&p1, &p1), dot(&p1, &p2), dot(&p2, &p2));
        let (t1, t2) = (dot(&p1, &y), dot(&p2, &y));
        let det = s11 * s22 - s12 * s12;
        let (a, b) = ((s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det);
        let mse = |p: &[f64]| y.iter().zip(p).map(|(t, q)| (t - q).powi(2)).sum::<f64>() / y.len() as f64;
        let st: Vec<f64> = p1.iter().zip(p2.iter()).map(|(u, v)| a * u + b * v).collect();
        println!("meta weights a={:.3} b={:.3}", a, b);
        println!("MSE model1={:.3} model2={:.3} stacked={:.3}", mse(&p1), mse(&p2), mse(&st));
    }
  `, r`
    meta weights a=0.451 b=0.576
    MSE model1=0.812 model2=0.625 stacked=0.017
  `, "The meta-model learns how much to trust each base model from their out-of-fold predictions. Because the two models err in different directions, the learned blend beats both."),

  "voting-classifiers": ex(r`
    probs = {"logreg": [0.62, 0.45, 0.80], "tree": [0.40, 0.95, 0.90], "knn": [0.55, 0.40, 0.65]}   # P(class 1) per model
    for i in range(3):
        ps = [p[i] for p in probs.values()]
        hard = int(sum(p >= 0.5 for p in ps) >= 2)
        soft_p = sum(ps) / len(ps)
        print(f"sample {i}: hard vote={hard}  soft P={soft_p:.3f} -> {int(soft_p >= 0.5)}")
  `, r`
    fn main() {
        // P(class 1) per model: logreg, tree, knn
        let probs = [[0.62, 0.45, 0.80], [0.40, 0.95, 0.90], [0.55, 0.40, 0.65_f64]];
        for i in 0..3 {
            let ps: Vec<f64> = probs.iter().map(|p| p[i]).collect();
            let hard = if ps.iter().filter(|&&p| p >= 0.5).count() >= 2 { 1 } else { 0 };
            let soft_p = ps.iter().sum::<f64>() / ps.len() as f64;
            println!("sample {}: hard vote={}  soft P={:.3} -> {}", i, hard, soft_p, if soft_p >= 0.5 { 1 } else { 0 });
        }
    }
  `, r`
    sample 0: hard vote=1  soft P=0.523 -> 1
    sample 1: hard vote=0  soft P=0.600 -> 1
    sample 2: hard vote=1  soft P=0.783 -> 1
  `, "Hard voting counts class votes; soft voting averages probabilities and therefore uses confidence. They can disagree: for sample 1 only one model votes 1, but it is very confident (0.95), so the averaged probability (0.600) flips the soft vote to 1."),

  "k-means": ex(r`
    pts = [(1.0, 1.0), (1.5, 2.0), (3.0, 4.0), (5.0, 7.0), (3.5, 5.0), (4.5, 5.0), (3.5, 4.5)]
    C = [pts[0], pts[3]]                           # initial centroids
    for it in range(1, 6):
        assign = [min(range(2), key=lambda k: (p[0] - C[k][0]) ** 2 + (p[1] - C[k][1]) ** 2) for p in pts]
        C = [(sum(p[0] for p, a in zip(pts, assign) if a == k) / assign.count(k),
              sum(p[1] for p, a in zip(pts, assign) if a == k) / assign.count(k)) for k in range(2)]
        inertia = sum((p[0] - C[a][0]) ** 2 + (p[1] - C[a][1]) ** 2 for p, a in zip(pts, assign))
        print(f"iter {it}: centroids=({C[0][0]:.2f}, {C[0][1]:.2f}) ({C[1][0]:.2f}, {C[1][1]:.2f}) inertia={inertia:.3f}")
  `, r`
    fn main() {
        let pts = [(1.0, 1.0), (1.5, 2.0), (3.0, 4.0), (5.0, 7.0), (3.5, 5.0), (4.5, 5.0), (3.5, 4.5_f64)];
        let mut c = [pts[0], pts[3]]; // initial centroids
        let d2 = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2);
        for it in 1..=5 {
            let assign: Vec<usize> = pts.iter().map(|&p| if d2(p, c[0]) <= d2(p, c[1]) { 0 } else { 1 }).collect();
            for k in 0..2 {
                let members: Vec<(f64, f64)> = pts.iter().zip(&assign).filter(|(_, &a)| a == k).map(|(p, _)| *p).collect();
                let n = members.len() as f64;
                c[k] = (members.iter().map(|p| p.0).sum::<f64>() / n, members.iter().map(|p| p.1).sum::<f64>() / n);
            }
            let inertia: f64 = pts.iter().zip(&assign).map(|(&p, &a)| d2(p, c[a])).sum();
            println!("iter {}: centroids=({:.2}, {:.2}) ({:.2}, {:.2}) inertia={:.3}", it, c[0].0, c[0].1, c[1].0, c[1].1, inertia);
        }
    }
  `, r`
    iter 1: centroids=(1.83, 2.33) (4.12, 5.38) inertia=12.208
    iter 2: centroids=(1.25, 1.50) (3.90, 5.10) inertia=8.525
    iter 3: centroids=(1.25, 1.50) (3.90, 5.10) inertia=8.525
    iter 4: centroids=(1.25, 1.50) (3.90, 5.10) inertia=8.525
    iter 5: centroids=(1.25, 1.50) (3.90, 5.10) inertia=8.525
  `, "k-means alternates two steps — assign each point to its nearest centroid, then move each centroid to the mean of its points — and the inertia (within-cluster squared distance) never increases. It converges after two iterations here."),

  dbscan: ex(r`
    xs = [1.0, 1.5, 2.0, 2.2, 8.0, 8.4, 8.9, 15.0]
    eps, min_pts = 1.0, 2
    nb = [[j for j in range(len(xs)) if abs(xs[i] - xs[j]) <= eps] for i in range(len(xs))]
    label = [None] * len(xs)
    c = -1
    for i in range(len(xs)):
        if label[i] is not None or len(nb[i]) < min_pts: continue
        c += 1
        queue = [i]
        while queue:                               # expand the cluster from core points
            p = queue.pop()
            if label[p] is not None: continue
            label[p] = c
            if len(nb[p]) >= min_pts: queue.extend(nb[p])
    print("labels:", [l if l is not None else -1 for l in label], "(-1 = noise)")
  `, r`
    fn main() {
        let xs = [1.0, 1.5, 2.0, 2.2, 8.0, 8.4, 8.9, 15.0_f64];
        let (eps, min_pts) = (1.0, 2);
        let nb: Vec<Vec<usize>> = (0..xs.len()).map(|i| (0..xs.len()).filter(|&j| (xs[i] - xs[j]).abs() <= eps).collect()).collect();
        let mut label: Vec<i32> = vec![-2; xs.len()]; // -2 = unvisited
        let mut c = -1;
        for i in 0..xs.len() {
            if label[i] != -2 || nb[i].len() < min_pts { continue; }
            c += 1;
            let mut queue = vec![i];
            while let Some(p) = queue.pop() {
                // expand the cluster from core points
                if label[p] != -2 { continue; }
                label[p] = c;
                if nb[p].len() >= min_pts { queue.extend(nb[p].iter().cloned()); }
            }
        }
        let shown: Vec<i32> = label.iter().map(|&l| if l == -2 { -1 } else { l }).collect();
        println!("labels: {:?} (-1 = noise)", shown);
    }
  `, r`
    labels: [0, 0, 0, 0, 1, 1, 1, -1] (-1 = noise)
  `, "DBSCAN grows clusters from core points (at least minPts neighbours within ε). It finds two clusters without being told how many, and the isolated value 15 is labelled noise."),

  gmm: ex(r`
    import math
    xs = [1.0, 1.4, 1.9, 2.1, 2.4, 6.8, 7.1, 7.5, 8.0, 8.3]
    pi, mu, var = [0.5, 0.5], [0.0, 5.0], [1.0, 1.0]
    N = lambda x, m, v: math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
    for _ in range(30):
        R = [[pi[k] * N(x, mu[k], var[k]) for k in range(2)] for x in xs]          # E-step
        R = [[r_ / sum(row) for r_ in row] for row in R]
        for k in range(2):                                                           # M-step
            nk = sum(row[k] for row in R)
            pi[k] = nk / len(xs)
            mu[k] = sum(row[k] * x for row, x in zip(R, xs)) / nk
            var[k] = sum(row[k] * (x - mu[k]) ** 2 for row, x in zip(R, xs)) / nk
    for k in range(2):
        print(f"component {k}: weight={pi[k]:.3f} mean={mu[k]:.3f} std={math.sqrt(var[k]):.3f}")
  `, r`
    fn main() {
        let xs = [1.0, 1.4, 1.9, 2.1, 2.4, 6.8, 7.1, 7.5, 8.0, 8.3_f64];
        let (mut pi, mut mu, mut var) = ([0.5, 0.5], [0.0, 5.0], [1.0, 1.0_f64]);
        let nrm = |x: f64, m: f64, v: f64| (-(x - m).powi(2) / (2.0 * v)).exp() / (2.0 * std::f64::consts::PI * v).sqrt();
        for _ in 0..30 {
            // E-step: responsibilities
            let r: Vec<[f64; 2]> = xs.iter().map(|&x| {
                let a = [pi[0] * nrm(x, mu[0], var[0]), pi[1] * nrm(x, mu[1], var[1])];
                let s = a[0] + a[1];
                [a[0] / s, a[1] / s]
            }).collect();
            for k in 0..2 {
                // M-step
                let nk: f64 = r.iter().map(|row| row[k]).sum();
                pi[k] = nk / xs.len() as f64;
                mu[k] = r.iter().zip(xs.iter()).map(|(row, x)| row[k] * x).sum::<f64>() / nk;
                var[k] = r.iter().zip(xs.iter()).map(|(row, x)| row[k] * (x - mu[k]).powi(2)).sum::<f64>() / nk;
            }
        }
        for k in 0..2 {
            println!("component {}: weight={:.3} mean={:.3} std={:.3}", k, pi[k], mu[k], var[k].sqrt());
        }
    }
  `, r`
    component 0: weight=0.500 mean=1.760 std=0.500
    component 1: weight=0.500 mean=7.540 std=0.554
  `, "EM alternates soft assignment (E-step: each point's responsibility for each Gaussian) and re-estimation (M-step: weighted weights, means and variances). It recovers the two groups around 1.8 and 7.5."),

  "hierarchical-clustering": ex(r`
    xs = [1.0, 2.0, 6.0, 7.0, 7.5, 15.0]
    clusters = [[x] for x in xs]
    while len(clusters) > 1:                       # single linkage: distance of the closest pair of members
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = min(abs(a - b) for a in clusters[i] for b in clusters[j])
                if best is None or d < best[0]: best = (d, i, j)
        d, i, j = best
        print(f"merge {clusters[i]} + {clusters[j]} at distance {d:.1f}")
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 6.0, 7.0, 7.5, 15.0_f64];
        let mut clusters: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        let fmt = |c: &Vec<f64>| format!("[{}]", c.iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>().join(", "));
        while clusters.len() > 1 {
            // single linkage: distance of the closest pair of members
            let mut best = (f64::MAX, 0, 0);
            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let d = clusters[i].iter().flat_map(|a| clusters[j].iter().map(move |b| (a - b).abs())).fold(f64::MAX, f64::min);
                    if d < best.0 { best = (d, i, j); }
                }
            }
            let (d, i, j) = best;
            println!("merge {} + {} at distance {:.1}", fmt(&clusters[i]), fmt(&clusters[j]), d);
            let cj = clusters.remove(j);
            clusters[i].extend(cj);
        }
    }
  `, r`
    merge [7.0] + [7.5] at distance 0.5
    merge [1.0] + [2.0] at distance 1.0
    merge [6.0] + [7.0, 7.5] at distance 1.0
    merge [1.0, 2.0] + [6.0, 7.0, 7.5] at distance 4.0
    merge [1.0, 2.0, 6.0, 7.0, 7.5] + [15.0] at distance 7.5
  `, "Agglomerative clustering starts with every point alone and repeatedly merges the two closest clusters. The merge distances (0.5, 1, 1, 4, 7.5) form the dendrogram; cutting it just before the big jump to 4.0 gives three clusters: {1, 2}, {6, 7, 7.5} and {15}."),

  "spectral-clustering": ex(r`
    import math
    # 4 nodes: {0,1} strongly linked, {2,3} strongly linked, weak bridge 1-2
    W = [[0, 1, 0, 0], [1, 0, 0.1, 0], [0, 0.1, 0, 1], [0, 0, 1, 0]]
    d = [sum(r_) for r_ in W]
    L = [[(d[i] if i == j else 0) - W[i][j] for j in range(4)] for i in range(4)]
    v = [1.0, 0.3, -0.2, -1.0]
    for _ in range(500):                           # power iteration on (c I - L), removing the constant vector
        m = sum(v) / 4
        v = [x - m for x in v]
        w = [3 * v[i] - sum(L[i][j] * v[j] for j in range(4)) for i in range(4)]
        n = math.sqrt(sum(x * x for x in w))
        v = [x / n for x in w]
    lam = sum(v[i] * sum(L[i][j] * v[j] for j in range(4)) for i in range(4))
    print(f"Fiedler value = {lam:.4f}")
    print("Fiedler vector:", ", ".join(f"{x:+.3f}" for x in v))
    print("clusters:", [0 if x < 0 else 1 for x in v])
  `, r`
    fn main() {
        // 4 nodes: {0,1} strongly linked, {2,3} strongly linked, weak bridge 1-2
        let w = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.1, 0.0], [0.0, 0.1, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0_f64]];
        let d: Vec<f64> = w.iter().map(|r| r.iter().sum()).collect();
        let l: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| if i == j { d[i] } else { 0.0 } - w[i][j]).collect()).collect();
        let mut v = vec![1.0, 0.3, -0.2, -1.0];
        for _ in 0..500 {
            // power iteration on (c I - L), removing the constant vector
            let m = v.iter().sum::<f64>() / 4.0;
            for x in v.iter_mut() { *x -= m; }
            let nw: Vec<f64> = (0..4).map(|i| 3.0 * v[i] - (0..4).map(|j| l[i][j] * v[j]).sum::<f64>()).collect();
            let n = nw.iter().map(|x| x * x).sum::<f64>().sqrt();
            v = nw.iter().map(|x| x / n).collect();
        }
        let lam: f64 = (0..4).map(|i| v[i] * (0..4).map(|j| l[i][j] * v[j]).sum::<f64>()).sum();
        println!("Fiedler value = {:.4}", lam);
        println!("Fiedler vector: {}", v.iter().map(|x| format!("{:+.3}", x)).collect::<Vec<_>>().join(", "));
        println!("clusters: {:?}", v.iter().map(|&x| if x < 0.0 { 0 } else { 1 }).collect::<Vec<_>>());
    }
  `, r`
    Fiedler value = 0.0950
    Fiedler vector: +0.524, +0.474, -0.474, -0.524
    clusters: [1, 1, 0, 0]
  `, "The graph Laplacian's second-smallest eigenvector (the Fiedler vector) changes sign across the weakest cut. Its tiny eigenvalue says the graph is almost two separate pieces, and the signs give the two clusters."),

  "mean-shift": ex(r`
    import math
    data = [1.0, 1.2, 1.5, 2.0, 6.0, 6.3, 6.5, 7.0]
    h = 0.8
    for start in [0.0, 3.5, 4.0, 9.0]:
        x = start
        for _ in range(100):                       # move to the kernel-weighted mean of the data
            w = [math.exp(-(x - d) ** 2 / (2 * h * h)) for d in data]
            x = sum(wi * d for wi, d in zip(w, data)) / sum(w)
        print(f"start {start:.1f} -> mode {x:.3f}")
  `, r`
    fn main() {
        let data = [1.0, 1.2, 1.5, 2.0, 6.0, 6.3, 6.5, 7.0_f64];
        let h = 0.8;
        for start in [0.0, 3.5, 4.0, 9.0] {
            let mut x = start;
            for _ in 0..100 {
                // move to the kernel-weighted mean of the data
                let w: Vec<f64> = data.iter().map(|d| (-(x - d).powi(2) / (2.0 * h * h)).exp()).collect();
                x = w.iter().zip(data.iter()).map(|(wi, d)| wi * d).sum::<f64>() / w.iter().sum::<f64>();
            }
            println!("start {:.1} -> mode {:.3}", start, x);
        }
    }
  `, r`
    start 0.0 -> mode 1.402
    start 3.5 -> mode 1.402
    start 4.0 -> mode 6.434
    start 9.0 -> mode 6.434
  `, "Mean shift climbs the kernel density estimate: wherever a point starts, it ends at one of the two density peaks. Points that reach the same peak belong to the same cluster."),

  "lda-qda": ex(r`
    import math
    a = [1.0, 1.5, 2.0, 2.5, 3.0]            # class A: narrow spread
    b = [4.0, 6.0, 7.0, 9.0, 11.0]           # class B: wide spread
    mean = lambda v: sum(v) / len(v)
    var = lambda v: sum((x - mean(v)) ** 2 for x in v) / len(v)
    ma, mb, va, vb = mean(a), mean(b), var(a), var(b)
    pooled = (len(a) * va + len(b) * vb) / (len(a) + len(b))
    print(f"LDA boundary (shared variance): x = {(ma + mb) / 2:.3f}")
    # QDA: solve log N(x; ma, va) = log N(x; mb, vb) (equal priors), a quadratic in x
    A = 1 / (2 * vb) - 1 / (2 * va)
    B = ma / va - mb / vb
    Cq = mb * mb / (2 * vb) - ma * ma / (2 * va) + 0.5 * math.log(vb / va)
    roots = sorted([(-B + s * math.sqrt(B * B - 4 * A * Cq)) / (2 * A) for s in (1, -1)])
    print("QDA boundaries (own variances):", ", ".join(f"x = {r_:.3f}" for r_ in roots))
    print(f"pooled variance = {pooled:.3f}")
  `, r`
    fn main() {
        let a = [1.0, 1.5, 2.0, 2.5, 3.0_f64]; // class A: narrow spread
        let b = [4.0, 6.0, 7.0, 9.0, 11.0_f64]; // class B: wide spread
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let var = |v: &[f64]| { let m = mean(v); v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64 };
        let (ma, mb, va, vb) = (mean(&a), mean(&b), var(&a), var(&b));
        let pooled = (a.len() as f64 * va + b.len() as f64 * vb) / (a.len() + b.len()) as f64;
        println!("LDA boundary (shared variance): x = {:.3}", (ma + mb) / 2.0);
        // QDA: solve log N(x; ma, va) = log N(x; mb, vb) (equal priors), a quadratic in x
        let qa = 1.0 / (2.0 * vb) - 1.0 / (2.0 * va);
        let qb = ma / va - mb / vb;
        let qc = mb * mb / (2.0 * vb) - ma * ma / (2.0 * va) + 0.5 * (vb / va).ln();
        let disc = (qb * qb - 4.0 * qa * qc).sqrt();
        let mut roots = [(-qb + disc) / (2.0 * qa), (-qb - disc) / (2.0 * qa)];
        roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("QDA boundaries (own variances): x = {:.3}, x = {:.3}", roots[0], roots[1]);
        println!("pooled variance = {:.3}", pooled);
    }
  `, r`
    LDA boundary (shared variance): x = 4.700
    QDA boundaries (own variances): x = -0.586, x = 3.575
    pooled variance = 3.170
  `, "With a shared variance (LDA) the boundary is a single point halfway between the means. Giving each class its own variance (QDA) yields a quadratic with two boundaries: the narrow class A wins only on a short interval, the wide class B on both sides of it."),

  "gaussian-processes": ex(r`
    import math
    X, y = [-2.0, 0.0, 1.5], [0.5, 1.0, -0.4]
    ell, noise = 1.0, 1e-6
    k = lambda a, b: math.exp(-(a - b) ** 2 / (2 * ell * ell))
    K = [[k(a, b) + (noise if i == j else 0) for j, b in enumerate(X)] for i, a in enumerate(X)]
    n = 3
    L = [[0.0] * n for _ in range(n)]              # Cholesky factor of K
    for i in range(n):
        for j in range(i + 1):
            s = K[i][j] - sum(L[i][m] * L[j][m] for m in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    def solve(b):                                  # K^{-1} b via forward and back substitution
        z = [0.0] * n
        for i in range(n): z[i] = (b[i] - sum(L[i][m] * z[m] for m in range(i))) / L[i][i]
        x = [0.0] * n
        for i in reversed(range(n)): x[i] = (z[i] - sum(L[m][i] * x[m] for m in range(i + 1, n))) / L[i][i]
        return x
    alpha = solve(y)
    for xs in [-1.0, 0.0, 3.0]:
        ks = [k(xs, a) for a in X]
        mu = sum(a * b for a, b in zip(ks, alpha))
        var = k(xs, xs) - sum(a * b for a, b in zip(ks, solve(ks)))
        print(f"x={xs:+.1f}: mean={mu:+.3f} std={math.sqrt(max(var, 0)):.3f}")
  `, r`
    fn main() {
        let x = [-2.0, 0.0, 1.5_f64];
        let y = [0.5, 1.0, -0.4_f64];
        let (ell, noise) = (1.0_f64, 1e-6);
        let k = |a: f64, b: f64| (-(a - b).powi(2) / (2.0 * ell * ell)).exp();
        let n = 3;
        let mut kk = [[0.0_f64; 3]; 3];
        for i in 0..n { for j in 0..n { kk[i][j] = k(x[i], x[j]) + if i == j { noise } else { 0.0 }; } }
        let mut l = [[0.0_f64; 3]; 3]; // Cholesky factor of K
        for i in 0..n {
            for j in 0..=i {
                let s = kk[i][j] - (0..j).map(|m| l[i][m] * l[j][m]).sum::<f64>();
                l[i][j] = if i == j { s.sqrt() } else { s / l[j][j] };
            }
        }
        let solve = |b: &[f64]| -> Vec<f64> {
            // K^{-1} b via forward and back substitution
            let mut z = vec![0.0; n];
            for i in 0..n { z[i] = (b[i] - (0..i).map(|m| l[i][m] * z[m]).sum::<f64>()) / l[i][i]; }
            let mut xx = vec![0.0; n];
            for i in (0..n).rev() { xx[i] = (z[i] - (i + 1..n).map(|m| l[m][i] * xx[m]).sum::<f64>()) / l[i][i]; }
            xx
        };
        let alpha = solve(&y);
        for xs in [-1.0, 0.0, 3.0] {
            let ks: Vec<f64> = x.iter().map(|&a| k(xs, a)).collect();
            let mu: f64 = ks.iter().zip(&alpha).map(|(a, b)| a * b).sum();
            let var = k(xs, xs) - ks.iter().zip(solve(&ks).iter()).map(|(a, b)| a * b).sum::<f64>();
            println!("x={:+.1}: mean={:+.3} std={:.3}", xs, mu, var.max(0.0).sqrt());
        }
    }
  `, r`
    x=-1.0: mean=+0.905 std=0.577
    x=+0.0: mean=+1.000 std=0.001
    x=+3.0: mean=-0.244 std=0.940
  `, "At a training input (x = 0) the posterior mean reproduces the observation and the uncertainty is essentially zero; between points it interpolates with moderate uncertainty, and far from the data (x = 3) the mean falls back towards 0 with the standard deviation growing towards the prior's 1."),

  "isolation-forest": ex(join(PY_LCG, r`
    import math
    data = [4.8, 5.0, 5.1, 5.2, 4.9, 5.3, 5.05, 12.0]
    def path_length(x, pts, depth=0):
        if len(pts) <= 1 or depth > 10: return depth
        lo, hi = min(pts), max(pts)
        t = lo + rand() * (hi - lo)                # random split value
        side = [p for p in pts if (p < t) == (x < t)]
        return path_length(x, side, depth + 1)
    c = 2 * (math.log(len(data) - 1) + 0.5772) - 2 * (len(data) - 1) / len(data)
    for x in [5.0, 12.0]:
        h = sum(path_length(x, data) for _ in range(200)) / 200
        print(f"x={x}: average path={h:.2f} score={2 ** (-h / c):.3f}")
  `), join(RS_LCG, r`
    fn path_length(x: f64, pts: &[f64], depth: usize) -> usize {
        if pts.len() <= 1 || depth > 10 { return depth; }
        let lo = pts.iter().cloned().fold(f64::MAX, f64::min);
        let hi = pts.iter().cloned().fold(f64::MIN, f64::max);
        let t = lo + rand() * (hi - lo); // random split value
        let side: Vec<f64> = pts.iter().cloned().filter(|&p| (p < t) == (x < t)).collect();
        path_length(x, &side, depth + 1)
    }

    fn main() {
        let data = [4.8, 5.0, 5.1, 5.2, 4.9, 5.3, 5.05, 12.0_f64];
        let n = data.len() as f64;
        let c = 2.0 * ((n - 1.0).ln() + 0.5772) - 2.0 * (n - 1.0) / n;
        for x in [5.0, 12.0] {
            let h = (0..200).map(|_| path_length(x, &data, 0)).sum::<usize>() as f64 / 200.0;
            println!("x={:.1}: average path={:.2} score={:.3}", x, h, 2f64.powf(-h / c));
        }
    }
  `), r`
    x=5.0: average path=4.78 score=0.366
    x=12.0: average path=1.04 score=0.803
  `, "Random splits isolate the far-away value 12 after very few cuts, while a typical value needs many. The short average path translates into a high anomaly score."),

  "association-rules": ex(r`
    baskets = [{"bread", "butter"}, {"bread", "milk"}, {"bread", "butter", "jam"}, {"milk"}, {"bread", "butter", "milk"}]
    n = len(baskets)
    supp = lambda s: sum(s <= b for b in baskets) / n
    X, Y = {"bread"}, {"butter"}
    s_xy, s_x, s_y = supp(X | Y), supp(X), supp(Y)
    print(f"support({{bread, butter}}) = {s_xy:.2f}")
    print(f"confidence(bread -> butter) = {s_xy / s_x:.2f}")
    print(f"lift = {s_xy / (s_x * s_y):.3f}")
  `, r`
    fn main() {
        let baskets: Vec<Vec<&str>> = vec![vec!["bread", "butter"], vec!["bread", "milk"], vec!["bread", "butter", "jam"], vec!["milk"], vec!["bread", "butter", "milk"]];
        let n = baskets.len() as f64;
        let supp = |s: &[&str]| baskets.iter().filter(|b| s.iter().all(|i| b.contains(i))).count() as f64 / n;
        let (s_xy, s_x, s_y) = (supp(&["bread", "butter"]), supp(&["bread"]), supp(&["butter"]));
        println!("support({{bread, butter}}) = {:.2}", s_xy);
        println!("confidence(bread -> butter) = {:.2}", s_xy / s_x);
        println!("lift = {:.3}", s_xy / (s_x * s_y));
    }
  `, r`
    support({bread, butter}) = 0.60
    confidence(bread -> butter) = 0.75
    lift = 1.250
  `, "Three of five baskets contain both items (support 0.6); of the four with bread, three also have butter (confidence 0.75). Lift above 1 means butter is more likely when bread is bought than in general."),

  "collaborative-filtering": ex(r`
    R = [[5, 4, 0], [4, 0, 3], [2, 2, 0]]          # 0 = unknown rating
    u, v = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]        # rank-1 factors
    lam = 0.1
    for _ in range(50):                            # alternating least squares
        u = [sum(R[i][j] * v[j] for j in range(3) if R[i][j]) / (sum(v[j] ** 2 for j in range(3) if R[i][j]) + lam) for i in range(3)]
        v = [sum(R[i][j] * u[i] for i in range(3) if R[i][j]) / (sum(u[i] ** 2 for i in range(3) if R[i][j]) + lam) for j in range(3)]
    for i in range(3):
        print(" ".join(f"{u[i] * v[j]:.2f}" + ("*" if R[i][j] == 0 else " ") for j in range(3)).rstrip())
    print("* = predicted for a missing rating")
  `, r`
    fn main() {
        let r = [[5.0, 4.0, 0.0], [4.0, 0.0, 3.0], [2.0, 2.0, 0.0_f64]]; // 0 = unknown rating
        let (mut u, mut v) = ([1.0_f64; 3], [1.0_f64; 3]); // rank-1 factors
        let lam = 0.1;
        for _ in 0..50 {
            // alternating least squares
            for i in 0..3 {
                let num: f64 = (0..3).filter(|&j| r[i][j] != 0.0).map(|j| r[i][j] * v[j]).sum();
                let den: f64 = (0..3).filter(|&j| r[i][j] != 0.0).map(|j| v[j] * v[j]).sum();
                u[i] = num / (den + lam);
            }
            for j in 0..3 {
                let num: f64 = (0..3).filter(|&i| r[i][j] != 0.0).map(|i| r[i][j] * u[i]).sum();
                let den: f64 = (0..3).filter(|&i| r[i][j] != 0.0).map(|i| u[i] * u[i]).sum();
                v[j] = num / (den + lam);
            }
        }
        for i in 0..3 {
            let row: Vec<String> = (0..3).map(|j| format!("{:.2}{}", u[i] * v[j], if r[i][j] == 0.0 { "*" } else { " " })).collect();
            println!("{}", row.join(" ").trim_end());
        }
        println!("* = predicted for a missing rating");
    }
  `, r`
    4.87  4.03  3.59*
    3.97  3.28* 2.92
    2.14  1.77  1.58*
    * = predicted for a missing rating
  `, "Alternating least squares fits one user factor and one item factor from the known ratings only; their products also fill the unknown cells. The ratings here follow one dominant taste pattern, so a single factor fits them well; real systems learn tens to hundreds of factors."),
};

export const modelEvaluationCode: Record<string, CodeExample> = {
  "confusion-matrix": ex(r`
    y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy={(tp + tn) / len(y_true):.2f} precision={precision:.2f} recall={recall:.2f} F1={2 * precision * recall / (precision + recall):.3f}")
  `, r`
    fn main() {
        let y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0];
        let y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0];
        let count = |t: i32, p: i32| y_true.iter().zip(y_pred.iter()).filter(|(a, b)| **a == t && **b == p).count() as f64;
        let (tp, fp, fnn, tn) = (count(1, 1), count(0, 1), count(1, 0), count(0, 0));
        let (precision, recall) = (tp / (tp + fp), tp / (tp + fnn));
        println!("TP={} FP={} FN={} TN={}", tp, fp, fnn, tn);
        println!("accuracy={:.2} precision={:.2} recall={:.2} F1={:.3}", (tp + tn) / 10.0, precision, recall, 2.0 * precision * recall / (precision + recall));
    }
  `, r`
    TP=4 FP=1 FN=1 TN=4
    accuracy=0.80 precision=0.80 recall=0.80 F1=0.800
  `, "The four counts summarise every kind of right and wrong answer; precision, recall and F1 are ratios of them. Here one positive is missed (FN) and one negative is falsely flagged (FP)."),

  "roc-auc": ex(r`
    y = [1, 1, 0, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1]
    pos = [si for si, yi in zip(s, y) if yi == 1]
    neg = [si for si, yi in zip(s, y) if yi == 0]
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    print(f"AUC = {auc:.4f}  (share of positive/negative pairs ranked correctly)")
    for t in [0.75, 0.5, 0.2]:
        tpr = sum(si >= t for si in pos) / len(pos)
        fpr = sum(si >= t for si in neg) / len(neg)
        print(f"threshold {t}: TPR={tpr:.2f} FPR={fpr:.2f}")
  `, r`
    fn main() {
        let y = [1, 1, 0, 1, 0, 0, 1, 0];
        let s = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1_f64];
        let pos: Vec<f64> = s.iter().zip(y.iter()).filter(|(_, &yi)| yi == 1).map(|(si, _)| *si).collect();
        let neg: Vec<f64> = s.iter().zip(y.iter()).filter(|(_, &yi)| yi == 0).map(|(si, _)| *si).collect();
        let mut good = 0.0;
        for p in &pos { for q in &neg { good += if p > q { 1.0 } else if p == q { 0.5 } else { 0.0 }; } }
        println!("AUC = {:.4}  (share of positive/negative pairs ranked correctly)", good / (pos.len() * neg.len()) as f64);
        for t in [0.75, 0.5, 0.2] {
            let tpr = pos.iter().filter(|&&v| v >= t).count() as f64 / pos.len() as f64;
            let fpr = neg.iter().filter(|&&v| v >= t).count() as f64 / neg.len() as f64;
            println!("threshold {}: TPR={:.2} FPR={:.2}", t, tpr, fpr);
        }
    }
  `, r`
    AUC = 0.7500  (share of positive/negative pairs ranked correctly)
    threshold 0.75: TPR=0.50 FPR=0.00
    threshold 0.5: TPR=0.75 FPR=0.50
    threshold 0.2: TPR=1.00 FPR=0.75
  `, "AUC equals the probability that a random positive gets a higher score than a random negative (12 of 16 pairs here). Each threshold gives one point (FPR, TPR) on the ROC curve."),

  "precision-recall-curve": ex(r`
    y = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    s = [0.95, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.3, 0.2, 0.1]
    order = sorted(range(len(s)), key=lambda i: -s[i])
    tp, ap, P = 0, 0.0, sum(y)
    for k, i in enumerate(order, 1):
        if y[i]:
            tp += 1
            ap += tp / k / P                       # average precision: precision at each recall step
        print(f"top {k:2d}: precision={tp / k:.3f} recall={tp / P:.2f}")
    print(f"average precision = {ap:.4f}")
  `, r`
    fn main() {
        let y = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0];
        let s = [0.95, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.3, 0.2, 0.1_f64];
        let mut order: Vec<usize> = (0..s.len()).collect();
        order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
        let p = y.iter().sum::<i32>() as f64;
        let (mut tp, mut ap) = (0.0, 0.0);
        for (k0, &i) in order.iter().enumerate() {
            let k = (k0 + 1) as f64;
            if y[i] == 1 {
                tp += 1.0;
                ap += tp / k / p; // average precision: precision at each recall step
            }
            println!("top {:2}: precision={:.3} recall={:.2}", k0 + 1, tp / k, tp / p);
        }
        println!("average precision = {:.4}", ap);
    }
  `, r`
    top  1: precision=1.000 recall=0.25
    top  2: precision=1.000 recall=0.50
    top  3: precision=0.667 recall=0.50
    top  4: precision=0.750 recall=0.75
    top  5: precision=0.600 recall=0.75
    top  6: precision=0.500 recall=0.75
    top  7: precision=0.571 recall=1.00
    top  8: precision=0.500 recall=1.00
    top  9: precision=0.444 recall=1.00
    top 10: precision=0.400 recall=1.00
    average precision = 0.8304
  `, "Walking down the ranked list, recall can only rise while precision falls whenever a negative is included. Average precision summarises the curve by averaging the precision at the rank of each positive."),

  calibration: ex(r`
    probs = [0.1, 0.15, 0.2, 0.35, 0.4, 0.6, 0.65, 0.8, 0.85, 0.9]
    y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    brier = sum((p - t) ** 2 for p, t in zip(probs, y)) / len(y)
    ece = 0.0
    for lo in [0.0, 0.5]:                          # two bins: [0, 0.5) and [0.5, 1]
        idx = [i for i, p in enumerate(probs) if lo <= p < lo + 0.5 or (lo == 0.5 and p == 1.0)]
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        ece += len(idx) / len(y) * abs(acc - conf)
        print(f"bin [{lo}, {lo + 0.5}): mean predicted={conf:.3f} observed rate={acc:.3f}")
    print(f"Brier score = {brier:.4f}, ECE = {ece:.4f}")
  `, r`
    fn main() {
        let probs = [0.1, 0.15, 0.2, 0.35, 0.4, 0.6, 0.65, 0.8, 0.85, 0.9_f64];
        let y = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0_f64];
        let n = y.len() as f64;
        let brier = probs.iter().zip(y.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n;
        let mut ece = 0.0;
        for lo in [0.0, 0.5] {
            // two bins: [0, 0.5) and [0.5, 1]
            let idx: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] >= lo && probs[i] < lo + 0.5).collect();
            let conf = idx.iter().map(|&i| probs[i]).sum::<f64>() / idx.len() as f64;
            let acc = idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
            ece += idx.len() as f64 / n * (acc - conf).abs();
            println!("bin [{:.1}, {:.1}): mean predicted={:.3} observed rate={:.3}", lo, lo + 0.5, conf, acc);
        }
        println!("Brier score = {:.4}, ECE = {:.4}", brier, ece);
    }
  `, r`
    bin [0.0, 0.5): mean predicted=0.240 observed rate=0.400
    bin [0.5, 1.0): mean predicted=0.760 observed rate=0.800
    Brier score = 0.1710, ECE = 0.1000
  `, "A calibrated model's predicted probabilities match observed frequencies. Here both bins under-predict: about 0.24 predicted versus 0.4 observed in the low bin. The expected calibration error averages these gaps, weighted by bin size."),

  "regression-metrics": ex(r`
    import math
    y = [100.0, 150.0, 200.0, 250.0, 300.0]
    p = [110.0, 140.0, 210.0, 230.0, 330.0]
    e = [a - b for a, b in zip(y, p)]
    mae = sum(abs(v) for v in e) / len(e)
    mse = sum(v * v for v in e) / len(e)
    mean = sum(y) / len(y)
    r2 = 1 - sum(v * v for v in e) / sum((t - mean) ** 2 for t in y)
    mape = sum(abs(v) / t for v, t in zip(e, y)) / len(e)
    print(f"MAE={mae:.2f} MSE={mse:.2f} RMSE={math.sqrt(mse):.3f} R^2={r2:.4f} MAPE={mape:.2%}")
  `, r`
    fn main() {
        let y = [100.0, 150.0, 200.0, 250.0, 300.0_f64];
        let p = [110.0, 140.0, 210.0, 230.0, 330.0_f64];
        let e: Vec<f64> = y.iter().zip(p.iter()).map(|(a, b)| a - b).collect();
        let n = e.len() as f64;
        let mae = e.iter().map(|v| v.abs()).sum::<f64>() / n;
        let mse = e.iter().map(|v| v * v).sum::<f64>() / n;
        let mean = y.iter().sum::<f64>() / n;
        let r2 = 1.0 - e.iter().map(|v| v * v).sum::<f64>() / y.iter().map(|t| (t - mean).powi(2)).sum::<f64>();
        let mape = e.iter().zip(y.iter()).map(|(v, t)| v.abs() / t).sum::<f64>() / n;
        println!("MAE={:.2} MSE={:.2} RMSE={:.3} R^2={:.4} MAPE={:.2}%", mae, mse, mse.sqrt(), r2, 100.0 * mape);
    }
  `, r`
    MAE=16.00 MSE=320.00 RMSE=17.889 R^2=0.9360 MAPE=7.93%
  `, "MAE is the typical absolute miss; RMSE weights the single 30-unit miss more heavily; R² compares the errors with simply predicting the mean; MAPE expresses errors relative to each true value."),

  "cross-validation": ex(r`
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys = [2.3, 4.1, 6.2, 7.8, 10.3, 11.9, 14.2, 15.8, 18.1, 20.2]
    k = 5
    scores = []
    for f in range(k):
        test = [i for i in range(len(xs)) if i % k == f]
        train = [i for i in range(len(xs)) if i % k != f]
        mx = sum(xs[i] for i in train) / len(train); my = sum(ys[i] for i in train) / len(train)
        b = sum((xs[i] - mx) * (ys[i] - my) for i in train) / sum((xs[i] - mx) ** 2 for i in train)
        a = my - b * mx
        mse = sum((ys[i] - a - b * xs[i]) ** 2 for i in test) / len(test)
        scores.append(mse)
        print(f"fold {f + 1}: test points {test} MSE={mse:.4f}")
    print(f"CV estimate: {sum(scores) / k:.4f}")
  `, r`
    fn main() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0_f64];
        let ys = [2.3, 4.1, 6.2, 7.8, 10.3, 11.9, 14.2, 15.8, 18.1, 20.2_f64];
        let k = 5;
        let mut scores = Vec::new();
        for f in 0..k {
            let test: Vec<usize> = (0..xs.len()).filter(|i| i % k == f).collect();
            let train: Vec<usize> = (0..xs.len()).filter(|i| i % k != f).collect();
            let n = train.len() as f64;
            let mx = train.iter().map(|&i| xs[i]).sum::<f64>() / n;
            let my = train.iter().map(|&i| ys[i]).sum::<f64>() / n;
            let b = train.iter().map(|&i| (xs[i] - mx) * (ys[i] - my)).sum::<f64>() / train.iter().map(|&i| (xs[i] - mx).powi(2)).sum::<f64>();
            let a = my - b * mx;
            let mse = test.iter().map(|&i| (ys[i] - a - b * xs[i]).powi(2)).sum::<f64>() / test.len() as f64;
            scores.push(mse);
            println!("fold {}: test points {:?} MSE={:.4}", f + 1, test, mse);
        }
        println!("CV estimate: {:.4}", scores.iter().sum::<f64>() / k as f64);
    }
  `, r`
    fold 1: test points [0, 5] MSE=0.0402
    fold 2: test points [1, 6] MSE=0.0115
    fold 3: test points [2, 7] MSE=0.0519
    fold 4: test points [3, 8] MSE=0.0634
    fold 5: test points [4, 9] MSE=0.0714
    CV estimate: 0.0477
  `, "Every point is used for testing exactly once and for training k − 1 times. Averaging the five fold errors gives a more stable estimate of generalisation error than a single split."),

  "hyperparameter-tuning": ex(r`
    train_x, train_y = [1.0, 2.0, 3.0, 4.0], [1.2, 1.9, 3.2, 3.9]
    val_x, val_y = [5.0, 6.0], [5.1, 5.8]
    best = None
    for lam in [0.0, 0.3, 1.0, 3.0, 10.0]:        # grid search over the ridge strength
        w = sum(x * y for x, y in zip(train_x, train_y)) / (sum(x * x for x in train_x) + lam)
        val = sum((y - w * x) ** 2 for x, y in zip(val_x, val_y)) / len(val_x)
        print(f"lambda={lam:4.1f}  w={w:.4f}  val MSE={val:.4f}")
        if best is None or val < best[1]: best = (lam, val)
    print(f"selected lambda = {best[0]}")
  `, r`
    fn main() {
        let (train_x, train_y) = ([1.0, 2.0, 3.0, 4.0_f64], [1.2, 1.9, 3.2, 3.9_f64]);
        let (val_x, val_y) = ([5.0, 6.0_f64], [5.1, 5.8_f64]);
        let mut best = (0.0, f64::MAX);
        for lam in [0.0, 0.3, 1.0, 3.0, 10.0] {
            // grid search over the ridge strength
            let w = train_x.iter().zip(train_y.iter()).map(|(x, y)| x * y).sum::<f64>() / (train_x.iter().map(|x| x * x).sum::<f64>() + lam);
            let val = val_x.iter().zip(val_y.iter()).map(|(x, y)| (y - w * x).powi(2)).sum::<f64>() / val_x.len() as f64;
            println!("lambda={:4.1}  w={:.4}  val MSE={:.4}", lam, w, val);
            if val < best.1 { best = (lam, val); }
        }
        println!("selected lambda = {:.1}", best.0);
    }
  `, r`
    lambda= 0.0  w=1.0067  val MSE=0.0310
    lambda= 0.3  w=0.9967  val MSE=0.0230
    lambda= 1.0  w=0.9742  val MSE=0.0272
    lambda= 3.0  w=0.9152  val MSE=0.1852
    lambda=10.0  w=0.7550  val MSE=1.6843
    selected lambda = 0.3
  `, "Each candidate λ is trained on the training split and scored on the validation split; the lowest validation error wins. Random search and Bayesian optimisation explore the same space more efficiently when there are several hyperparameters."),
};
