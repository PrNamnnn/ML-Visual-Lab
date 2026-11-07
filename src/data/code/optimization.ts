import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

export const optimizationCode: Record<string, CodeExample> = {
  "objective-functions": ex(r`
    # house size (100 m^2) -> price (100k); fit price = w * size
    xs = [0.6, 1.0, 1.5, 2.1, 2.5]
    ys = [1.9, 3.1, 4.6, 6.4, 7.2]
    n = len(xs)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))

    def J(w, lam):  # mean squared error + L2 penalty
        return sum((y - w * x) ** 2 for x, y in zip(xs, ys)) / n + lam * w * w

    for lam in [0.0, 0.5]:
        w = sxy / (sxx + n * lam)               # closed-form minimiser
        print(f"lambda={lam:.1f}: w={w:.4f} J={J(w, lam):.4f}")
    w = min(2.8, max(0.0, sxy / sxx))           # constraint 0 <= w <= 2.8
    print(f"constrained w={w:.4f}")
  `, r`
    fn main() {
        // house size (100 m^2) -> price (100k); fit price = w * size
        let xs = [0.6, 1.0, 1.5, 2.1, 2.5];
        let ys = [1.9, 3.1, 4.6, 6.4, 7.2];
        let n = xs.len() as f64;
        let sxx: f64 = xs.iter().map(|x| x * x).sum();
        let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        // mean squared error + L2 penalty
        let j = |w: f64, lam: f64| -> f64 {
            xs.iter().zip(ys.iter()).map(|(x, y)| (y - w * x).powi(2)).sum::<f64>() / n + lam * w * w
        };
        for lam in [0.0, 0.5] {
            let w = sxy / (sxx + n * lam); // closed-form minimiser
            println!("lambda={:.1}: w={:.4} J={:.4}", lam, w, j(w, lam));
        }
        let w = (sxy / sxx).max(0.0).min(2.8); // constraint 0 <= w <= 2.8
        println!("constrained w={:.4}", w);
    }
  `, r`
    lambda=0.0: w=2.9839 J=0.0253
    lambda=0.5: w=2.5391 J=3.8134
    constrained w=2.8000
  `, "The same data gives different 'best' slopes depending on the objective: the penalty pulls w toward 0, and the constraint simply forbids values above 2.8. Because J is a convex parabola in w, the constrained optimum is the unconstrained one clipped to the feasible interval."),

  "convex-optimization": ex(r`
    import math

    def is_convex_on_samples(f, pts):
        # check f(t a + (1-t) b) <= t f(a) + (1-t) f(b) for many pairs and t
        for a in pts:
            for b in pts:
                for t in [0.25, 0.5, 0.75]:
                    lhs = f(t * a + (1 - t) * b)
                    rhs = t * f(a) + (1 - t) * f(b)
                    if lhs > rhs + 1e-12:
                        return False
        return True

    pts = [-3 + 0.5 * i for i in range(13)]
    print("x^2     convex:", is_convex_on_samples(lambda x: x * x, pts))
    print("|x|     convex:", is_convex_on_samples(abs, pts))
    print("sin(x)  convex:", is_convex_on_samples(math.sin, pts))
    print("x^4-x^2 convex:", is_convex_on_samples(lambda x: x**4 - x**2, pts))
  `, r`
    fn is_convex_on_samples(f: &dyn Fn(f64) -> f64, pts: &[f64]) -> bool {
        // check f(t a + (1-t) b) <= t f(a) + (1-t) f(b) for many pairs and t
        for &a in pts {
            for &b in pts {
                for t in [0.25, 0.5, 0.75] {
                    let lhs = f(t * a + (1.0 - t) * b);
                    let rhs = t * f(a) + (1.0 - t) * f(b);
                    if lhs > rhs + 1e-12 { return false; }
                }
            }
        }
        true
    }

    fn main() {
        let pts: Vec<f64> = (0..13).map(|i| -3.0 + 0.5 * i as f64).collect();
        let cap = |b: bool| if b { "True" } else { "False" };
        println!("x^2     convex: {}", cap(is_convex_on_samples(&|x| x * x, &pts)));
        println!("|x|     convex: {}", cap(is_convex_on_samples(&|x: f64| x.abs(), &pts)));
        println!("sin(x)  convex: {}", cap(is_convex_on_samples(&|x: f64| x.sin(), &pts)));
        println!("x^4-x^2 convex: {}", cap(is_convex_on_samples(&|x: f64| x.powi(4) - x * x, &pts)));
    }
  `, r`
    x^2     convex: True
    |x|     convex: True
    sin(x)  convex: False
    x^4-x^2 convex: False
  `, "A function is convex when every chord lies on or above the graph. Sampling that inequality cannot prove convexity, but one violation disproves it: sin and the double-well x⁴ − x² both fail, so gradient descent on them can get stuck in a local minimum."),

  "constrained-optimization": ex(r`
    import math

    def project_kkt(a, b, r):
        # minimise (x-a)^2 + (y-b)^2 subject to x^2 + y^2 <= r^2
        n = math.hypot(a, b)
        if n <= r:                       # target already feasible
            return a, b, 0.0             # constraint inactive -> mu = 0
        return r * a / n, r * b / n, n / r - 1  # on the boundary, mu > 0

    for a, b in [(3.0, 4.0), (1.0, 1.0)]:
        x, y, mu = project_kkt(a, b, 2.5)
        state = "active" if mu > 0 else "inactive"
        print(f"target ({a}, {b}): x*=({x:.3f}, {y:.3f}) mu={mu:.3f} {state}")
  `, r`
    // minimise (x-a)^2 + (y-b)^2 subject to x^2 + y^2 <= r^2
    fn project_kkt(a: f64, b: f64, r: f64) -> (f64, f64, f64) {
        let n = a.hypot(b);
        if n <= r {
            return (a, b, 0.0); // target already feasible -> constraint inactive, mu = 0
        }
        (r * a / n, r * b / n, n / r - 1.0) // on the boundary, mu > 0
    }

    fn main() {
        for (a, b) in [(3.0, 4.0), (1.0, 1.0)] {
            let (x, y, mu) = project_kkt(a, b, 2.5);
            let state = if mu > 0.0 { "active" } else { "inactive" };
            println!("target ({:.1}, {:.1}): x*=({:.3}, {:.3}) mu={:.3} {}", a, b, x, y, mu, state);
        }
    }
  `, r`
    target (3.0, 4.0): x*=(1.500, 2.000) mu=1.000 active
    target (1.0, 1.0): x*=(1.000, 1.000) mu=0.000 inactive
  `, "KKT conditions in action: when the target lies outside the disk the solution sits on the boundary with a positive multiplier μ; when it lies inside, the constraint does nothing and μ = 0 (complementary slackness μ·g = 0)."),

  "gradient-descent": ex(r`
    # minimise f(x, y) = x^2 + 3 y^2
    def grad(x, y):
        return 2 * x, 6 * y

    x, y, lr = 2.0, 1.0, 0.1
    for step in range(1, 21):
        gx, gy = grad(x, y)
        x, y = x - lr * gx, y - lr * gy
        if step % 5 == 0:
            f = x * x + 3 * y * y
            print(f"step {step:2d}: x={x:.4f} y={y:.4f} f={f:.6f}")
  `, r`
    // minimise f(x, y) = x^2 + 3 y^2
    fn grad(x: f64, y: f64) -> (f64, f64) { (2.0 * x, 6.0 * y) }

    fn main() {
        let (mut x, mut y, lr) = (2.0_f64, 1.0_f64, 0.1);
        for step in 1..=20 {
            let (gx, gy) = grad(x, y);
            x -= lr * gx;
            y -= lr * gy;
            if step % 5 == 0 {
                let f = x * x + 3.0 * y * y;
                println!("step {:2}: x={:.4} y={:.4} f={:.6}", step, x, y, f);
            }
        }
    }
  `, r`
    step  5: x=0.6554 y=0.0102 f=0.429811
    step 10: x=0.2147 y=0.0001 f=0.046117
    step 15: x=0.0704 y=0.0000 f=0.004952
    step 20: x=0.0231 y=0.0000 f=0.000532
  `, "Each step moves against the gradient. The steep y-direction (curvature 6) shrinks by a factor 0.4 per step while the flat x-direction only shrinks by 0.8 — gradient descent is slowest along low-curvature directions."),

  "stochastic-gradient-descent": ex(r`
    # fit y = w x + b with one example per update (SGD)
    data = [(0.0, 1.1), (1.0, 2.9), (2.0, 5.2), (3.0, 7.1), (4.0, 8.8)]
    order = [3, 0, 4, 1, 2]          # a fixed "shuffled" order
    w, b, lr = 0.0, 0.0, 0.02
    for epoch in range(1, 201):
        for i in order:
            x, y = data[i]
            err = (w * x + b) - y    # gradient of 0.5*err^2
            w -= lr * err * x
            b -= lr * err
        if epoch in (1, 10, 200):
            print(f"epoch {epoch:3d}: w={w:.3f} b={b:.3f}")
  `, r`
    fn main() {
        // fit y = w x + b with one example per update (SGD)
        let data = [(0.0, 1.1), (1.0, 2.9), (2.0, 5.2), (3.0, 7.1), (4.0, 8.8)];
        let order = [3, 0, 4, 1, 2]; // a fixed "shuffled" order
        let (mut w, mut b, lr) = (0.0_f64, 0.0_f64, 0.02);
        for epoch in 1..=200 {
            for &i in order.iter() {
                let (x, y) = data[i];
                let err = (w * x + b) - y; // gradient of 0.5*err^2
                w -= lr * err * x;
                b -= lr * err;
            }
            if epoch == 1 || epoch == 10 || epoch == 200 {
                println!("epoch {:3}: w={:.3} b={:.3}", epoch, w, b);
            }
        }
    }
  `, r`
    epoch   1: w=1.127 b=0.389
    epoch  10: w=2.059 b=0.808
    epoch 200: w=1.954 b=1.111
  `, "SGD updates the parameters after every single example instead of after the whole dataset. It reaches the least-squares line (w ≈ 1.95, b ≈ 1.12) closely, but with a constant learning rate it keeps jittering around it rather than settling exactly."),

  momentum: ex(r`
    # f(x, y) = x^2 + 10 y^2 : a narrow valley
    def grad(p):
        return [2 * p[0], 20 * p[1]]

    def run(beta, steps=60, lr=0.02):
        p, v = [3.0, 1.0], [0.0, 0.0]
        for _ in range(steps):
            g = grad(p)
            v = [beta * v[i] + g[i] for i in range(2)]   # velocity
            p = [p[i] - lr * v[i] for i in range(2)]
        return p[0] ** 2 + 10 * p[1] ** 2

    print(f"plain GD   f={run(0.0):.10f}")
    print(f"momentum   f={run(0.7):.10f}")
  `, r`
    // f(x, y) = x^2 + 10 y^2 : a narrow valley
    fn grad(p: [f64; 2]) -> [f64; 2] { [2.0 * p[0], 20.0 * p[1]] }

    fn run(beta: f64, steps: usize, lr: f64) -> f64 {
        let (mut p, mut v) = ([3.0, 1.0], [0.0, 0.0]);
        for _ in 0..steps {
            let g = grad(p);
            for i in 0..2 {
                v[i] = beta * v[i] + g[i]; // velocity
                p[i] -= lr * v[i];
            }
        }
        p[0] * p[0] + 10.0 * p[1] * p[1]
    }

    fn main() {
        println!("plain GD   f={:.10}", run(0.0, 60, 0.02));
        println!("momentum   f={:.10}", run(0.7, 60, 0.02));
    }
  `, r`
    plain GD   f=0.0671105000
    momentum   f=0.0000000148
  `, "Momentum accumulates a velocity, so consistent gradient directions (along the long, flat axis of the valley) build up speed while oscillating ones partly cancel. With the same learning rate and 60 steps, β = 0.7 reaches a loss millions of times smaller than plain gradient descent. Too much momentum (β = 0.9 here) overshoots and oscillates, so β is a hyperparameter to tune."),

  "adaptive-methods": ex(r`
    import math
    # f(x, y) = x^2 + 100 y^2 : very different curvatures
    def grad(x, y):
        return 2 * x, 200 * y

    def adagrad(steps=100, lr=0.5, eps=1e-8):
        x, y, sx, sy = 3.0, 1.0, 0.0, 0.0
        for _ in range(steps):
            gx, gy = grad(x, y)
            sx += gx * gx; sy += gy * gy          # sum of squared gradients
            x -= lr * gx / (math.sqrt(sx) + eps)
            y -= lr * gy / (math.sqrt(sy) + eps)
        return x, y

    def rmsprop(steps=100, lr=0.05, rho=0.9, eps=1e-8):
        x, y, sx, sy = 3.0, 1.0, 0.0, 0.0
        for _ in range(steps):
            gx, gy = grad(x, y)
            sx = rho * sx + (1 - rho) * gx * gx   # moving average instead
            sy = rho * sy + (1 - rho) * gy * gy
            x -= lr * gx / (math.sqrt(sx) + eps)
            y -= lr * gy / (math.sqrt(sy) + eps)
        return x, y

    for name, (x, y) in [("AdaGrad", adagrad()), ("RMSProp", rmsprop())]:
        print(f"{name}: x={x:.4f} y={y:.4f}")
  `, r`
    // f(x, y) = x^2 + 100 y^2 : very different curvatures
    fn grad(x: f64, y: f64) -> (f64, f64) { (2.0 * x, 200.0 * y) }

    fn adagrad(steps: usize, lr: f64, eps: f64) -> (f64, f64) {
        let (mut x, mut y, mut sx, mut sy) = (3.0_f64, 1.0_f64, 0.0_f64, 0.0_f64);
        for _ in 0..steps {
            let (gx, gy) = grad(x, y);
            sx += gx * gx; sy += gy * gy; // sum of squared gradients
            x -= lr * gx / (sx.sqrt() + eps);
            y -= lr * gy / (sy.sqrt() + eps);
        }
        (x, y)
    }

    fn rmsprop(steps: usize, lr: f64, rho: f64, eps: f64) -> (f64, f64) {
        let (mut x, mut y, mut sx, mut sy) = (3.0_f64, 1.0_f64, 0.0_f64, 0.0_f64);
        for _ in 0..steps {
            let (gx, gy) = grad(x, y);
            sx = rho * sx + (1.0 - rho) * gx * gx; // moving average instead
            sy = rho * sy + (1.0 - rho) * gy * gy;
            x -= lr * gx / (sx.sqrt() + eps);
            y -= lr * gy / (sy.sqrt() + eps);
        }
        (x, y)
    }

    fn main() {
        let (x, y) = adagrad(100, 0.5, 1e-8);
        println!("AdaGrad: x={:.4} y={:.4}", x, y);
        let (x, y) = rmsprop(100, 0.05, 0.9, 1e-8);
        println!("RMSProp: x={:.4} y={:.4}", x, y);
    }
  `, r`
    AdaGrad: x=0.0007 y=0.0000
    RMSProp: x=0.0000 y=0.0245
  `, "Both methods divide each coordinate's step by the size of its recent gradients, so the steep y-direction and the flat x-direction move at similar speeds. AdaGrad's accumulated sum only grows, so its steps keep shrinking; RMSProp forgets old gradients and keeps a steady pace — which is why, with a constant learning rate, it ends up hovering around the minimum (y ≈ 0.02) instead of landing exactly on it."),

  adam: ex(r`
    import math
    # minimise f(x) = (x - 3)^2 with Adam
    x, m, v = 0.0, 0.0, 0.0
    lr, b1, b2, eps = 0.3, 0.9, 0.999, 1e-8
    for t in range(1, 101):
        g = 2 * (x - 3)
        m = b1 * m + (1 - b1) * g          # first moment (mean)
        v = b2 * v + (1 - b2) * g * g      # second moment (uncentred variance)
        m_hat = m / (1 - b1 ** t)          # bias correction
        v_hat = v / (1 - b2 ** t)
        x -= lr * m_hat / (math.sqrt(v_hat) + eps)
        if t in (1, 2, 10, 50, 100):
            print(f"t={t:3d}: x={x:.4f}")
  `, r`
    fn main() {
        // minimise f(x) = (x - 3)^2 with Adam
        let (mut x, mut m, mut v) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (lr, b1, b2, eps) = (0.3, 0.9_f64, 0.999_f64, 1e-8);
        for t in 1..=100 {
            let g = 2.0 * (x - 3.0);
            m = b1 * m + (1.0 - b1) * g; // first moment (mean)
            v = b2 * v + (1.0 - b2) * g * g; // second moment (uncentred variance)
            let m_hat = m / (1.0 - b1.powi(t)); // bias correction
            let v_hat = v / (1.0 - b2.powi(t));
            x -= lr * m_hat / (v_hat.sqrt() + eps);
            if [1, 2, 10, 50, 100].contains(&t) {
                println!("t={:3}: x={:.4}", t, x);
            }
        }
    }
  `, r`
    t=  1: x=0.3000
    t=  2: x=0.5988
    t= 10: x=2.7713
    t= 50: x=3.0145
    t=100: x=2.9912
  `, "Thanks to bias correction the very first Adam step has size ≈ lr regardless of the gradient's scale (x jumps to 0.3). Momentum then carries x past 3 and back before it settles — Adam's step size is set by lr, not by the raw gradient magnitude."),

  "learning-rate-scheduling": ex(r`
    import math
    base, total, warm = 0.1, 100, 10

    def step_decay(t):            # halve every 30 steps
        return base * 0.5 ** (t // 30)

    def cosine(t):                # smooth decay to 0
        return 0.5 * base * (1 + math.cos(math.pi * t / total))

    def warmup_cosine(t):         # linear warm-up, then cosine
        if t < warm:
            return base * (t + 1) / warm
        return 0.5 * base * (1 + math.cos(math.pi * (t - warm) / (total - warm)))

    for t in [0, 5, 30, 60, 99]:
        print(f"t={t:2d}  step={step_decay(t):.4f}  cos={cosine(t):.4f}  warm+cos={warmup_cosine(t):.4f}")
  `, r`
    use std::f64::consts::PI;
    const BASE: f64 = 0.1;
    const TOTAL: f64 = 100.0;
    const WARM: f64 = 10.0;

    fn step_decay(t: u32) -> f64 { BASE * 0.5_f64.powi((t / 30) as i32) } // halve every 30 steps
    fn cosine(t: u32) -> f64 { 0.5 * BASE * (1.0 + (PI * t as f64 / TOTAL).cos()) } // smooth decay to 0
    fn warmup_cosine(t: u32) -> f64 {
        let t = t as f64;
        if t < WARM { return BASE * (t + 1.0) / WARM; } // linear warm-up
        0.5 * BASE * (1.0 + (PI * (t - WARM) / (TOTAL - WARM)).cos())
    }

    fn main() {
        for t in [0, 5, 30, 60, 99] {
            println!("t={:2}  step={:.4}  cos={:.4}  warm+cos={:.4}", t, step_decay(t), cosine(t), warmup_cosine(t));
        }
    }
  `, r`
    t= 0  step=0.1000  cos=0.1000  warm+cos=0.0100
    t= 5  step=0.1000  cos=0.0994  warm+cos=0.0600
    t=30  step=0.0500  cos=0.0794  warm+cos=0.0883
    t=60  step=0.0250  cos=0.0345  warm+cos=0.0413
    t=99  step=0.0125  cos=0.0000  warm+cos=0.0000
  `, "Schedules start with large steps for fast progress and shrink them to settle into a minimum. Warm-up keeps the first few steps small while optimiser statistics (like Adam's moments) are still unreliable — standard practice for Transformers."),

  "gradient-clipping": ex(r`
    import math

    def clip_by_norm(g, max_norm):
        norm = math.sqrt(sum(x * x for x in g))
        if norm <= max_norm:
            return g, norm
        scale = max_norm / norm          # shrink, but keep the direction
        return [x * scale for x in g], norm

    for g in [[0.3, 0.4], [30.0, -40.0]]:
        clipped, norm = clip_by_norm(g, 1.0)
        print(f"norm={norm:.1f} -> [{clipped[0]:.3f}, {clipped[1]:.3f}]")
  `, r`
    fn clip_by_norm(g: &[f64], max_norm: f64) -> (Vec<f64>, f64) {
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= max_norm {
            return (g.to_vec(), norm);
        }
        let scale = max_norm / norm; // shrink, but keep the direction
        (g.iter().map(|x| x * scale).collect(), norm)
    }

    fn main() {
        for g in [[0.3, 0.4], [30.0, -40.0]] {
            let (c, norm) = clip_by_norm(&g, 1.0);
            println!("norm={:.1} -> [{:.3}, {:.3}]", norm, c[0], c[1]);
        }
    }
  `, r`
    norm=0.5 -> [0.300, 0.400]
    norm=50.0 -> [0.600, -0.800]
  `, "A small gradient passes through unchanged; an exploding gradient of norm 50 is rescaled to norm 1 while keeping its direction (0.6, −0.8). This is what torch.nn.utils.clip_grad_norm_ does to stop RNN and Transformer training from blowing up."),

  "weight-decay": ex(r`
    # one weight, loss gradient pushes it toward 2.0
    def train(decay, steps=200, lr=0.1):
        w = 5.0
        for _ in range(steps):
            g = 2 * (w - 2.0)            # gradient of (w - 2)^2
            w -= lr * g                  # ordinary step
            w -= lr * decay * w          # decoupled weight decay (AdamW style)
        return w

    for decay in [0.0, 0.1, 1.0]:
        print(f"decay={decay:.1f}: final w={train(decay):.4f}")
  `, r`
    // one weight, loss gradient pushes it toward 2.0
    fn train(decay: f64, steps: usize, lr: f64) -> f64 {
        let mut w = 5.0_f64;
        for _ in 0..steps {
            let g = 2.0 * (w - 2.0); // gradient of (w - 2)^2
            w -= lr * g; // ordinary step
            w -= lr * decay * w; // decoupled weight decay (AdamW style)
        }
        w
    }

    fn main() {
        for decay in [0.0, 0.1, 1.0] {
            println!("decay={:.1}: final w={:.4}", decay, train(decay, 200, 0.1));
        }
    }
  `, r`
    decay=0.0: final w=2.0000
    decay=0.1: final w=1.9038
    decay=1.0: final w=1.2857
  `, "Weight decay shrinks the weight a little every step, so the final value balances the loss (which wants w = 2) against the pull toward 0. Stronger decay gives a smaller weight — the same effect as an L2 penalty."),

  "newtons-method": ex(r`
    import math
    # minimise f(x) = e^x - 2x  (minimum at x = ln 2)
    fp = lambda x: math.exp(x) - 2      # f'(x)
    fpp = lambda x: math.exp(x)         # f''(x)
    x = 2.0
    for k in range(1, 6):
        x = x - fp(x) / fpp(x)          # Newton step
        print(f"iter {k}: x={x:.10f}  error={abs(x - math.log(2)):.12f}")
  `, r`
    fn main() {
        // minimise f(x) = e^x - 2x  (minimum at x = ln 2)
        let fp = |x: f64| x.exp() - 2.0; // f'(x)
        let fpp = |x: f64| x.exp(); // f''(x)
        let mut x = 2.0_f64;
        for k in 1..=5 {
            x = x - fp(x) / fpp(x); // Newton step
            println!("iter {}: x={:.10}  error={:.12}", k, x, (x - 2.0_f64.ln()).abs());
        }
    }
  `, r`
    iter 1: x=1.2706705665  error=0.577523385913
    iter 2: x=0.8319573037  error=0.138810123180
    iter 3: x=0.7023505840  error=0.009203403457
    iter 4: x=0.6931894023  error=0.000042221691
    iter 5: x=0.6931471815  error=0.000000000891
  `, "Newton's method divides the slope by the curvature. Near the minimum the error is roughly squared every iteration (quadratic convergence): the number of correct digits about doubles each step."),

  "quasi-newton": ex(r`
    # BFGS on f(x, y) = (x - 1)^2 + 5 (y + 2)^2 + x y
    def f(p):  return (p[0] - 1) ** 2 + 5 * (p[1] + 2) ** 2 + p[0] * p[1]
    def g(p):  return [2 * (p[0] - 1) + p[1], 10 * (p[1] + 2) + p[0]]

    p = [0.0, 0.0]
    B = [[1.0, 0.0], [0.0, 1.0]]         # inverse-Hessian estimate
    gp = g(p)
    for k in range(1, 7):
        d = [-(B[0][0] * gp[0] + B[0][1] * gp[1]), -(B[1][0] * gp[0] + B[1][1] * gp[1])]
        a = 1.0                          # backtracking (Armijo) line search
        while f([p[0] + a * d[0], p[1] + a * d[1]]) > f(p) + 1e-4 * a * (gp[0] * d[0] + gp[1] * d[1]):
            a *= 0.5
        s = [a * d[0], a * d[1]]
        p = [p[0] + s[0], p[1] + s[1]]
        gn = g(p)
        y = [gn[0] - gp[0], gn[1] - gp[1]]
        sy = s[0] * y[0] + s[1] * y[1]
        if sy > 1e-12:                   # BFGS update of B
            rho = 1 / sy
            By = [B[0][0] * y[0] + B[0][1] * y[1], B[1][0] * y[0] + B[1][1] * y[1]]
            yBy = y[0] * By[0] + y[1] * By[1]
            for i in range(2):
                for j in range(2):
                    B[i][j] += (1 + rho * yBy) * rho * s[i] * s[j] - rho * (By[i] * s[j] + s[i] * By[j])
        gp = gn
        print(f"iter {k}: x={p[0]:.5f} y={p[1]:.5f} f={f(p):.6f}")
  `, r`
    // BFGS on f(x, y) = (x - 1)^2 + 5 (y + 2)^2 + x y
    fn f(p: [f64; 2]) -> f64 { (p[0] - 1.0).powi(2) + 5.0 * (p[1] + 2.0).powi(2) + p[0] * p[1] }
    fn g(p: [f64; 2]) -> [f64; 2] { [2.0 * (p[0] - 1.0) + p[1], 10.0 * (p[1] + 2.0) + p[0]] }

    fn main() {
        let mut p = [0.0, 0.0];
        let mut b = [[1.0, 0.0], [0.0, 1.0]]; // inverse-Hessian estimate
        let mut gp = g(p);
        for k in 1..=6 {
            let d = [-(b[0][0] * gp[0] + b[0][1] * gp[1]), -(b[1][0] * gp[0] + b[1][1] * gp[1])];
            let mut a = 1.0; // backtracking (Armijo) line search
            while f([p[0] + a * d[0], p[1] + a * d[1]]) > f(p) + 1e-4 * a * (gp[0] * d[0] + gp[1] * d[1]) {
                a *= 0.5;
            }
            let s = [a * d[0], a * d[1]];
            p = [p[0] + s[0], p[1] + s[1]];
            let gn = g(p);
            let y = [gn[0] - gp[0], gn[1] - gp[1]];
            let sy = s[0] * y[0] + s[1] * y[1];
            if sy > 1e-12 {
                // BFGS update of B
                let rho = 1.0 / sy;
                let by = [b[0][0] * y[0] + b[0][1] * y[1], b[1][0] * y[0] + b[1][1] * y[1]];
                let yby = y[0] * by[0] + y[1] * by[1];
                for i in 0..2 {
                    for j in 0..2 {
                        b[i][j] += (1.0 + rho * yby) * rho * s[i] * s[j] - rho * (by[i] * s[j] + s[i] * by[j]);
                    }
                }
            }
            gp = gn;
            println!("iter {}: x={:.5} y={:.5} f={:.6}", k, p[0], p[1], f(p));
        }
    }
  `, r`
    iter 1: x=0.25000 y=-2.50000 f=1.187500
    iter 2: x=3.91778 y=-2.35699 f=-0.083531
    iter 3: x=2.09585 y=-2.19672 f=-3.209614
    iter 4: x=2.10531 y=-2.21131 f=-3.210523
    iter 5: x=2.10527 y=-2.21053 f=-3.210526
    iter 6: x=2.10526 y=-2.21053 f=-3.210526
  `, "BFGS starts out like gradient descent (B = I) but learns the curvature from gradient differences. On this 2-D quadratic it reaches the exact minimum (x, y) = (40/19, −42/19) ≈ (2.10526, −2.21053) within a few iterations, without ever computing a second derivative."),

  "line-search": ex(r`
    # backtracking line search on the Rosenbrock function from (-1.2, 1)
    def f(x, y):  return (1 - x) ** 2 + 100 * (y - x * x) ** 2
    def g(x, y):  return -2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)

    x, y = -1.2, 1.0
    gx, gy = g(x, y)
    dx, dy = -gx, -gy                    # steepest-descent direction
    slope = gx * dx + gy * dy            # directional derivative (< 0)
    alpha, beta, c = 1.0, 0.5, 1e-4
    trials = 1
    while f(x + alpha * dx, y + alpha * dy) > f(x, y) + c * alpha * slope:
        alpha *= beta
        trials += 1
    print(f"f before = {f(x, y):.4f}")
    print(f"accepted alpha = {alpha:.6f} after {trials} trials")
    print(f"f after  = {f(x + alpha * dx, y + alpha * dy):.4f}")
  `, r`
    // backtracking line search on the Rosenbrock function from (-1.2, 1)
    fn f(x: f64, y: f64) -> f64 { (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2) }
    fn g(x: f64, y: f64) -> (f64, f64) { (-2.0 * (1.0 - x) - 400.0 * x * (y - x * x), 200.0 * (y - x * x)) }

    fn main() {
        let (x, y) = (-1.2_f64, 1.0_f64);
        let (gx, gy) = g(x, y);
        let (dx, dy) = (-gx, -gy); // steepest-descent direction
        let slope = gx * dx + gy * dy; // directional derivative (< 0)
        let (mut alpha, beta, c) = (1.0_f64, 0.5, 1e-4);
        let mut trials = 1;
        while f(x + alpha * dx, y + alpha * dy) > f(x, y) + c * alpha * slope {
            alpha *= beta;
            trials += 1;
        }
        println!("f before = {:.4}", f(x, y));
        println!("accepted alpha = {:.6} after {} trials", alpha, trials);
        println!("f after  = {:.4}", f(x + alpha * dx, y + alpha * dy));
    }
  `, r`
    f before = 24.2000
    accepted alpha = 0.000977 after 11 trials
    f after  = 5.1011
  `, "The full step α = 1 overshoots badly because the gradient here is large (‖∇f‖ ≈ 233), so backtracking halves α until the Armijo condition is met. The search costs only function evaluations and guarantees the step actually decreases f."),

  "coordinate-descent": ex(r`
    # minimise f(x, y) = x^2 + x y + y^2 - 3x  (minimum at x=2, y=-1)
    x, y = 0.0, 0.0
    for sweep in range(1, 7):
        x = (3 - y) / 2      # exact minimiser over x with y fixed: df/dx = 2x + y - 3 = 0
        y = -x / 2           # exact minimiser over y with x fixed: df/dy = x + 2y = 0
        f = x * x + x * y + y * y - 3 * x
        print(f"sweep {sweep}: x={x:.5f} y={y:.5f} f={f:.6f}")
  `, r`
    fn main() {
        // minimise f(x, y) = x^2 + x y + y^2 - 3x  (minimum at x=2, y=-1)
        let (mut x, mut y) = (0.0_f64, 0.0_f64);
        for sweep in 1..=6 {
            x = (3.0 - y) / 2.0; // exact minimiser over x with y fixed: df/dx = 2x + y - 3 = 0
            y = -x / 2.0; // exact minimiser over y with x fixed: df/dy = x + 2y = 0
            let f = x * x + x * y + y * y - 3.0 * x;
            println!("sweep {}: x={:.5} y={:.5} f={:.6}", sweep, x, y, f);
        }
    }
  `, r`
    sweep 1: x=1.50000 y=-0.75000 f=-2.812500
    sweep 2: x=1.87500 y=-0.93750 f=-2.988281
    sweep 3: x=1.96875 y=-0.98438 f=-2.999268
    sweep 4: x=1.99219 y=-0.99609 f=-2.999954
    sweep 5: x=1.99805 y=-0.99902 f=-2.999997
    sweep 6: x=1.99951 y=-0.99976 f=-3.000000
  `, "Each sweep solves two one-dimensional problems exactly, holding the other coordinate fixed. The error shrinks by a factor of 4 per sweep here; the coupling term xy is what keeps it from finishing in a single sweep."),

  "lagrange-multipliers": ex(r`
    # minimise f = x^2 + 2 y^2  subject to  x + y = 3
    # stationarity: 2x = lam, 4y = lam, plus the constraint x + y = 3
    # -> solve the 3x3 linear system  [2 0 -1; 0 4 -1; 1 1 0] [x y lam] = [0 0 3]
    A = [[2.0, 0.0, -1.0], [0.0, 4.0, -1.0], [1.0, 1.0, 0.0]]
    b = [0.0, 0.0, 3.0]
    n = 3
    for i in range(n):                         # Gaussian elimination with pivoting
        p = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for r in range(i + 1, n):
            m = A[r][i] / A[i][i]
            A[r] = [A[r][k] - m * A[i][k] for k in range(n)]
            b[r] -= m * b[i]
    sol = [0.0] * n
    for i in reversed(range(n)):
        sol[i] = (b[i] - sum(A[i][k] * sol[k] for k in range(i + 1, n))) / A[i][i]
    x, y, lam = sol
    print(f"x={x:.4f} y={y:.4f} lambda={lam:.4f}")
    print(f"f*={x * x + 2 * y * y:.4f}")
  `, r`
    fn main() {
        // minimise f = x^2 + 2 y^2  subject to  x + y = 3
        // stationarity: 2x = lam, 4y = lam, plus the constraint x + y = 3
        let mut a = [[2.0, 0.0, -1.0], [0.0, 4.0, -1.0], [1.0, 1.0, 0.0_f64]];
        let mut b = [0.0, 0.0, 3.0_f64];
        let n = 3;
        for i in 0..n {
            // Gaussian elimination with pivoting
            let p = (i..n).max_by(|&r, &s| a[r][i].abs().partial_cmp(&a[s][i].abs()).unwrap()).unwrap();
            a.swap(i, p);
            b.swap(i, p);
            for r in i + 1..n {
                let m = a[r][i] / a[i][i];
                for k in 0..n { a[r][k] -= m * a[i][k]; }
                b[r] -= m * b[i];
            }
        }
        let mut sol = [0.0; 3];
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| a[i][k] * sol[k]).sum();
            sol[i] = (b[i] - s) / a[i][i];
        }
        let (x, y, lam) = (sol[0], sol[1], sol[2]);
        println!("x={:.4} y={:.4} lambda={:.4}", x, y, lam);
        println!("f*={:.4}", x * x + 2.0 * y * y);
    }
  `, r`
    x=2.0000 y=1.0000 lambda=4.0000
    f*=6.0000
  `, "Setting the gradient of the Lagrangian to zero turns a constrained problem into a linear system. The answer x = 2, y = 1 sits on the line x + y = 3, and λ = 4 says relaxing the constraint to x + y = 3 + ε raises the optimum by about 4ε."),
};
