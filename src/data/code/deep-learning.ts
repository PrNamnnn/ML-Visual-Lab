import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";
import { dedent } from "@/lib/utilities/highlight";

const join = (...parts: string[]) => parts.map(dedent).join("\n\n");
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

/** Activation snippet: evaluates f and f' at five points in both languages. */
/** Verified outputs of the activation snippets (filled from a run where Python and Rust agree). */
const ACT_OUT: Record<string, string> = {
  "linear-activation": "x=-2.0  f(x)=-2.0000  f'(x)=+1.0000\nx=-0.5  f(x)=-0.5000  f'(x)=+1.0000\nx=+0.0  f(x)=+0.0000  f'(x)=+1.0000\nx=+0.5  f(x)=+0.5000  f'(x)=+1.0000\nx=+2.0  f(x)=+2.0000  f'(x)=+1.0000",
  "sigmoid-activation": "x=-2.0  f(x)=+0.1192  f'(x)=+0.1050\nx=-0.5  f(x)=+0.3775  f'(x)=+0.2350\nx=+0.0  f(x)=+0.5000  f'(x)=+0.2500\nx=+0.5  f(x)=+0.6225  f'(x)=+0.2350\nx=+2.0  f(x)=+0.8808  f'(x)=+0.1050",
  "tanh-activation": "x=-2.0  f(x)=-0.9640  f'(x)=+0.0707\nx=-0.5  f(x)=-0.4621  f'(x)=+0.7864\nx=+0.0  f(x)=+0.0000  f'(x)=+1.0000\nx=+0.5  f(x)=+0.4621  f'(x)=+0.7864\nx=+2.0  f(x)=+0.9640  f'(x)=+0.0707",
  "relu-activation": "x=-2.0  f(x)=+0.0000  f'(x)=+0.0000\nx=-0.5  f(x)=+0.0000  f'(x)=+0.0000\nx=+0.0  f(x)=+0.0000  f'(x)=+0.0000\nx=+0.5  f(x)=+0.5000  f'(x)=+1.0000\nx=+2.0  f(x)=+2.0000  f'(x)=+1.0000",
  "leaky-relu-activation": "x=-2.0  f(x)=-0.0200  f'(x)=+0.0100\nx=-0.5  f(x)=-0.0050  f'(x)=+0.0100\nx=+0.0  f(x)=+0.0000  f'(x)=+0.0100\nx=+0.5  f(x)=+0.5000  f'(x)=+1.0000\nx=+2.0  f(x)=+2.0000  f'(x)=+1.0000",
  "prelu-activation": "x=-2.0  f(x)=-0.5000  f'(x)=+0.2500\nx=-0.5  f(x)=-0.1250  f'(x)=+0.2500\nx=+0.0  f(x)=+0.0000  f'(x)=+0.2500\nx=+0.5  f(x)=+0.5000  f'(x)=+1.0000\nx=+2.0  f(x)=+2.0000  f'(x)=+1.0000",
  "elu-activation": "x=-2.0  f(x)=-0.8647  f'(x)=+0.1353\nx=-0.5  f(x)=-0.3935  f'(x)=+0.6065\nx=+0.0  f(x)=+0.0000  f'(x)=+1.0000\nx=+0.5  f(x)=+0.5000  f'(x)=+1.0000\nx=+2.0  f(x)=+2.0000  f'(x)=+1.0000",
  "selu-activation": "x=-2.0  f(x)=-1.5202  f'(x)=+0.2379\nx=-0.5  f(x)=-0.6918  f'(x)=+1.0663\nx=+0.0  f(x)=+0.0000  f'(x)=+1.7581\nx=+0.5  f(x)=+0.5254  f'(x)=+1.0507\nx=+2.0  f(x)=+2.1014  f'(x)=+1.0507",
  "gelu-activation": "x=-2.0  f(x)=-0.0454  f'(x)=-0.0861\nx=-0.5  f(x)=-0.1543  f'(x)=+0.1326\nx=+0.0  f(x)=+0.0000  f'(x)=+0.5000\nx=+0.5  f(x)=+0.3457  f'(x)=+0.8674\nx=+2.0  f(x)=+1.9546  f'(x)=+1.0861",
  "swish-activation": "x=-2.0  f(x)=-0.2384  f'(x)=-0.0908\nx=-0.5  f(x)=-0.1888  f'(x)=+0.2600\nx=+0.0  f(x)=+0.0000  f'(x)=+0.5000\nx=+0.5  f(x)=+0.3112  f'(x)=+0.7400\nx=+2.0  f(x)=+1.7616  f'(x)=+1.0908",
};
const act = (key: string, pyF: string, pyD: string, rsF: string, rsD: string, explanation: string, pyPre = "", rsPre = "") =>
  ex(
    `import math\n${pyPre}def f(x):\n    return ${pyF}\n\ndef df(x):\n    return ${pyD}\n\nfor x in [-2.0, -0.5, 0.0, 0.5, 2.0]:\n    print(f"x={x:+.1f}  f(x)={f(x) + 0.0:+.4f}  f'(x)={df(x) + 0.0:+.4f}")`,
    `${rsPre}fn f(x: f64) -> f64 {\n    ${rsF}\n}\n\nfn df(x: f64) -> f64 {\n    ${rsD}\n}\n\nfn main() {\n    for x in [-2.0, -0.5, 0.0, 0.5, 2.0_f64] {\n        println!("x={:+.1}  f(x)={:+.4}  f'(x)={:+.4}", x, f(x) + 0.0, df(x) + 0.0);\n    }\n}`,
    ACT_OUT[key] ?? "?",
    explanation,
  );

export const deepLearningCode: Record<string, CodeExample> = {
  "artificial-neuron": ex(r`
    import math
    x = [0.5, -1.2, 3.0]                 # inputs
    w = [0.8, 0.1, -0.4]                 # weights
    b = 0.2                              # bias
    z = sum(xi * wi for xi, wi in zip(x, w)) + b
    print(f"z = w.x + b = {z:.3f}")
    print(f"sigmoid(z) = {1 / (1 + math.exp(-z)):.4f}")
    print(f"relu(z)    = {max(0.0, z):.4f}")
  `, r`
    fn main() {
        let x = [0.5, -1.2, 3.0_f64]; // inputs
        let w = [0.8, 0.1, -0.4_f64]; // weights
        let b = 0.2; // bias
        let z: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
        println!("z = w.x + b = {:.3}", z);
        println!("sigmoid(z) = {:.4}", 1.0 / (1.0 + (-z).exp()));
        println!("relu(z)    = {:.4}", z.max(0.0));
    }
  `, r`
    z = w.x + b = -0.720
    sigmoid(z) = 0.3274
    relu(z)    = 0.0000
  `, "A neuron computes a weighted sum of its inputs plus a bias, then applies an activation. The negative pre-activation gives a sigmoid output below 0.5 and a ReLU output of exactly 0."),

  "multilayer-perceptron": ex(r`
    step = lambda z: 1.0 if z > 0 else 0.0
    def mlp(x1, x2):
        h1 = step(x1 + x2 - 0.5)         # hidden unit 1: OR
        h2 = step(x1 + x2 - 1.5)         # hidden unit 2: AND
        return step(h1 - h2 - 0.5)       # output: OR and not AND = XOR
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(f"XOR({x1}, {x2}) = {mlp(x1, x2):.0f}")
  `, r`
    fn step(z: f64) -> f64 { if z > 0.0 { 1.0 } else { 0.0 } }

    fn mlp(x1: f64, x2: f64) -> f64 {
        let h1 = step(x1 + x2 - 0.5); // hidden unit 1: OR
        let h2 = step(x1 + x2 - 1.5); // hidden unit 2: AND
        step(h1 - h2 - 0.5) // output: OR and not AND = XOR
    }

    fn main() {
        for (x1, x2) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            println!("XOR({}, {}) = {:.0}", x1, x2, mlp(x1 as f64, x2 as f64));
        }
    }
  `, r`
    XOR(0, 0) = 0
    XOR(0, 1) = 1
    XOR(1, 0) = 1
    XOR(1, 1) = 0
  `, "A single neuron cannot compute XOR because the classes are not linearly separable. One hidden layer fixes it: one unit computes OR, another AND, and the output combines them."),

  backpropagation: ex(r`
    import math
    x, y = 1.5, 1.0
    w1, w2 = 0.4, -0.7
    def loss(w1, w2):
        h = math.tanh(w1 * x)
        p = 1 / (1 + math.exp(-w2 * h))
        return 0.5 * (p - y) ** 2
    # backward pass by the chain rule
    h = math.tanh(w1 * x); p = 1 / (1 + math.exp(-w2 * h))
    dp = p - y
    dz2 = dp * p * (1 - p)
    g2 = dz2 * h
    g1 = dz2 * w2 * (1 - h * h) * x
    eps = 1e-6                            # numerical check with central differences
    n1 = (loss(w1 + eps, w2) - loss(w1 - eps, w2)) / (2 * eps)
    n2 = (loss(w1, w2 + eps) - loss(w1, w2 - eps)) / (2 * eps)
    print(f"dL/dw1: backprop={g1:.6f} numeric={n1:.6f}")
    print(f"dL/dw2: backprop={g2:.6f} numeric={n2:.6f}")
  `, r`
    fn main() {
        let (x, y) = (1.5_f64, 1.0_f64);
        let (w1, w2) = (0.4_f64, -0.7_f64);
        let loss = |w1: f64, w2: f64| {
            let h = (w1 * x).tanh();
            let p = 1.0 / (1.0 + (-w2 * h).exp());
            0.5 * (p - y).powi(2)
        };
        // backward pass by the chain rule
        let h = (w1 * x).tanh();
        let p = 1.0 / (1.0 + (-w2 * h).exp());
        let dz2 = (p - y) * p * (1.0 - p);
        let g2 = dz2 * h;
        let g1 = dz2 * w2 * (1.0 - h * h) * x;
        let eps = 1e-6; // numerical check with central differences
        let n1 = (loss(w1 + eps, w2) - loss(w1 - eps, w2)) / (2.0 * eps);
        let n2 = (loss(w1, w2 + eps) - loss(w1, w2 - eps)) / (2.0 * eps);
        println!("dL/dw1: backprop={:.6} numeric={:.6}", g1, n1);
        println!("dL/dw2: backprop={:.6} numeric={:.6}", g2, n2);
    }
  `, r`
    dL/dw1: backprop=0.106923 numeric=0.106923
    dL/dw2: backprop=-0.076856 numeric=-0.076856
  `, "Backpropagation multiplies local derivatives from the output back to each weight. A finite-difference check — the standard way to test a gradient implementation — agrees to six decimals."),

  "activation-functions": ex(r`
    import math
    acts = {
        "sigmoid": lambda z: 1 / (1 + math.exp(-z)),
        "tanh": math.tanh,
        "relu": lambda z: max(0.0, z),
        "leaky relu": lambda z: z if z > 0 else 0.01 * z,
    }
    for name, f in acts.items():
        print(f"{name:<10}", " ".join(f"{f(z) + 0.0:+.3f}" for z in [-2.0, 0.0, 2.0]))
  `, r`
    fn main() {
        let acts: [(&str, fn(f64) -> f64); 4] = [
            ("sigmoid", |z| 1.0 / (1.0 + (-z).exp())),
            ("tanh", |z| z.tanh()),
            ("relu", |z| z.max(0.0)),
            ("leaky relu", |z| if z > 0.0 { z } else { 0.01 * z }),
        ];
        for (name, f) in acts.iter() {
            let vals: Vec<String> = [-2.0, 0.0, 2.0].iter().map(|&z| format!("{:+.3}", f(z) + 0.0)).collect();
            println!("{:<10} {}", name, vals.join(" "));
        }
    }
  `, r`
    sigmoid    +0.119 +0.500 +0.881
    tanh       -0.964 +0.000 +0.964
    relu       +0.000 +0.000 +2.000
    leaky relu -0.020 +0.000 +2.000
  `, "Activations add non-linearity. Sigmoid squashes into (0, 1), tanh into (−1, 1) and is zero-centred, ReLU passes positives unchanged and zeroes negatives, and leaky ReLU keeps a small slope for negatives."),

  "loss-functions": ex(r`
    import math
    y_true, y_pred = 1.0, 0.8
    print(f"squared error   = {(y_true - y_pred) ** 2:.4f}")
    print(f"absolute error  = {abs(y_true - y_pred):.4f}")
    print(f"cross-entropy   = {-math.log(y_pred):.4f}  (prob. 0.8 for the true class)")
    print(f"cross-entropy   = {-math.log(0.1):.4f}  (prob. 0.1 for the true class)")
  `, r`
    fn main() {
        let (y_true, y_pred) = (1.0_f64, 0.8_f64);
        println!("squared error   = {:.4}", (y_true - y_pred).powi(2));
        println!("absolute error  = {:.4}", (y_true - y_pred).abs());
        println!("cross-entropy   = {:.4}  (prob. 0.8 for the true class)", -y_pred.ln());
        println!("cross-entropy   = {:.4}  (prob. 0.1 for the true class)", -0.1_f64.ln());
    }
  `, r`
    squared error   = 0.0400
    absolute error  = 0.2000
    cross-entropy   = 0.2231  (prob. 0.8 for the true class)
    cross-entropy   = 2.3026  (prob. 0.1 for the true class)
  `, "The loss turns 'how wrong' into one number to minimise. Cross-entropy punishes confident mistakes heavily: assigning only 0.1 to the true class costs over ten times more than assigning 0.8."),

  "linear-activation": act("linear-activation", "x", "1.0", "x", "1.0", "The identity activation leaves values unchanged and has slope 1 everywhere. Stacking linear layers with it collapses to a single linear map, so it is only used for regression outputs."),
  "sigmoid-activation": act("sigmoid-activation", "1 / (1 + math.exp(-x))", "f(x) * (1 - f(x))", "1.0 / (1.0 + (-x).exp())", "f(x) * (1.0 - f(x))", "The sigmoid maps any input into (0, 1). Its derivative σ(1 − σ) peaks at 0.25 at x = 0 and shrinks quickly for large |x| — the root of vanishing gradients in deep sigmoid networks."),
  "tanh-activation": act("tanh-activation", "math.tanh(x)", "1 - math.tanh(x) ** 2", "x.tanh()", "1.0 - x.tanh().powi(2)", "tanh is a zero-centred, rescaled sigmoid with range (−1, 1). Its maximum slope is 1 at x = 0, four times the sigmoid's, but it also saturates for large |x|."),
  "relu-activation": act("relu-activation", "max(0.0, x)", "1.0 if x > 0 else 0.0", "x.max(0.0)", "if x > 0.0 { 1.0 } else { 0.0 }", "ReLU passes positive values unchanged with slope 1 and outputs exactly 0 for negatives. The constant slope avoids saturation; the zero region can leave units permanently inactive ('dying ReLU')."),
  "leaky-relu-activation": act("leaky-relu-activation", "x if x > 0 else 0.01 * x", "1.0 if x > 0 else 0.01", "if x > 0.0 { x } else { 0.01 * x }", "if x > 0.0 { 1.0 } else { 0.01 }", "Leaky ReLU keeps a small slope (0.01) for negative inputs, so gradients never become exactly zero and units cannot die."),
  "prelu-activation": act("prelu-activation", "x if x > 0 else a * x", "1.0 if x > 0 else a", "if x > 0.0 { x } else { A * x }", "if x > 0.0 { 1.0 } else { A }", "PReLU is a leaky ReLU whose negative slope a is learned during training (here a = 0.25). The derivative with respect to a, used to update it, is x for negative inputs.", "a = 0.25                        # learned negative slope\n\n", "const A: f64 = 0.25; // learned negative slope\n\n"),
  "elu-activation": act("elu-activation", "x if x > 0 else math.exp(x) - 1", "1.0 if x > 0 else math.exp(x)", "if x > 0.0 { x } else { x.exp() - 1.0 }", "if x > 0.0 { 1.0 } else { x.exp() }", "ELU is linear for positive inputs and smoothly approaches −1 for negative ones, pushing mean activations towards zero; its slope is continuous at 0."),
  "selu-activation": act("selu-activation", "LAM * (x if x > 0 else ALPHA * (math.exp(x) - 1))", "LAM * (1.0 if x > 0 else ALPHA * math.exp(x))", "LAM * if x > 0.0 { x } else { ALPHA * (x.exp() - 1.0) }", "LAM * if x > 0.0 { 1.0 } else { ALPHA * x.exp() }", "SELU is a scaled ELU with constants λ ≈ 1.0507 and α ≈ 1.6733 chosen so activations self-normalise towards mean 0 and variance 1 through many layers (with matching initialisation).", "LAM, ALPHA = 1.0507009873554805, 1.6732632423543772\n\n", "const LAM: f64 = 1.0507009873554805;\nconst ALPHA: f64 = 1.6732632423543772;\n\n"),
  "gelu-activation": act("gelu-activation", "0.5 * x * (1 + math.tanh(C * (x + 0.044715 * x ** 3)))", "(f(x + 1e-6) - f(x - 1e-6)) / 2e-6", "0.5 * x * (1.0 + (C * (x + 0.044715 * x.powi(3))).tanh())", "(f(x + 1e-6) - f(x - 1e-6)) / 2e-6", "GELU weights each input by the probability that a standard normal is below it, x·Φ(x), shown here with the tanh approximation used in GPT-2/BERT (Rust has no erf). It is smooth and slightly negative for small negative x; the derivative is computed numerically.", "C = math.sqrt(2 / math.pi)\n\n", "const C: f64 = 0.7978845608028654; // sqrt(2 / pi)\n\n"),
  "swish-activation": act("swish-activation", "x / (1 + math.exp(-x))", "f(x) + (1 - f(x)) / (1 + math.exp(-x))", "x / (1.0 + (-x).exp())", "f(x) + (1.0 - f(x)) / (1.0 + (-x).exp())", "Swish (SiLU) is x·σ(x): smooth, non-monotonic with a small dip below zero, and close to ReLU for large |x|. Its derivative uses f + σ(1 − f)."),

  "softmax-activation": ex(r`
    import math
    logits = [2.0, 1.0, 0.1]
    m = max(logits)                       # subtract the max for numerical stability
    e = [math.exp(z - m) for z in logits]
    p = [v / sum(e) for v in e]
    print("probabilities:", ", ".join(f"{v:.4f}" for v in p), f"(sum = {sum(p):.4f})")
    big = [1000.0, 999.0]
    e = [math.exp(z - max(big)) for z in big]
    print("stable softmax of [1000, 999]:", ", ".join(f"{v / sum(e):.4f}" for v in e))
  `, r`
    fn softmax(z: &[f64]) -> Vec<f64> {
        let m = z.iter().cloned().fold(f64::MIN, f64::max); // subtract the max for numerical stability
        let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    fn main() {
        let p = softmax(&[2.0, 1.0, 0.1]);
        let shown: Vec<String> = p.iter().map(|v| format!("{:.4}", v)).collect();
        println!("probabilities: {} (sum = {:.4})", shown.join(", "), p.iter().sum::<f64>());
        let q = softmax(&[1000.0, 999.0]);
        println!("stable softmax of [1000, 999]: {}", q.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    probabilities: 0.6590, 0.2424, 0.0986 (sum = 1.0000)
    stable softmax of [1000, 999]: 0.7311, 0.2689
  `, "Softmax turns logits into probabilities that sum to 1, preserving their order. Subtracting the maximum first leaves the result unchanged mathematically but prevents exp(1000) from overflowing."),

  "mse-loss": ex(r`
    y = [3.0, 5.0, 2.5, 7.0]
    p = [2.5, 5.0, 4.0, 8.0]
    mse = sum((a - b) ** 2 for a, b in zip(y, p)) / len(y)
    grad = [2 * (b - a) / len(y) for a, b in zip(y, p)]
    print(f"MSE = {mse:.4f}")
    print("dMSE/dpred:", ", ".join(f"{g + 0.0:+.3f}" for g in grad))
  `, r`
    fn main() {
        let y = [3.0, 5.0, 2.5, 7.0_f64];
        let p = [2.5, 5.0, 4.0, 8.0_f64];
        let n = y.len() as f64;
        let mse = y.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n;
        let grad: Vec<String> = y.iter().zip(p.iter()).map(|(a, b)| format!("{:+.3}", 2.0 * (b - a) / n + 0.0)).collect();
        println!("MSE = {:.4}", mse);
        println!("dMSE/dpred: {}", grad.join(", "));
    }
  `, r`
    MSE = 0.8750
    dMSE/dpred: -0.250, +0.000, +0.750, +0.500
  `, "Squared error grows with the square of the miss, so its gradient is proportional to the error: the 1.5 miss pushes three times harder than the 0.5 miss."),

  "mae-loss": ex(r`
    y = [3.0, 5.0, 2.5, 7.0]
    p = [2.5, 5.0, 4.0, 8.0]
    mae = sum(abs(a - b) for a, b in zip(y, p)) / len(y)
    sign = lambda v: (v > 0) - (v < 0)
    print(f"MAE = {mae:.4f}")
    print("dMAE/dpred:", ", ".join(f"{sign(b - a) / len(y):+.3f}" for a, b in zip(y, p)))
  `, r`
    fn main() {
        let y = [3.0, 5.0, 2.5, 7.0_f64];
        let p = [2.5, 5.0, 4.0, 8.0_f64];
        let n = y.len() as f64;
        let mae = y.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / n;
        let sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
        let g: Vec<String> = y.iter().zip(p.iter()).map(|(a, b)| format!("{:+.3}", sign(b - a) / n)).collect();
        println!("MAE = {:.4}", mae);
        println!("dMAE/dpred: {}", g.join(", "));
    }
  `, r`
    MAE = 0.7500
    dMAE/dpred: -0.250, +0.000, +0.250, +0.250
  `, "Absolute error grows linearly, so every non-zero error gets the same gradient magnitude regardless of size — robust to outliers, but the gradient never shrinks near the optimum."),

  "huber-loss": ex(r`
    delta = 1.0
    def huber(e):
        return 0.5 * e * e if abs(e) <= delta else delta * (abs(e) - 0.5 * delta)
    for e in [0.2, 0.9, 1.5, 5.0]:
        print(f"error={e:.1f}: squared/2={0.5 * e * e:.3f} huber={huber(e):.3f}")
  `, r`
    fn main() {
        let delta = 1.0_f64;
        let huber = |e: f64| if e.abs() <= delta { 0.5 * e * e } else { delta * (e.abs() - 0.5 * delta) };
        for e in [0.2, 0.9, 1.5, 5.0_f64] {
            println!("error={:.1}: squared/2={:.3} huber={:.3}", e, 0.5 * e * e, huber(e));
        }
    }
  `, r`
    error=0.2: squared/2=0.020 huber=0.020
    error=0.9: squared/2=0.405 huber=0.405
    error=1.5: squared/2=1.125 huber=1.000
    error=5.0: squared/2=12.500 huber=4.500
  `, "Huber loss equals half the squared error for small errors and grows only linearly beyond δ, so a 5-unit outlier costs 4.5 instead of 12.5."),

  "binary-cross-entropy": ex(r`
    import math
    y = [1, 0, 1, 0]
    p = [0.9, 0.2, 0.6, 0.95]
    eps = 1e-12
    terms = [-(t * math.log(q + eps) + (1 - t) * math.log(1 - q + eps)) for t, q in zip(y, p)]
    print("per-example:", ", ".join(f"{v:.4f}" for v in terms))
    print(f"mean BCE = {sum(terms) / len(terms):.4f}")
  `, r`
    fn main() {
        let y = [1.0, 0.0, 1.0, 0.0_f64];
        let p = [0.9, 0.2, 0.6, 0.95_f64];
        let eps = 1e-12;
        let terms: Vec<f64> = y.iter().zip(p.iter()).map(|(t, q)| -(t * (q + eps).ln() + (1.0 - t) * (1.0 - q + eps).ln())).collect();
        println!("per-example: {}", terms.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));
        println!("mean BCE = {:.4}", terms.iter().sum::<f64>() / terms.len() as f64);
    }
  `, r`
    per-example: 0.1054, 0.2231, 0.5108, 2.9957
    mean BCE = 0.9588
  `, "Binary cross-entropy is the negative log-probability of the true label. The last example predicts 0.95 for a negative and costs about 3.0 — more than the other three combined."),

  "categorical-cross-entropy": ex(r`
    import math
    logits = [2.0, 0.5, -1.0]
    true_class = 0
    m = max(logits)
    lse = m + math.log(sum(math.exp(z - m) for z in logits))   # log-sum-exp, numerically stable
    loss = lse - logits[true_class]
    p = [math.exp(z - lse) for z in logits]
    print("softmax:", ", ".join(f"{v:.4f}" for v in p))
    print(f"CE loss = {loss:.4f} = -log p[true] = {-math.log(p[true_class]):.4f}")
    print("gradient wrt logits:", ", ".join(f"{q - (k == true_class):+.4f}" for k, q in enumerate(p)))
  `, r`
    fn main() {
        let logits = [2.0, 0.5, -1.0_f64];
        let true_class = 0;
        let m = logits.iter().cloned().fold(f64::MIN, f64::max);
        let lse = m + logits.iter().map(|z| (z - m).exp()).sum::<f64>().ln(); // log-sum-exp, numerically stable
        let loss = lse - logits[true_class];
        let p: Vec<f64> = logits.iter().map(|z| (z - lse).exp()).collect();
        println!("softmax: {}", p.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));
        println!("CE loss = {:.4} = -log p[true] = {:.4}", loss, -p[true_class].ln());
        let g: Vec<String> = p.iter().enumerate().map(|(k, q)| format!("{:+.4}", q - if k == true_class { 1.0 } else { 0.0 })).collect();
        println!("gradient wrt logits: {}", g.join(", "));
    }
  `, r`
    softmax: 0.7856, 0.1753, 0.0391
    CE loss = 0.2413 = -log p[true] = 0.2413
    gradient wrt logits: -0.2144, +0.1753, +0.0391
  `, "Softmax followed by cross-entropy has the famously simple gradient p − y: push the true class's logit up by how far its probability falls short of 1, and the others down by their probability."),

  "hinge-loss": ex(r`
    for y, score in [(1, 2.0), (1, 0.5), (1, -1.0), (-1, -3.0), (-1, 0.2)]:
        m = y * score
        print(f"y={y:+d} score={score:+.1f}: margin={m:+.1f} hinge={max(0.0, 1 - m):.1f}")
  `, r`
    fn main() {
        for (y, score) in [(1.0, 2.0), (1.0, 0.5), (1.0, -1.0), (-1.0, -3.0), (-1.0, 0.2_f64)] {
            let m = y * score;
            println!("y={:+} score={:+.1}: margin={:+.1} hinge={:.1}", y as i32, score, m, (1.0 - m).max(0.0));
        }
    }
  `, r`
    y=+1 score=+2.0: margin=+2.0 hinge=0.0
    y=+1 score=+0.5: margin=+0.5 hinge=0.5
    y=+1 score=-1.0: margin=-1.0 hinge=2.0
    y=-1 score=-3.0: margin=+3.0 hinge=0.0
    y=-1 score=+0.2: margin=-0.2 hinge=1.2
  `, "Hinge loss is zero once an example is on the correct side with margin at least 1; inside the margin or on the wrong side it grows linearly. This is the loss behind support vector machines."),

  "focal-loss": ex(r`
    import math
    def focal(p_true, gamma):
        return -((1 - p_true) ** gamma) * math.log(p_true)
    for p in [0.9, 0.6, 0.1]:
        print(f"p_true={p:.1f}: CE={focal(p, 0):.4f} focal(g=2)={focal(p, 2):.4f}")
  `, r`
    fn main() {
        let focal = |p_true: f64, gamma: f64| -(1.0 - p_true).powf(gamma) * p_true.ln();
        for p in [0.9, 0.6, 0.1_f64] {
            println!("p_true={:.1}: CE={:.4} focal(g=2)={:.4}", p, focal(p, 0.0), focal(p, 2.0));
        }
    }
  `, r`
    p_true=0.9: CE=0.1054 focal(g=2)=0.0011
    p_true=0.6: CE=0.5108 focal(g=2)=0.0817
    p_true=0.1: CE=2.3026 focal(g=2)=1.8651
  `, "The factor (1 − p)^γ down-weights examples the model already gets right: an easy example at p = 0.9 contributes 100× less than under cross-entropy, while hard examples keep most of their loss. Used for heavily imbalanced detection."),

  "kl-divergence": ex(r`
    import math
    P = [0.5, 0.3, 0.2]
    Q = [0.4, 0.4, 0.2]
    kl = lambda a, b: sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)
    print(f"KL(P||Q) = {kl(P, Q):.5f}")
    print(f"KL(Q||P) = {kl(Q, P):.5f}  (not symmetric)")
    print(f"KL(P||P) = {kl(P, P):.5f}")
  `, r`
    fn kl(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).filter(|(x, _)| **x > 0.0).map(|(x, y)| x * (x / y).ln()).sum()
    }

    fn main() {
        let p = [0.5, 0.3, 0.2_f64];
        let q = [0.4, 0.4, 0.2_f64];
        println!("KL(P||Q) = {:.5}", kl(&p, &q));
        println!("KL(Q||P) = {:.5}  (not symmetric)", kl(&q, &p));
        println!("KL(P||P) = {:.5}", kl(&p, &p));
    }
  `, r`
    KL(P||Q) = 0.02527
    KL(Q||P) = 0.02582  (not symmetric)
    KL(P||P) = 0.00000
  `, "KL divergence measures how much extra information is needed when using Q to describe data from P. It is zero only for identical distributions and is not symmetric."),

  "contrastive-loss": ex(r`
    import math
    margin = 1.0
    pairs = [((0.0, 0.0), (0.3, 0.4), 1), ((0.0, 0.0), (0.6, 0.8), 0), ((0.0, 0.0), (1.2, 1.6), 0)]
    for a, b, same in pairs:
        d = math.dist(a, b)
        loss = d * d if same else max(0.0, margin - d) ** 2
        print(f"{'similar' if same else 'different':<9} d={d:.2f} loss={loss:.4f}")
  `, r`
    fn main() {
        let margin = 1.0_f64;
        let pairs: [((f64, f64), (f64, f64), i32); 3] = [((0.0, 0.0), (0.3, 0.4), 1), ((0.0, 0.0), (0.6, 0.8), 0), ((0.0, 0.0), (1.2, 1.6), 0)];
        for (a, b, same) in pairs.iter() {
            let d = (a.0 - b.0).hypot(a.1 - b.1);
            let loss = if *same == 1 { d * d } else { (margin - d).max(0.0).powi(2) };
            println!("{:<9} d={:.2} loss={:.4}", if *same == 1 { "similar" } else { "different" }, d, loss);
        }
    }
  `, r`
    similar   d=0.50 loss=0.2500
    different d=1.00 loss=0.0000
    different d=2.00 loss=0.0000
  `, "Similar pairs are pulled together (loss d²); different pairs are pushed apart only while they are closer than the margin. The far-apart different pair already costs nothing."),

  "triplet-loss": ex(r`
    import math
    anchor, positive = (0.0, 0.0), (0.5, 0.0)
    margin = 0.5
    for negative in [(0.6, 0.0), (1.2, 0.0), (2.0, 0.0)]:
        dp, dn = math.dist(anchor, positive), math.dist(anchor, negative)
        print(f"d(a,p)={dp:.2f} d(a,n)={dn:.2f} loss={max(0.0, dp - dn + margin):.2f}")
  `, r`
    fn main() {
        let (anchor, positive) = ((0.0_f64, 0.0_f64), (0.5_f64, 0.0_f64));
        let margin = 0.5;
        let dist = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);
        for negative in [(0.6, 0.0), (1.2, 0.0), (2.0, 0.0_f64)] {
            let (dp, dn) = (dist(anchor, positive), dist(anchor, negative));
            println!("d(a,p)={:.2} d(a,n)={:.2} loss={:.2}", dp, dn, (dp - dn + margin).max(0.0));
        }
    }
  `, r`
    d(a,p)=0.50 d(a,n)=0.60 loss=0.40
    d(a,p)=0.50 d(a,n)=1.20 loss=0.00
    d(a,p)=0.50 d(a,n)=2.00 loss=0.00
  `, "Triplet loss requires the negative to be at least a margin farther from the anchor than the positive. Only the 'hard' negative close to the anchor produces a loss; once d(a, n) ≥ d(a, p) + margin the triplet is satisfied."),

  "parameter-initialization": ex(join(PY_LCG, r`
    import math
    def randn():                           # Box-Muller from the LCG
        u1, u2 = max(rand(), 1e-12), rand()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    n, layers = 64, 8
    for name, std in [("N(0,1)", 1.0), ("N(0,0.01)", 0.01), ("Xavier 1/sqrt(n)", 1 / math.sqrt(n))]:
        h = [randn() for _ in range(n)]
        for _ in range(layers):            # tanh layers of width 64
            W = [[std * randn() for _ in range(n)] for _ in range(n)]
            h = [math.tanh(sum(W[i][j] * h[j] for j in range(n))) for i in range(n)]
        m = sum(h) / n
        print(f"{name:<17} std of layer-{layers} activations = {math.sqrt(sum((v - m) ** 2 for v in h) / n):.4f}")
  `), join(RS_LCG, r`
    fn randn() -> f64 {
        // Box-Muller from the LCG
        let (u1, u2) = (rand().max(1e-12), rand());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn main() {
        let (n, layers) = (64usize, 8);
        for (name, std) in [("N(0,1)", 1.0), ("N(0,0.01)", 0.01), ("Xavier 1/sqrt(n)", 1.0 / (n as f64).sqrt())] {
            let mut h: Vec<f64> = (0..n).map(|_| randn()).collect();
            for _ in 0..layers {
                // tanh layers of width 64
                let w: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| std * randn()).collect()).collect();
                h = (0..n).map(|i| (0..n).map(|j| w[i][j] * h[j]).sum::<f64>().tanh()).collect();
            }
            let m = h.iter().sum::<f64>() / n as f64;
            let sd = (h.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
            println!("{:<17} std of layer-{} activations = {:.4}", name, layers, sd);
        }
    }
  `), r`
    N(0,1)            std of layer-8 activations = 0.9039
    N(0,0.01)         std of layer-8 activations = 0.0000
    Xavier 1/sqrt(n)  std of layer-8 activations = 0.2771
  `, "With unit-variance weights, tanh units saturate near ±1; with tiny weights the signal shrinks to almost nothing after eight layers. Scaling the weights by 1/√n (Xavier) keeps activations in a healthy range."),

  "vanishing-gradients": ex(r`
    for depth in [1, 5, 10, 20]:
        sig = 0.25 ** depth                 # best case sigmoid slope 0.25 per layer (weights = 1)
        relu = 1.0 ** depth                 # ReLU slope 1 on the active path
        print(f"depth {depth:2d}: sigmoid factor={sig:.3e}  relu factor={relu:.1f}")
  `, r`
    fn main() {
        for depth in [1, 5, 10, 20] {
            let sig = 0.25_f64.powi(depth); // best case sigmoid slope 0.25 per layer (weights = 1)
            let relu = 1.0_f64.powi(depth); // ReLU slope 1 on the active path
            println!("depth {:2}: sigmoid factor={}  relu factor={:.1}", depth, sci(sig), relu);
        }
    }

    // format like Python's {:.3e}: mantissa with 3 decimals and a signed two-digit exponent
    fn sci(v: f64) -> String {
        let e = v.abs().log10().floor() as i32;
        let m = v / 10f64.powi(e);
        format!("{:.3}e{}{:02}", m, if e < 0 { "-" } else { "+" }, e.abs())
    }
  `, r`
    depth  1: sigmoid factor=2.500e-01  relu factor=1.0
    depth  5: sigmoid factor=9.766e-04  relu factor=1.0
    depth 10: sigmoid factor=9.537e-07  relu factor=1.0
    depth 20: sigmoid factor=9.095e-13  relu factor=1.0
  `, "Backpropagation multiplies one local slope per layer. The sigmoid's slope is at most 0.25, so after 20 layers the gradient is shrunk by about 10⁻¹²; ReLU's slope of 1 on active units passes it through unchanged."),

  "residual-connections": ex(r`
    layer_slope = 0.5                       # derivative of each residual branch F
    for depth in [5, 10, 20]:
        plain = layer_slope ** depth        # y = F(F(...)): gradients multiply
        resid = (1 + layer_slope) ** depth  # y = x + F(x): the identity adds 1 to each factor
        print(f"depth {depth:2d}: plain={plain:.6f} residual={resid:.1f}")
  `, r`
    fn main() {
        let layer_slope = 0.5_f64; // derivative of each residual branch F
        for depth in [5, 10, 20] {
            let plain = layer_slope.powi(depth); // y = F(F(...)): gradients multiply
            let resid = (1.0 + layer_slope).powi(depth); // y = x + F(x): the identity adds 1 to each factor
            println!("depth {:2}: plain={:.6} residual={:.1}", depth, plain, resid);
        }
    }
  `, r`
    depth  5: plain=0.031250 residual=7.6
    depth 10: plain=0.000977 residual=57.7
    depth 20: plain=0.000001 residual=3325.3
  `, "With y = x + F(x) each layer's Jacobian is I + F′, so there is always a path with factor 1 for the gradient. Without the skip, factors below 1 multiply into nothing. (In practice normalisation and small initial F′ keep the residual product from exploding.)"),

  "batch-normalization": ex(r`
    import math
    batch = [[1.0, 200.0], [2.0, 220.0], [3.0, 180.0], [6.0, 240.0]]   # 4 examples x 2 features
    gamma, beta, eps = [1.0, 2.0], [0.0, 0.5], 1e-5
    for j in range(2):
        col = [row[j] for row in batch]
        mu = sum(col) / len(col)
        var = sum((v - mu) ** 2 for v in col) / len(col)
        out = [gamma[j] * (v - mu) / math.sqrt(var + eps) + beta[j] for v in col]
        print(f"feature {j}: mean={mu:.1f} var={var:.2f} ->", ", ".join(f"{v:+.3f}" for v in out))
  `, r`
    fn main() {
        let batch = [[1.0, 200.0], [2.0, 220.0], [3.0, 180.0], [6.0, 240.0_f64]]; // 4 examples x 2 features
        let (gamma, beta, eps) = ([1.0, 2.0_f64], [0.0, 0.5_f64], 1e-5);
        for j in 0..2 {
            let col: Vec<f64> = batch.iter().map(|r| r[j]).collect();
            let n = col.len() as f64;
            let mu = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
            let out: Vec<String> = col.iter().map(|v| format!("{:+.3}", gamma[j] * (v - mu) / (var + eps).sqrt() + beta[j])).collect();
            println!("feature {}: mean={:.1} var={:.2} -> {}", j, mu, var, out.join(", "));
        }
    }
  `, r`
    feature 0: mean=3.0 var=3.50 -> -1.069, -0.535, +0.000, +1.604
    feature 1: mean=210.0 var=500.00 -> -0.394, +1.394, -2.183, +3.183
  `, "Each feature is standardised with its own mean and variance computed over the batch, then scaled by γ and shifted by β. Features on wildly different scales (units vs hundreds) end up comparable."),

  "layer-normalization": ex(r`
    import math
    token = [2.0, -1.0, 0.5, 4.5]           # one token's hidden vector
    mu = sum(token) / len(token)
    var = sum((v - mu) ** 2 for v in token) / len(token)
    ln = [(v - mu) / math.sqrt(var + 1e-5) for v in token]
    rms = math.sqrt(sum(v * v for v in token) / len(token))
    print(f"mean={mu:.3f} var={var:.4f}")
    print("layer norm:", ", ".join(f"{v:+.3f}" for v in ln))
    print("RMS norm  :", ", ".join(f"{v / rms:+.3f}" for v in token))
  `, r`
    fn main() {
        let token = [2.0, -1.0, 0.5, 4.5_f64]; // one token's hidden vector
        let n = token.len() as f64;
        let mu = token.iter().sum::<f64>() / n;
        let var = token.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
        let rms = (token.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        println!("mean={:.3} var={:.4}", mu, var);
        println!("layer norm: {}", token.iter().map(|v| format!("{:+.3}", (v - mu) / (var + 1e-5).sqrt())).collect::<Vec<_>>().join(", "));
        println!("RMS norm  : {}", token.iter().map(|v| format!("{:+.3}", v / rms)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    mean=1.500 var=4.1250
    layer norm: +0.246, -1.231, -0.492, +1.477
    RMS norm  : +0.792, -0.396, +0.198, +1.782
  `, "Layer norm uses the statistics of one example's own features, so it works for any batch size. RMSNorm (used in LLaMA-style models) skips the mean subtraction and only divides by the root-mean-square."),

  dropout: ex(join(PY_LCG, r`
    h = [1.0, 2.0, 3.0, 4.0]
    p = 0.5
    mask = [1.0 if rand() >= p else 0.0 for _ in h]
    print("mask   :", mask)
    print("output :", [m * v / (1 - p) for m, v in zip(mask, h)])
    total = 0.0                             # average over many masks
    for _ in range(20000):
        total += sum((1.0 if rand() >= p else 0.0) * v / (1 - p) for v in h)
    print(f"mean of sum over masks = {total / 20000:.3f} (no dropout: {sum(h):.1f})")
  `), join(RS_LCG, r`
    fn main() {
        let h = [1.0, 2.0, 3.0, 4.0_f64];
        let p = 0.5;
        let mask: Vec<f64> = h.iter().map(|_| if rand() >= p { 1.0 } else { 0.0 }).collect();
        println!("mask   : [{}]", mask.iter().map(|m| format!("{:.1}", m)).collect::<Vec<_>>().join(", "));
        println!("output : [{}]", mask.iter().zip(h.iter()).map(|(m, v)| format!("{:.1}", m * v / (1.0 - p))).collect::<Vec<_>>().join(", "));
        let mut total = 0.0; // average over many masks
        for _ in 0..20000 {
            total += h.iter().map(|v| if rand() >= p { 1.0 } else { 0.0 } * v / (1.0 - p)).sum::<f64>();
        }
        println!("mean of sum over masks = {:.3} (no dropout: {:.1})", total / 20000.0, h.iter().sum::<f64>());
    }
  `), r`
    mask   : [1.0, 1.0, 0.0, 1.0]
    output : [2.0, 4.0, 0.0, 8.0]
    mean of sum over masks = 10.024 (no dropout: 10.0)
  `, "Inverted dropout zeroes each unit with probability p and scales survivors by 1/(1 − p). Any single mask changes the output a lot, but averaged over many masks the expected value equals the no-dropout value."),

  "early-stopping": ex(r`
    val = [0.90, 0.71, 0.60, 0.55, 0.53, 0.54, 0.56, 0.52, 0.57, 0.60, 0.62]
    patience = 2
    best, best_epoch, wait = float("inf"), -1, 0
    for epoch, v in enumerate(val):
        if v < best:
            best, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"stop at epoch {epoch}; restore epoch {best_epoch} (val loss {best:.2f})")
                break
  `, r`
    fn main() {
        let val = [0.90, 0.71, 0.60, 0.55, 0.53, 0.54, 0.56, 0.52, 0.57, 0.60, 0.62_f64];
        let patience = 2;
        let (mut best, mut best_epoch, mut wait) = (f64::INFINITY, 0usize, 0);
        for (epoch, &v) in val.iter().enumerate() {
            if v < best {
                best = v;
                best_epoch = epoch;
                wait = 0;
            } else {
                wait += 1;
                if wait >= patience {
                    println!("stop at epoch {}; restore epoch {} (val loss {:.2})", epoch, best_epoch, best);
                    break;
                }
            }
        }
    }
  `, r`
    stop at epoch 6; restore epoch 4 (val loss 0.53)
  `, "With patience 2, training stops after two epochs without improvement and restores the best weights. Note the cost: a patience of 2 stops at epoch 6 and misses the slightly better epoch 7 — patience trades waiting time against such noise."),

  convolution: ex(r`
    img = [[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]]
    k = [[1, 0], [0, -1]]                   # a 2x2 diagonal-difference kernel
    out = [[sum(img[i + a][j + b] * k[a][b] for a in range(2) for b in range(2)) for j in range(3)] for i in range(3)]
    for row in out:
        print(" ".join(f"{v:+d}" for v in row))
  `, r`
    fn main() {
        let img = [[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]];
        let k = [[1, 0], [0, -1]]; // a 2x2 diagonal-difference kernel
        for i in 0..3 {
            let row: Vec<String> = (0..3).map(|j| {
                let mut s = 0;
                for a in 0..2 { for b in 0..2 { s += img[i + a][j + b] * k[a][b]; } }
                format!("{:+}", s)
            }).collect();
            println!("{}", row.join(" "));
        }
    }
  `, r`
    +0 -1 -1
    -1 +1 +3
    +2 +0 -2
  `, "A 2×2 kernel slides over the 4×4 image without padding (stride 1), giving a 3×3 output: each value is the dot product of the kernel with one patch. The same 4 weights are reused at every position."),

  pooling: ex(r`
    fm = [[1, 3, 2, 0], [4, 6, 1, 2], [0, 2, 9, 4], [1, 1, 3, 5]]
    mx = [[max(fm[i][j], fm[i][j + 1], fm[i + 1][j], fm[i + 1][j + 1]) for j in (0, 2)] for i in (0, 2)]
    av = [[(fm[i][j] + fm[i][j + 1] + fm[i + 1][j] + fm[i + 1][j + 1]) / 4 for j in (0, 2)] for i in (0, 2)]
    print("max pool:", mx)
    print("avg pool:", av)
  `, r`
    fn main() {
        let fm = [[1, 3, 2, 0], [4, 6, 1, 2], [0, 2, 9, 4], [1, 1, 3, 5]];
        let mut mx = Vec::new();
        let mut av = Vec::new();
        for i in [0, 2] {
            let mut r1 = Vec::new();
            let mut r2 = Vec::new();
            for j in [0, 2] {
                let w = [fm[i][j], fm[i][j + 1], fm[i + 1][j], fm[i + 1][j + 1]];
                r1.push(format!("{}", w.iter().max().unwrap()));
                r2.push(format!("{:?}", w.iter().sum::<i32>() as f64 / 4.0));
            }
            mx.push(format!("[{}]", r1.join(", ")));
            av.push(format!("[{}]", r2.join(", ")));
        }
        println!("max pool: [{}]", mx.join(", "));
        println!("avg pool: [{}]", av.join(", "));
    }
  `, r`
    max pool: [[6, 2], [2, 9]]
    avg pool: [[3.5, 1.25], [1.0, 5.25]]
  `, "2×2 pooling with stride 2 halves each spatial dimension. Max pooling keeps the strongest response in each window; average pooling keeps the mean. Neither has learnable parameters."),

  "cnn-architectures": ex(r`
    h, c = 32, 1                            # LeNet-5 style: input 32x32x1
    layers = [("conv", 5, 6, 1, 0), ("pool", 2, 0, 2, 0), ("conv", 5, 16, 1, 0), ("pool", 2, 0, 2, 0)]
    total = 0
    for kind, k, out_c, s, p in layers:
        params = k * k * c * out_c + out_c if kind == "conv" else 0
        h = (h + 2 * p - k) // s + 1
        c = out_c if kind == "conv" else c
        total += params
        print(f"{kind}: output {h}x{h}x{c}, params {params}")
    fc = h * h * c * 120 + 120
    print(f"flatten {h * h * c} -> fc 120: params {fc}; total so far {total + fc}")
  `, r`
    fn main() {
        let (mut h, mut c) = (32usize, 1usize); // LeNet-5 style: input 32x32x1
        let layers = [("conv", 5, 6, 1, 0), ("pool", 2, 0, 2, 0), ("conv", 5, 16, 1, 0), ("pool", 2, 0, 2, 0)];
        let mut total = 0;
        for (kind, k, out_c, s, p) in layers.iter() {
            let params = if *kind == "conv" { k * k * c * out_c + out_c } else { 0 };
            h = (h + 2 * p - k) / s + 1;
            if *kind == "conv" { c = *out_c; }
            total += params;
            println!("{}: output {}x{}x{}, params {}", kind, h, h, c, params);
        }
        let fc = h * h * c * 120 + 120;
        println!("flatten {} -> fc 120: params {}; total so far {}", h * h * c, fc, total + fc);
    }
  `, r`
    conv: output 28x28x6, params 156
    pool: output 14x14x6, params 0
    conv: output 10x10x16, params 2416
    pool: output 5x5x16, params 0
    flatten 400 -> fc 120: params 48120; total so far 50692
  `, "Applying H_out = ⌊(H + 2p − k)/s⌋ + 1 layer by layer: convolutions shrink 32 → 28 and 14 → 10, pooling halves the size. The convolutions need only 2 572 parameters, while the first fully connected layer alone needs 48 120."),

  "transfer-learning": ex(r`
    import math
    phi = lambda x: [math.sin(x), math.cos(x)]   # frozen 'pretrained' feature extractor
    xs = [0.0, 0.8, 1.6, 2.4]                    # only four labelled target examples
    ys = [2 * math.sin(x) - math.cos(x) for x in xs]
    # train only the linear head w on top of the frozen features (least squares, 2x2 system)
    F = [phi(x) for x in xs]
    a = sum(f[0] ** 2 for f in F); b = sum(f[0] * f[1] for f in F); d = sum(f[1] ** 2 for f in F)
    e = sum(f[0] * y for f, y in zip(F, ys)); g = sum(f[1] * y for f, y in zip(F, ys))
    det = a * d - b * b
    w = ((d * e - b * g) / det, (a * g - b * e) / det)
    print(f"head weights = ({w[0]:.4f}, {w[1]:.4f})")
    x_new = 3.0
    print(f"prediction at x=3: {w[0] * math.sin(x_new) + w[1] * math.cos(x_new):.4f} (true {2 * math.sin(x_new) - math.cos(x_new):.4f})")
  `, r`
    fn main() {
        let phi = |x: f64| [x.sin(), x.cos()]; // frozen 'pretrained' feature extractor
        let xs = [0.0, 0.8, 1.6, 2.4_f64]; // only four labelled target examples
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x.sin() - x.cos()).collect();
        // train only the linear head w on top of the frozen features (least squares, 2x2 system)
        let f: Vec<[f64; 2]> = xs.iter().map(|&x| phi(x)).collect();
        let a: f64 = f.iter().map(|v| v[0] * v[0]).sum();
        let b: f64 = f.iter().map(|v| v[0] * v[1]).sum();
        let d: f64 = f.iter().map(|v| v[1] * v[1]).sum();
        let e: f64 = f.iter().zip(&ys).map(|(v, y)| v[0] * y).sum();
        let g: f64 = f.iter().zip(&ys).map(|(v, y)| v[1] * y).sum();
        let det = a * d - b * b;
        let w = ((d * e - b * g) / det, (a * g - b * e) / det);
        println!("head weights = ({:.4}, {:.4})", w.0, w.1);
        let x_new = 3.0_f64;
        println!("prediction at x=3: {:.4} (true {:.4})", w.0 * x_new.sin() + w.1 * x_new.cos(), 2.0 * x_new.sin() - x_new.cos());
    }
  `, r`
    head weights = (2.0000, -1.0000)
    prediction at x=3: 1.2722 (true 1.2722)
  `, "Because the frozen features already describe the problem well, a two-parameter head fitted to just four examples recovers the target exactly and generalises to new inputs. Learning the features from four examples would be hopeless."),

  rnn: ex(r`
    import math
    wx, wh, b = 0.8, 0.5, 0.0
    h = 0.0
    for t, x in enumerate([1.0, 0.0, 0.0, -1.0, 0.0]):
        h = math.tanh(wx * x + wh * h + b)   # the same weights at every time step
        print(f"t={t} x={x:+.0f} h={h:+.4f}")
  `, r`
    fn main() {
        let (wx, wh, b) = (0.8_f64, 0.5_f64, 0.0_f64);
        let mut h = 0.0_f64;
        for (t, x) in [1.0, 0.0, 0.0, -1.0, 0.0_f64].iter().enumerate() {
            h = (wx * x + wh * h + b).tanh(); // the same weights at every time step
            println!("t={} x={:+.0} h={:+.4}", t, x, h);
        }
    }
  `, r`
    t=0 x=+1 h=+0.6640
    t=1 x=+0 h=+0.3203
    t=2 x=+0 h=+0.1588
    t=3 x=-1 h=-0.6173
    t=4 x=+0 h=-0.2992
  `, "The hidden state carries a fading memory: after the first input the state decays by about half per step (w_h = 0.5) until a new input arrives. Long-range memory requires gated cells such as LSTMs."),

  lstm: ex(r`
    import math
    sig = lambda z: 1 / (1 + math.exp(-z))
    # scalar LSTM cell with hand-set weights: (input weight, recurrent weight, bias) per gate
    Wf, Wi, Wo, Wc = (0.0, 0.0, 3.0), (2.0, 0.0, -1.0), (0.0, 0.0, 2.0), (1.0, 0.0, 0.0)
    h = c = 0.0
    for t, x in enumerate([1.0, 0.0, 0.0, 0.0, 0.0]):
        g = lambda W: W[0] * x + W[1] * h + W[2]
        f, i, o = sig(g(Wf)), sig(g(Wi)), sig(g(Wo))
        c = f * c + i * math.tanh(g(Wc))     # forget gate keeps most of the memory
        h = o * math.tanh(c)
        print(f"t={t} f={f:.3f} i={i:.3f} c={c:.4f} h={h:.4f}")
  `, r`
    fn main() {
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        // scalar LSTM cell with hand-set weights: (input weight, recurrent weight, bias) per gate
        let (wf, wi, wo, wc) = ((0.0, 0.0, 3.0), (2.0, 0.0, -1.0), (0.0, 0.0, 2.0), (1.0, 0.0, 0.0_f64));
        let (mut h, mut c) = (0.0_f64, 0.0_f64);
        for (t, &x) in [1.0, 0.0, 0.0, 0.0, 0.0_f64].iter().enumerate() {
            let g = |w: (f64, f64, f64)| w.0 * x + w.1 * h + w.2;
            let (f, i, o) = (sig(g(wf)), sig(g(wi)), sig(g(wo)));
            c = f * c + i * g(wc).tanh(); // forget gate keeps most of the memory
            h = o * c.tanh();
            println!("t={} f={:.3} i={:.3} c={:.4} h={:.4}", t, f, i, c, h);
        }
    }
  `, r`
    t=0 f=0.953 i=0.731 c=0.5568 h=0.4453
    t=1 f=0.953 i=0.269 c=0.5304 h=0.4278
    t=2 f=0.953 i=0.269 c=0.5052 h=0.4106
    t=3 f=0.953 i=0.269 c=0.4813 h=0.3939
    t=4 f=0.953 i=0.269 c=0.4584 h=0.3777
  `, "The cell state c is updated additively and multiplied by the forget gate (≈ 0.95 here), so the information from t = 0 fades far more slowly than in the plain RNN example. The input gate controls how much new content is written."),

  "bidirectional-rnn": ex(r`
    import math
    xs = [1.0, 0.0, 0.0, 2.0]
    step = lambda x, h: math.tanh(0.8 * x + 0.5 * h)
    fwd, h = [], 0.0
    for x in xs:                             # left to right
        h = step(x, h); fwd.append(h)
    bwd, h = [0.0] * len(xs), 0.0
    for t in reversed(range(len(xs))):       # right to left
        h = step(xs[t], h); bwd[t] = h
    for t in range(len(xs)):
        print(f"t={t}: forward={fwd[t]:+.3f} backward={bwd[t]:+.3f}")
  `, r`
    fn main() {
        let xs = [1.0, 0.0, 0.0, 2.0_f64];
        let step = |x: f64, h: f64| (0.8 * x + 0.5 * h).tanh();
        let mut fwd = Vec::new();
        let mut h = 0.0;
        for &x in xs.iter() {
            // left to right
            h = step(x, h);
            fwd.push(h);
        }
        let mut bwd = vec![0.0; xs.len()];
        h = 0.0;
        for t in (0..xs.len()).rev() {
            // right to left
            h = step(xs[t], h);
            bwd[t] = h;
        }
        for t in 0..xs.len() {
            println!("t={}: forward={:+.3} backward={:+.3}", t, fwd[t], bwd[t]);
        }
    }
  `, r`
    t=0: forward=+0.664 backward=+0.719
    t=1: forward=+0.320 backward=+0.212
    t=2: forward=+0.159 backward=+0.431
    t=3: forward=+0.933 backward=+0.922
  `, "At t = 1 the forward state only remembers the first input, while the backward state already 'knows' about the large input arriving at t = 3. Concatenating both gives every position context from both sides."),

  seq2seq: ex(r`
    emb = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (1.0, 1.0)}
    def encode(seq):                          # a fixed-size context: average of the token vectors
        v = [emb[t] for t in seq]
        return tuple(sum(x[k] for x in v) / len(v) for k in range(2))
    for s in ["abc", "cba", "bca"]:
        print(f"{s} -> context {encode(s)}")
    print("all three contexts are identical, so a decoder cannot recover the order")
  `, r`
    fn main() {
        let emb = |t: char| match t { 'a' => (1.0, 0.0), 'b' => (0.0, 1.0), _ => (1.0, 1.0_f64) };
        // a fixed-size context: average of the token vectors
        let encode = |s: &str| {
            let v: Vec<(f64, f64)> = s.chars().map(emb).collect();
            let n = v.len() as f64;
            (v.iter().map(|x| x.0).sum::<f64>() / n, v.iter().map(|x| x.1).sum::<f64>() / n)
        };
        for s in ["abc", "cba", "bca"] {
            let c = encode(s);
            println!("{} -> context ({:?}, {:?})", s, c.0, c.1);
        }
        println!("all three contexts are identical, so a decoder cannot recover the order");
    }
  `, r`
    abc -> context (0.6666666666666666, 0.6666666666666666)
    cba -> context (0.6666666666666666, 0.6666666666666666)
    bca -> context (0.6666666666666666, 0.6666666666666666)
    all three contexts are identical, so a decoder cannot recover the order
  `, "An encoder–decoder passes the input through a context. If that context is too lossy (here an order-blind average) different inputs collapse to the same vector. RNN encoders keep order but still squeeze everything into one fixed vector — the bottleneck attention removes."),

  autoencoders: ex(r`
    import math
    X = [(2.0, 1.9), (1.0, 1.1), (-1.0, -0.9), (-2.0, -2.1), (0.5, 0.4)]
    w = [1.0, 0.0]                            # tied linear autoencoder: code z = w.x, reconstruction w z
    for _ in range(500):
        g = [0.0, 0.0]
        for x in X:
            z = w[0] * x[0] + w[1] * x[1]
            r_ = [x[0] - w[0] * z, x[1] - w[1] * z]
            for k in range(2):                # gradient of the reconstruction error
                g[k] += -2 * (r_[k] * z + (r_[0] * w[0] + r_[1] * w[1]) * x[k])
        w = [w[k] - 0.01 * g[k] / len(X) for k in range(2)]
    err = sum((x[0] - w[0] * (w[0] * x[0] + w[1] * x[1])) ** 2 + (x[1] - w[1] * (w[0] * x[0] + w[1] * x[1])) ** 2 for x in X) / len(X)
    print(f"learned direction w = ({w[0]:.4f}, {w[1]:.4f}), |w| = {math.hypot(*w):.4f}")
    print(f"mean reconstruction error with a 1-number code = {err:.4f}")
  `, r`
    fn main() {
        let x = [(2.0, 1.9), (1.0, 1.1), (-1.0, -0.9), (-2.0, -2.1), (0.5, 0.4_f64)];
        let mut w = [1.0_f64, 0.0]; // tied linear autoencoder: code z = w.x, reconstruction w z
        for _ in 0..500 {
            let mut g = [0.0_f64; 2];
            for p in x.iter() {
                let xv = [p.0, p.1];
                let z = w[0] * xv[0] + w[1] * xv[1];
                let r = [xv[0] - w[0] * z, xv[1] - w[1] * z];
                for k in 0..2 {
                    // gradient of the reconstruction error
                    g[k] += -2.0 * (r[k] * z + (r[0] * w[0] + r[1] * w[1]) * xv[k]);
                }
            }
            for k in 0..2 { w[k] -= 0.01 * g[k] / x.len() as f64; }
        }
        let err: f64 = x.iter().map(|p| { let z = w[0] * p.0 + w[1] * p.1; (p.0 - w[0] * z).powi(2) + (p.1 - w[1] * z).powi(2) }).sum::<f64>() / x.len() as f64;
        println!("learned direction w = ({:.4}, {:.4}), |w| = {:.4}", w[0], w[1], w[0].hypot(w[1]));
        println!("mean reconstruction error with a 1-number code = {:.4}", err);
    }
  `, r`
    learned direction w = (0.7080, 0.7062), |w| = 1.0000
    mean reconstruction error with a 1-number code = 0.0050
  `, "Training a tied linear autoencoder with a one-number bottleneck rotates w onto the data's main diagonal direction with unit length — exactly the first principal component. Each 2-D point is then stored as one number with a tiny reconstruction error."),

  vae: ex(r`
    import math
    mu, log_var = 0.8, math.log(0.25)         # encoder output for one input
    eps = -0.5                                # one draw from N(0, 1)
    z = mu + math.exp(0.5 * log_var) * eps    # reparameterisation trick
    kl = 0.5 * (mu ** 2 + math.exp(log_var) - 1 - log_var)
    recon = 0.5 * (2.0 - 1.5 * z) ** 2        # toy decoder 1.5 z, target x = 2.0
    print(f"z = {z:.4f}")
    print(f"KL(q || N(0,1)) = {kl:.4f}")
    print(f"negative ELBO = recon + KL = {recon:.4f} + {kl:.4f} = {recon + kl:.4f}")
  `, r`
    fn main() {
        let (mu, log_var) = (0.8_f64, 0.25_f64.ln()); // encoder output for one input
        let eps = -0.5; // one draw from N(0, 1)
        let z = mu + (0.5 * log_var).exp() * eps; // reparameterisation trick
        let kl = 0.5 * (mu * mu + log_var.exp() - 1.0 - log_var);
        let recon = 0.5 * (2.0 - 1.5 * z).powi(2); // toy decoder 1.5 z, target x = 2.0
        println!("z = {:.4}", z);
        println!("KL(q || N(0,1)) = {:.4}", kl);
        println!("negative ELBO = recon + KL = {:.4} + {:.4} = {:.4}", recon, kl, recon + kl);
    }
  `, r`
    z = 0.5500
    KL(q || N(0,1)) = 0.6381
    negative ELBO = recon + KL = 0.6903 + 0.6381 = 1.3285
  `, "The encoder outputs a mean and log-variance; z = μ + σ·ε turns sampling into a differentiable computation. The loss adds reconstruction error to the closed-form KL term that keeps q(z | x) close to the prior."),

  gans: ex(r`
    import math
    sig = lambda z: 1 / (1 + math.exp(-z))
    a, b = 1.5, -3.0                          # discriminator D(x) = sigmoid(a x + b)
    real, fake = [2.8, 3.1, 3.5], [0.5, 1.0, 1.4]
    d_real = [sig(a * x + b) for x in real]
    d_fake = [sig(a * x + b) for x in fake]
    loss_d = -sum(math.log(p) for p in d_real) / 3 - sum(math.log(1 - p) for p in d_fake) / 3
    loss_g = -sum(math.log(p) for p in d_fake) / 3   # non-saturating generator loss
    print("D(real):", ", ".join(f"{p:.3f}" for p in d_real))
    print("D(fake):", ", ".join(f"{p:.3f}" for p in d_fake))
    print(f"discriminator loss = {loss_d:.4f}, generator loss = {loss_g:.4f}")
  `, r`
    fn main() {
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let (a, b) = (1.5_f64, -3.0_f64); // discriminator D(x) = sigmoid(a x + b)
        let (real, fake) = ([2.8, 3.1, 3.5_f64], [0.5, 1.0, 1.4_f64]);
        let d_real: Vec<f64> = real.iter().map(|x| sig(a * x + b)).collect();
        let d_fake: Vec<f64> = fake.iter().map(|x| sig(a * x + b)).collect();
        let loss_d = -d_real.iter().map(|p| p.ln()).sum::<f64>() / 3.0 - d_fake.iter().map(|p| (1.0 - p).ln()).sum::<f64>() / 3.0;
        let loss_g = -d_fake.iter().map(|p| p.ln()).sum::<f64>() / 3.0; // non-saturating generator loss
        println!("D(real): {}", d_real.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>().join(", "));
        println!("D(fake): {}", d_fake.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>().join(", "));
        println!("discriminator loss = {:.4}, generator loss = {:.4}", loss_d, loss_g);
    }
  `, r`
    D(real): 0.769, 0.839, 0.905
    D(fake): 0.095, 0.182, 0.289
    discriminator loss = 0.3940, generator loss = 1.7643
  `, "This discriminator separates real from fake samples well, so its loss is small while the generator's loss is large — the generator's gradient will push its samples towards the region D considers real."),

  "diffusion-models": ex(r`
    import math
    T = 10
    betas = [0.02 + (0.4 - 0.02) * t / (T - 1) for t in range(T)]
    abar, alpha_bar = 1.0, []
    for b in betas:
        abar *= 1 - b
        alpha_bar.append(abar)
    x0, eps = 2.0, 0.5
    for t in [0, 4, 9]:
        a = alpha_bar[t]
        xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * eps     # jump straight to step t
        print(f"t={t + 1:2d}: alpha_bar={a:.4f} signal={math.sqrt(a):.3f} noise={math.sqrt(1 - a):.3f} x_t={xt:.4f}")
  `, r`
    fn main() {
        let t_steps = 10;
        let betas: Vec<f64> = (0..t_steps).map(|t| 0.02 + (0.4 - 0.02) * t as f64 / (t_steps - 1) as f64).collect();
        let mut abar = 1.0;
        let mut alpha_bar = Vec::new();
        for b in betas.iter() {
            abar *= 1.0 - b;
            alpha_bar.push(abar);
        }
        let (x0, eps) = (2.0_f64, 0.5_f64);
        for t in [0, 4, 9] {
            let a: f64 = alpha_bar[t];
            let xt = a.sqrt() * x0 + (1.0 - a).sqrt() * eps; // jump straight to step t
            println!("t={:2}: alpha_bar={:.4} signal={:.3} noise={:.3} x_t={:.4}", t + 1, a, a.sqrt(), (1.0 - a).sqrt(), xt);
        }
    }
  `, r`
    t= 1: alpha_bar=0.9800 signal=0.990 noise=0.141 x_t=2.0506
    t= 5: alpha_bar=0.5697 signal=0.755 noise=0.656 x_t=1.8375
    t=10: alpha_bar=0.0839 signal=0.290 noise=0.957 x_t=1.0580
  `, "The forward process has a closed form: x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε. As ᾱ_t falls towards 0 the signal fades and x_t becomes pure noise; the model is trained to predict ε from x_t so it can run the process backwards."),

  "u-net": ex(r`
    size, ch, depth = 128, 32, 4
    skips = []
    for level in range(depth):                # encoder: conv blocks then downsample
        print(f"enc {level}: {size}x{size}x{ch}")
        skips.append((size, ch))
        size, ch = size // 2, ch * 2
    print(f"bottleneck: {size}x{size}x{ch}")
    for s, c in reversed(skips):              # decoder: upsample and concatenate the skip
        size = s
        print(f"dec: up to {size}x{size}, concat skip -> {c + ch} channels -> conv to {c}")
        ch = c
  `, r`
    fn main() {
        let (mut size, mut ch, depth) = (128usize, 32usize, 4);
        let mut skips = Vec::new();
        for level in 0..depth {
            // encoder: conv blocks then downsample
            println!("enc {}: {}x{}x{}", level, size, size, ch);
            skips.push((size, ch));
            size /= 2;
            ch *= 2;
        }
        println!("bottleneck: {}x{}x{}", size, size, ch);
        for &(s, c) in skips.iter().rev() {
            // decoder: upsample and concatenate the skip
            size = s;
            println!("dec: up to {}x{}, concat skip -> {} channels -> conv to {}", size, size, c + ch, c);
            ch = c;
        }
    }
  `, r`
    enc 0: 128x128x32
    enc 1: 64x64x64
    enc 2: 32x32x128
    enc 3: 16x16x256
    bottleneck: 8x8x512
    dec: up to 16x16, concat skip -> 768 channels -> conv to 256
    dec: up to 32x32, concat skip -> 384 channels -> conv to 128
    dec: up to 64x64, concat skip -> 192 channels -> conv to 64
    dec: up to 128x128, concat skip -> 96 channels -> conv to 32
  `, "The encoder halves the resolution and doubles the channels at each level. Each decoder level upsamples and concatenates the encoder feature map of the same size, so fine spatial detail skips the bottleneck."),
};
