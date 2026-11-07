import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";
import { dedent } from "@/lib/utilities/highlight";

/* Shared print helpers, repeated in each snippet so every example runs on its own. */
const PY_SHOW = `def show(v):
    return ", ".join(f"{x + 0.0:.3f}" for x in v)`;
const RS_SHOW = `fn show(v: &[f64]) -> String {
    v.iter().map(|x| format!("{:.3}", x + 0.0)).collect::<Vec<_>>().join(", ")
}`;
const join = (...parts: string[]) => parts.map(dedent).join("\n\n");

export const preprocessingCode: Record<string, CodeExample> = {
  "missing-data": ex(join(PY_SHOW, r`
    ages = [23.0, None, 35.0, 41.0, None, 29.0]
    known = [a for a in ages if a is not None]
    mean = sum(known) / len(known)
    s = sorted(known)
    median = (s[1] + s[2]) / 2                     # 4 known values -> average the middle two
    indicator = [1.0 if a is None else 0.0 for a in ages]
    print("mean   :", show([mean if a is None else a for a in ages]))
    print("median :", show([median if a is None else a for a in ages]))
    print("missing:", show(indicator))
  `), join(RS_SHOW, r`
    fn main() {
        let ages: [Option<f64>; 6] = [Some(23.0), None, Some(35.0), Some(41.0), None, Some(29.0)];
        let mut known: Vec<f64> = ages.iter().flatten().cloned().collect();
        let mean = known.iter().sum::<f64>() / known.len() as f64;
        known.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = (known[1] + known[2]) / 2.0; // 4 known values -> average the middle two
        let fill = |v: f64| ages.iter().map(|a| a.unwrap_or(v)).collect::<Vec<f64>>();
        let indicator: Vec<f64> = ages.iter().map(|a| if a.is_none() { 1.0 } else { 0.0 }).collect();
        println!("mean   : {}", show(&fill(mean)));
        println!("median : {}", show(&fill(median)));
        println!("missing: {}", show(&indicator));
    }
  `), r`
    mean   : 23.000, 32.000, 35.000, 41.000, 32.000, 29.000
    median : 23.000, 32.000, 35.000, 41.000, 32.000, 29.000
    missing: 0.000, 1.000, 0.000, 0.000, 1.000, 0.000
  `, "Mean and median imputation fill the gaps with one typical value (here 32 for both); the missing-indicator column keeps the information that a value was absent, which can itself be predictive. Compute the fill values on the training split only."),

  "outlier-treatment": ex(join(PY_SHOW, r`
    x = [12.0, 14.0, 15.0, 15.0, 16.0, 18.0, 19.0, 95.0]
    s = sorted(x)
    def q(p):                                  # linear-interpolation quantile
        pos = p * (len(s) - 1)
        lo = int(pos)
        return s[lo] + (pos - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])
    q1, q3 = q(0.25), q(0.75)
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"Q1={q1:.2f} Q3={q3:.2f} fences=[{lo_f:.2f}, {hi_f:.2f}]")
    print("outliers:", [v for v in x if v < lo_f or v > hi_f])
    print("clipped :", show([min(max(v, lo_f), hi_f) for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [12.0, 14.0, 15.0, 15.0, 16.0, 18.0, 19.0, 95.0_f64];
        let mut s = x.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // linear-interpolation quantile
        let q = |p: f64| {
            let pos = p * (s.len() - 1) as f64;
            let lo = pos.floor() as usize;
            s[lo] + (pos - lo as f64) * (s[(lo + 1).min(s.len() - 1)] - s[lo])
        };
        let (q1, q3) = (q(0.25), q(0.75));
        let iqr = q3 - q1;
        let (lo_f, hi_f) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
        println!("Q1={:.2} Q3={:.2} fences=[{:.2}, {:.2}]", q1, q3, lo_f, hi_f);
        let out: Vec<String> = x.iter().filter(|&&v| v < lo_f || v > hi_f).map(|v| format!("{:.1}", v)).collect();
        println!("outliers: [{}]", out.join(", "));
        let clipped: Vec<f64> = x.iter().map(|&v| v.max(lo_f).min(hi_f)).collect();
        println!("clipped : {}", show(&clipped));
    }
  `), r`
    Q1=14.75 Q3=18.25 fences=[9.50, 23.50]
    outliers: [95.0]
    clipped : 12.000, 14.000, 15.000, 15.000, 16.000, 18.000, 19.000, 23.500
  `, "Tukey's fences mark values more than 1.5 × IQR beyond the quartiles. Only 95 is flagged; clipping (winsorising) pulls it to the upper fence instead of deleting the row."),

  "robust-scaling": ex(join(PY_SHOW, r`
    x = [10.0, 12.0, 13.0, 15.0, 16.0, 200.0]
    s = sorted(x)
    def q(p):
        pos = p * (len(s) - 1)
        lo = int(pos)
        return s[lo] + (pos - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])
    med, iqr = q(0.5), q(0.75) - q(0.25)
    print(f"median={med:.2f} IQR={iqr:.2f}")
    print("robust:", show([(v - med) / iqr for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [10.0, 12.0, 13.0, 15.0, 16.0, 200.0_f64];
        let mut s = x.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| {
            let pos = p * (s.len() - 1) as f64;
            let lo = pos.floor() as usize;
            s[lo] + (pos - lo as f64) * (s[(lo + 1).min(s.len() - 1)] - s[lo])
        };
        let (med, iqr) = (q(0.5), q(0.75) - q(0.25));
        println!("median={:.2} IQR={:.2}", med, iqr);
        let scaled: Vec<f64> = x.iter().map(|v| (v - med) / iqr).collect();
        println!("robust: {}", show(&scaled));
    }
  `), r`
    median=14.00 IQR=3.50
    robust: -1.143, -0.571, -0.286, 0.286, 0.571, 53.143
  `, "Centring on the median and dividing by the IQR ignores the extreme 200, so the ordinary values keep a sensible spread around 0 while the outlier stays clearly far away. Standardisation would let the outlier inflate the standard deviation and squash everything else."),

  "feature-scaling": ex(join(PY_SHOW, r`
    import math
    x = [1.0, 2.0, 3.0, 100.0]
    lo, hi = min(x), max(x)
    mu = sum(x) / len(x)
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))
    print("min-max :", show([(v - lo) / (hi - lo) for v in x]))
    print("z-score :", show([(v - mu) / sd for v in x]))
    print("max-abs :", show([v / max(abs(t) for t in x) for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [1.0, 2.0, 3.0, 100.0_f64];
        let (lo, hi) = (x.iter().cloned().fold(f64::MAX, f64::min), x.iter().cloned().fold(f64::MIN, f64::max));
        let mu = x.iter().sum::<f64>() / x.len() as f64;
        let sd = (x.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        let amax = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("min-max : {}", show(&x.iter().map(|v| (v - lo) / (hi - lo)).collect::<Vec<_>>()));
        println!("z-score : {}", show(&x.iter().map(|v| (v - mu) / sd).collect::<Vec<_>>()));
        println!("max-abs : {}", show(&x.iter().map(|v| v / amax).collect::<Vec<_>>()));
    }
  `), r`
    min-max : 0.000, 0.010, 0.020, 1.000
    z-score : -0.601, -0.577, -0.554, 1.732
    max-abs : 0.010, 0.020, 0.030, 1.000
  `, "All three put the feature on a comparable scale, but one extreme value (100) squashes the other three near 0 in every case — a hint that a log or robust transform may be needed first."),

  "min-max-normalization": ex(join(PY_SHOW, r`
    train = [20.0, 35.0, 50.0, 65.0, 80.0]
    test = [10.0, 50.0, 95.0]
    lo, hi = min(train), max(train)          # fitted on the training data only
    scale = lambda v: (v - lo) / (hi - lo)
    print("train:", show([scale(v) for v in train]))
    print("test :", show([scale(v) for v in test]))
  `), join(RS_SHOW, r`
    fn main() {
        let train = [20.0, 35.0, 50.0, 65.0, 80.0_f64];
        let test = [10.0, 50.0, 95.0_f64];
        // fitted on the training data only
        let lo = train.iter().cloned().fold(f64::MAX, f64::min);
        let hi = train.iter().cloned().fold(f64::MIN, f64::max);
        let scale = |v: &f64| (v - lo) / (hi - lo);
        println!("train: {}", show(&train.iter().map(scale).collect::<Vec<_>>()));
        println!("test : {}", show(&test.iter().map(scale).collect::<Vec<_>>()));
    }
  `), r`
    train: 0.000, 0.250, 0.500, 0.750, 1.000
    test : -0.167, 0.500, 1.250
  `, "Training values map exactly onto [0, 1]. Test values outside the training range fall outside [0, 1] (−0.167 and 1.250) — expected behaviour, which is why the scaler must be fitted on training data and reused unchanged."),

  standardization: ex(join(PY_SHOW, r`
    import math
    heights = [150.0, 160.0, 165.0, 170.0, 185.0]
    mu = sum(heights) / len(heights)
    sd = math.sqrt(sum((h - mu) ** 2 for h in heights) / len(heights))
    z = [(h - mu) / sd for h in heights]
    print(f"mean={mu:.2f} std={sd:.3f}")
    print("z    :", show(z))
    print(f"check: mean(z)={abs(sum(z) / len(z)):.3f} std(z)={math.sqrt(sum(v * v for v in z) / len(z)):.3f}")
  `), join(RS_SHOW, r`
    fn main() {
        let h = [150.0, 160.0, 165.0, 170.0, 185.0_f64];
        let n = h.len() as f64;
        let mu = h.iter().sum::<f64>() / n;
        let sd = (h.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
        let z: Vec<f64> = h.iter().map(|v| (v - mu) / sd).collect();
        println!("mean={:.2} std={:.3}", mu, sd);
        println!("z    : {}", show(&z));
        let zm = z.iter().sum::<f64>() / n;
        let zs = (z.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        println!("check: mean(z)={:.3} std(z)={:.3}", zm.abs(), zs);
    }
  `), r`
    mean=166.00 std=11.576
    z    : -1.382, -0.518, -0.086, 0.346, 1.641
    check: mean(z)=0.000 std(z)=1.000
  `, "After standardisation the feature has mean 0 and standard deviation 1; each value now says how many standard deviations it lies from the average."),

  "maxabs-scaling": ex(join(PY_SHOW, r`
    x = [-4.0, 0.0, 0.0, 2.0, 8.0]
    m = max(abs(v) for v in x)
    print(f"max |x| = {m:.1f}")
    print("scaled:", show([v / m for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [-4.0, 0.0, 0.0, 2.0, 8.0_f64];
        let m = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("max |x| = {:.1}", m);
        println!("scaled: {}", show(&x.iter().map(|v| v / m).collect::<Vec<_>>()));
    }
  `), r`
    max |x| = 8.0
    scaled: -0.500, 0.000, 0.000, 0.250, 1.000
  `, "Dividing by the largest absolute value maps the data into [−1, 1] without shifting it, so zeros stay zeros — the reason MaxAbs scaling is used for sparse matrices."),

  "unit-vector-normalization": ex(join(PY_SHOW, r`
    import math
    docs = [[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [10.0, 0.0, 0.0]]
    for d in docs:
        n = math.sqrt(sum(v * v for v in d))
        u = [v / n for v in d]
        print(f"norm={n:.3f} -> [{show(u)}] length={math.sqrt(sum(v * v for v in u)):.3f}")
  `), join(RS_SHOW, r`
    fn main() {
        let docs = [[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [10.0, 0.0, 0.0_f64]];
        for d in docs.iter() {
            let n = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let u: Vec<f64> = d.iter().map(|v| v / n).collect();
            let len = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("norm={:.3} -> [{}] length={:.3}", n, show(&u), len);
        }
    }
  `), r`
    norm=5.000 -> [0.600, 0.800, 0.000] length=1.000
    norm=1.732 -> [0.577, 0.577, 0.577] length=1.000
    norm=10.000 -> [1.000, 0.000, 0.000] length=1.000
  `, "Each row (sample) is divided by its own L2 norm, so every vector ends up with length 1 and only its direction remains — dot products then equal cosine similarities."),

  "decimal-scaling": ex(join(PY_SHOW, r`
    import math
    x = [-725.0, 18.0, 940.0, 312.0]
    j = math.ceil(math.log10(max(abs(v) for v in x)))   # smallest j with max|x| / 10^j < 1
    print(f"j = {j}")
    print("scaled:", show([v / 10 ** j for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [-725.0, 18.0, 940.0, 312.0_f64];
        let m = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let j = m.log10().ceil() as i32; // smallest j with max|x| / 10^j < 1
        println!("j = {}", j);
        println!("scaled: {}", show(&x.iter().map(|v| v / 10f64.powi(j)).collect::<Vec<_>>()));
    }
  `), r`
    j = 3
    scaled: -0.725, 0.018, 0.940, 0.312
  `, "Decimal scaling moves the decimal point: the largest magnitude (940) needs j = 3, so every value is divided by 1000. Simple and reversible, but it does not centre the data."),

  "log-transformation": ex(join(PY_SHOW, r`
    import math
    def skew(v):
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        return sum(((x - m) / s) ** 3 for x in v) / len(v)
    income = [20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 90.0, 250.0]
    logged = [math.log(v) for v in income]
    print("log:", show(logged))
    print(f"skewness before={skew(income):.3f} after={skew(logged):.3f}")
  `), join(RS_SHOW, r`
    fn skew(v: &[f64]) -> f64 {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n).sqrt();
        v.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
    }

    fn main() {
        let income = [20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 90.0, 250.0_f64];
        let logged: Vec<f64> = income.iter().map(|v| v.ln()).collect();
        println!("log: {}", show(&logged));
        println!("skewness before={:.3} after={:.3}", skew(&income), skew(&logged));
    }
  `), r`
    log: 2.996, 3.219, 3.401, 3.555, 3.689, 4.094, 4.500, 5.521
    skewness before=1.913 after=1.009
  `, "A right-skewed variable (one very large income) becomes far more symmetric after taking logs: the skewness drops sharply. Multiplicative differences become additive."),

  "log1p-transformation": ex(join(PY_SHOW, r`
    import math
    clicks = [0.0, 0.0, 1.0, 3.0, 10.0, 1000.0]
    print("log1p :", show([math.log1p(v) for v in clicks]))
    print("inverse:", show([math.expm1(math.log1p(v)) for v in clicks]))
  `), join(RS_SHOW, r`
    fn main() {
        let clicks = [0.0, 0.0, 1.0, 3.0, 10.0, 1000.0_f64];
        let t: Vec<f64> = clicks.iter().map(|v| v.ln_1p()).collect();
        println!("log1p : {}", show(&t));
        println!("inverse: {}", show(&t.iter().map(|v| v.exp_m1()).collect::<Vec<_>>()));
    }
  `), r`
    log1p : 0.000, 0.000, 0.693, 1.386, 2.398, 6.909
    inverse: 0.000, 0.000, 1.000, 3.000, 10.000, 1000.000
  `, "log(1 + x) is defined at 0, so count data with zeros can be log-transformed safely; expm1 inverts it exactly. Both functions are also more accurate than log(1 + x) for tiny x."),

  "square-root-transformation": ex(join(PY_SHOW, r`
    import math
    counts = [0.0, 1.0, 4.0, 9.0, 16.0, 100.0]
    print("sqrt:", show([math.sqrt(v) for v in counts]))
  `), join(RS_SHOW, r`
    fn main() {
        let counts = [0.0, 1.0, 4.0, 9.0, 16.0, 100.0_f64];
        println!("sqrt: {}", show(&counts.iter().map(|v| v.sqrt()).collect::<Vec<_>>()));
    }
  `), r`
    sqrt: 0.000, 1.000, 2.000, 3.000, 4.000, 10.000
  `, "The square root compresses large values less aggressively than a log and keeps zero at zero — a classic variance-stabilising transform for Poisson-like counts."),

  "cube-root-transformation": ex(join(PY_SHOW, r`
    import math
    change = [-27.0, -8.0, -1.0, 0.0, 1.0, 64.0]
    cbrt = lambda v: math.copysign(abs(v) ** (1 / 3), v)
    print("cbrt:", show([cbrt(v) for v in change]))
  `), join(RS_SHOW, r`
    fn main() {
        let change = [-27.0, -8.0, -1.0, 0.0, 1.0, 64.0_f64];
        println!("cbrt: {}", show(&change.iter().map(|v| v.cbrt()).collect::<Vec<_>>()));
    }
  `), r`
    cbrt: -3.000, -2.000, -1.000, 0.000, 1.000, 4.000
  `, "Unlike the log and square root, the cube root is defined for negative numbers and keeps their sign, so it can tame heavy tails on both sides of zero."),

  "reciprocal-transformation": ex(join(PY_SHOW, r`
    minutes_per_km = [4.0, 5.0, 6.0, 8.0, 12.0]
    speed = [60.0 / v for v in minutes_per_km]          # km per hour
    print("km/h:", show(speed))
  `), join(RS_SHOW, r`
    fn main() {
        let minutes_per_km = [4.0, 5.0, 6.0, 8.0, 12.0_f64];
        let speed: Vec<f64> = minutes_per_km.iter().map(|v| 60.0 / v).collect(); // km per hour
        println!("km/h: {}", show(&speed));
    }
  `), r`
    km/h: 15.000, 12.000, 10.000, 7.500, 5.000
  `, "The reciprocal reverses the order and can turn a rate into a more natural quantity (pace → speed). It is undefined at 0 and changes large values very little."),

  "power-transformation": ex(join(PY_SHOW, r`
    x = [1.0, 2.0, 4.0, 8.0]
    for p in [0.5, 1.0, 2.0]:
        print(f"p={p:.1f}:", show([v ** p for v in x]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [1.0, 2.0, 4.0, 8.0_f64];
        for p in [0.5, 1.0, 2.0] {
            println!("p={:.1}: {}", p, show(&x.iter().map(|v| v.powf(p)).collect::<Vec<_>>()));
        }
    }
  `), r`
    p=0.5: 1.000, 1.414, 2.000, 2.828
    p=1.0: 1.000, 2.000, 4.000, 8.000
    p=2.0: 1.000, 4.000, 16.000, 64.000
  `, "Powers below 1 compress large values (reduce right skew); powers above 1 stretch them. Box–Cox and Yeo–Johnson choose the exponent automatically."),

  "box-cox-transformation": ex(join(PY_SHOW, r`
    import math
    x = [1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 15.0]
    def boxcox(v, lam):
        return math.log(v) if abs(lam) < 1e-12 else (v ** lam - 1) / lam
    def loglik(lam):                    # profile log-likelihood of a normal model on the transformed data
        y = [boxcox(v, lam) for v in x]
        m = sum(y) / len(y)
        var = sum((t - m) ** 2 for t in y) / len(y)
        return -len(x) / 2 * math.log(var) + (lam - 1) * sum(math.log(v) for v in x)
    best = max((l / 10 for l in range(-20, 21)), key=loglik)
    print(f"best lambda = {best:.1f}")
    print("transformed:", show([boxcox(v, best) for v in x]))
  `), join(RS_SHOW, r`
    fn boxcox(v: f64, lam: f64) -> f64 {
        if lam.abs() < 1e-12 { v.ln() } else { (v.powf(lam) - 1.0) / lam }
    }

    fn main() {
        let x = [1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 15.0_f64];
        let n = x.len() as f64;
        // profile log-likelihood of a normal model on the transformed data
        let loglik = |lam: f64| {
            let y: Vec<f64> = x.iter().map(|&v| boxcox(v, lam)).collect();
            let m = y.iter().sum::<f64>() / n;
            let var = y.iter().map(|t| (t - m).powi(2)).sum::<f64>() / n;
            -n / 2.0 * var.ln() + (lam - 1.0) * x.iter().map(|v| v.ln()).sum::<f64>()
        };
        let mut best = -2.0;
        for l in -20..=20 {
            let lam = l as f64 / 10.0;
            if loglik(lam) > loglik(best) { best = lam; }
        }
        println!("best lambda = {:.1}", best);
        println!("transformed: {}", show(&x.iter().map(|&v| boxcox(v, best)).collect::<Vec<_>>()));
    }
  `), r`
    best lambda = -0.1
    transformed: 0.000, 0.670, 0.876, 1.040, 1.294, 1.640, 1.973, 2.372
  `, "Box–Cox searches the exponent λ that makes the transformed data look most normal (highest log-likelihood). For this right-skewed sample the best λ is close to 0, i.e. close to a log transform. It requires strictly positive values."),

  "yeo-johnson-transformation": ex(join(PY_SHOW, r`
    import math
    def yj(v, lam):
        if v >= 0:
            return math.log1p(v) if abs(lam) < 1e-12 else ((v + 1) ** lam - 1) / lam
        return -math.log1p(-v) if abs(lam - 2) < 1e-12 else -((1 - v) ** (2 - lam) - 1) / (2 - lam)
    x = [-3.0, -1.0, 0.0, 1.0, 3.0, 10.0]
    for lam in [0.5, 1.0]:
        print(f"lambda={lam:.1f}:", show([yj(v, lam) for v in x]))
  `), join(RS_SHOW, r`
    fn yj(v: f64, lam: f64) -> f64 {
        if v >= 0.0 {
            if lam.abs() < 1e-12 { v.ln_1p() } else { ((v + 1.0).powf(lam) - 1.0) / lam }
        } else if (lam - 2.0).abs() < 1e-12 {
            -(-v).ln_1p()
        } else {
            -((1.0 - v).powf(2.0 - lam) - 1.0) / (2.0 - lam)
        }
    }

    fn main() {
        let x = [-3.0, -1.0, 0.0, 1.0, 3.0, 10.0_f64];
        for lam in [0.5, 1.0] {
            println!("lambda={:.1}: {}", lam, show(&x.iter().map(|&v| yj(v, lam)).collect::<Vec<_>>()));
        }
    }
  `), r`
    lambda=0.5: -4.667, -1.219, 0.000, 0.828, 2.000, 4.633
    lambda=1.0: -3.000, -1.000, 0.000, 1.000, 3.000, 10.000
  `, "Yeo–Johnson extends Box–Cox to zero and negative values by treating the two sides separately. λ = 1 is the identity; λ = 0.5 compresses the large positive tail (10 → 4.633) while keeping order."),

  "exponential-transformation": ex(join(PY_SHOW, r`
    import math
    log_odds = [-2.0, -1.0, 0.0, 1.0, 2.0]
    print("exp(x):", show([math.exp(v) for v in log_odds]))
  `), join(RS_SHOW, r`
    fn main() {
        let log_odds = [-2.0, -1.0, 0.0, 1.0, 2.0_f64];
        println!("exp(x): {}", show(&log_odds.iter().map(|v| v.exp()).collect::<Vec<_>>()));
    }
  `), r`
    exp(x): 0.135, 0.368, 1.000, 2.718, 7.389
  `, "The exponential is the inverse of the log: it turns additive log-scale values (log-odds, log-prices) back into multiplicative quantities and stretches left-skewed data."),

  "quantile-transformation": ex(join(PY_SHOW, r`
    x = [3.0, 100.0, 7.0, 1.0, 50.0]
    order = sorted(range(len(x)), key=lambda i: x[i])
    u = [0.0] * len(x)
    for rank, i in enumerate(order):
        u[i] = (rank + 0.5) / len(x)           # empirical CDF value in (0, 1)
    print("uniform:", show(u))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [3.0, 100.0, 7.0, 1.0, 50.0_f64];
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let mut u = vec![0.0; x.len()];
        for (rank, &i) in order.iter().enumerate() {
            u[i] = (rank as f64 + 0.5) / x.len() as f64; // empirical CDF value in (0, 1)
        }
        println!("uniform: {}", show(&u));
    }
  `), r`
    uniform: 0.300, 0.900, 0.500, 0.100, 0.700
  `, "Each value is replaced by its position in the empirical distribution, giving evenly spaced values in (0, 1) regardless of the original shape. Applying the inverse normal CDF to these would produce a normal-shaped feature."),

  "rank-transformation": ex(r`
    x = [40.0, 10.0, 30.0, 10.0, 50.0]
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1                  # ties share the average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    print("ranks:", ", ".join(f"{v:.1f}" for v in ranks))
  `, r`
    fn main() {
        let x = [40.0, 10.0, 30.0, 10.0, 50.0_f64];
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let mut ranks = vec![0.0; x.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j + 1 < order.len() && x[order[j + 1]] == x[order[i]] { j += 1; }
            let avg = (i + j) as f64 / 2.0 + 1.0; // ties share the average rank
            for k in i..=j { ranks[order[k]] = avg; }
            i = j + 1;
        }
        println!("ranks: {}", ranks.iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    ranks: 4.0, 1.5, 3.0, 1.5, 5.0
  `, "Values are replaced by their ranks; the two tied 10s share rank 1.5. Ranks ignore distances entirely, which makes them immune to outliers (the basis of Spearman correlation)."),

  binning: ex(r`
    ages = [18, 22, 25, 31, 38, 45, 52, 67]
    lo, hi, k = 18, 68, 5
    width = (hi - lo) / k
    equal_width = [min(int((a - lo) / width), k - 1) for a in ages]
    equal_freq = [i * 4 // len(ages) for i in range(len(ages))]   # ages are sorted: 4 bins of 2
    print("width =", width)
    print("equal-width bins:", equal_width)
    print("equal-frequency :", equal_freq)
  `, r`
    fn main() {
        let ages = [18, 22, 25, 31, 38, 45, 52, 67];
        let (lo, hi, k) = (18.0, 68.0, 5usize);
        let width = (hi - lo) / k as f64;
        let equal_width: Vec<usize> = ages.iter().map(|&a| (((a as f64 - lo) / width) as usize).min(k - 1)).collect();
        let equal_freq: Vec<usize> = (0..ages.len()).map(|i| i * 4 / ages.len()).collect(); // ages are sorted: 4 bins of 2
        println!("width = {:.1}", width);
        println!("equal-width bins: {:?}", equal_width);
        println!("equal-frequency : {:?}", equal_freq);
    }
  `, r`
    width = 10.0
    equal-width bins: [0, 0, 0, 1, 2, 2, 3, 4]
    equal-frequency : [0, 0, 1, 1, 2, 2, 3, 3]
  `, "Equal-width bins split the range into intervals of 10 years, so some bins hold more people than others; equal-frequency (quantile) bins put the same number of people into each bin."),

  "one-hot-encoding": ex(r`
    colours = ["red", "green", "blue", "green"]
    cats = sorted(set(colours))
    print("columns:", cats)
    for c in colours:
        print(f"{c:>5} ->", [1 if c == k else 0 for k in cats])
  `, r`
    fn main() {
        let colours = ["red", "green", "blue", "green"];
        let mut cats: Vec<&str> = colours.to_vec();
        cats.sort();
        cats.dedup();
        println!("columns: [{}]", cats.iter().map(|c| format!("'{}'", c)).collect::<Vec<_>>().join(", "));
        for c in colours.iter() {
            let row: Vec<i32> = cats.iter().map(|k| if c == k { 1 } else { 0 }).collect();
            println!("{:>5} -> {:?}", c, row);
        }
    }
  `, r`
    columns: ['blue', 'green', 'red']
      red -> [0, 0, 1]
    green -> [0, 1, 0]
     blue -> [1, 0, 0]
    green -> [0, 1, 0]
  `, "Each category gets its own 0/1 column, so no false order is implied between colours. The number of columns equals the number of distinct categories."),

  "label-encoding": ex(r`
    sizes = ["M", "S", "L", "M", "XL"]
    classes = sorted(set(sizes))                 # alphabetical, like scikit-learn's LabelEncoder
    code = {c: i for i, c in enumerate(classes)}
    print("mapping:", code)
    print("encoded:", [code[s] for s in sizes])
    order = {"S": 0, "M": 1, "L": 2, "XL": 3}   # explicit order for an ordinal feature
    print("ordinal:", [order[s] for s in sizes])
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let sizes = ["M", "S", "L", "M", "XL"];
        let mut classes: Vec<&str> = sizes.to_vec();
        classes.sort(); // alphabetical, like scikit-learn's LabelEncoder
        classes.dedup();
        let code: BTreeMap<&str, usize> = classes.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let shown: Vec<String> = code.iter().map(|(k, v)| format!("'{}': {}", k, v)).collect();
        println!("mapping: {{{}}}", shown.join(", "));
        println!("encoded: {:?}", sizes.iter().map(|s| code[s]).collect::<Vec<_>>());
        let order = |s: &str| match s { "S" => 0, "M" => 1, "L" => 2, _ => 3 }; // explicit ordinal order
        println!("ordinal: {:?}", sizes.iter().map(|s| order(s)).collect::<Vec<_>>());
    }
  `, r`
    mapping: {'L': 0, 'M': 1, 'S': 2, 'XL': 3}
    encoded: [1, 2, 0, 1, 3]
    ordinal: [1, 0, 2, 1, 3]
  `, "Alphabetical label encoding gives L = 0, M = 1, S = 2, XL = 3 — an order that means nothing for sizes. For ordinal features, supply the real order explicitly (S < M < L < XL)."),

  "target-encoding": ex(r`
    city = ["A", "A", "A", "B", "B", "C"]
    bought = [1, 1, 0, 1, 0, 1]
    prior = sum(bought) / len(bought)
    m = 2.0                                       # smoothing strength
    for c in sorted(set(city)):
        ys = [y for k, y in zip(city, bought) if k == c]
        raw = sum(ys) / len(ys)
        smooth = (sum(ys) + m * prior) / (len(ys) + m)
        print(f"{c}: n={len(ys)} raw={raw:.3f} smoothed={smooth:.3f}")
  `, r`
    fn main() {
        let city = ["A", "A", "A", "B", "B", "C"];
        let bought = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0_f64];
        let prior = bought.iter().sum::<f64>() / bought.len() as f64;
        let m = 2.0; // smoothing strength
        for c in ["A", "B", "C"] {
            let ys: Vec<f64> = city.iter().zip(bought.iter()).filter(|(k, _)| **k == c).map(|(_, y)| *y).collect();
            let n = ys.len() as f64;
            let s: f64 = ys.iter().sum();
            println!("{}: n={} raw={:.3} smoothed={:.3}", c, ys.len(), s / n, (s + m * prior) / (n + m));
        }
    }
  `, r`
    A: n=3 raw=0.667 smoothed=0.667
    B: n=2 raw=0.500 smoothed=0.583
    C: n=1 raw=1.000 smoothed=0.778
  `, "City C has a single row with target 1, so its raw mean of 1.0 is unreliable; smoothing pulls it towards the global mean (0.667). In practice compute these encodings out-of-fold to avoid leaking the target."),

  "frequency-encoding": ex(r`
    browser = ["chrome", "chrome", "safari", "firefox", "chrome", "safari", "edge", "chrome"]
    counts = {}
    for b in browser:
        counts[b] = counts.get(b, 0) + 1
    for b in sorted(counts):
        print(f"{b:<8} count={counts[b]} freq={counts[b] / len(browser):.3f}")
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let browser = ["chrome", "chrome", "safari", "firefox", "chrome", "safari", "edge", "chrome"];
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for b in browser.iter() { *counts.entry(b).or_insert(0) += 1; }
        for (b, c) in counts.iter() {
            println!("{:<8} count={} freq={:.3}", b, c, *c as f64 / browser.len() as f64);
        }
    }
  `, r`
    chrome   count=4 freq=0.500
    edge     count=1 freq=0.125
    firefox  count=1 freq=0.125
    safari   count=2 freq=0.250
  `, "Each category is replaced by how often it occurs. It needs only one column and handles high-cardinality features, but different categories with equal counts become indistinguishable."),

  pca: ex(join(PY_SHOW, r`
    import math
    X = [(2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0), (2.3, 2.7)]
    n = len(X)
    mx, my = sum(p[0] for p in X) / n, sum(p[1] for p in X) / n
    a = sum((p[0] - mx) ** 2 for p in X) / (n - 1)
    b = sum((p[0] - mx) * (p[1] - my) for p in X) / (n - 1)
    c = sum((p[1] - my) ** 2 for p in X) / (n - 1)
    # eigenvalues of the 2x2 covariance [[a, b], [b, c]]
    tr, det = a + c, a * c - b * b
    l1 = tr / 2 + math.sqrt(tr * tr / 4 - det)
    l2 = tr - l1
    v = (b, l1 - a)
    norm = math.hypot(*v)
    v = (v[0] / norm, v[1] / norm)
    print(f"eigenvalues: {l1:.4f}, {l2:.4f}  explained by PC1: {l1 / tr:.1%}")
    print(f"PC1 direction: ({v[0]:.4f}, {v[1]:.4f})")
    print("PC1 scores:", show([(p[0] - mx) * v[0] + (p[1] - my) * v[1] for p in X]))
  `), join(RS_SHOW, r`
    fn main() {
        let x = [(2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0), (2.3, 2.7_f64)];
        let n = x.len() as f64;
        let mx = x.iter().map(|p| p.0).sum::<f64>() / n;
        let my = x.iter().map(|p| p.1).sum::<f64>() / n;
        let a = x.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>() / (n - 1.0);
        let b = x.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / (n - 1.0);
        let c = x.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>() / (n - 1.0);
        // eigenvalues of the 2x2 covariance [[a, b], [b, c]]
        let (tr, det) = (a + c, a * c - b * b);
        let l1 = tr / 2.0 + (tr * tr / 4.0 - det).sqrt();
        let l2 = tr - l1;
        let norm = b.hypot(l1 - a);
        let v = (b / norm, (l1 - a) / norm);
        println!("eigenvalues: {:.4}, {:.4}  explained by PC1: {:.1}%", l1, l2, 100.0 * l1 / tr);
        println!("PC1 direction: ({:.4}, {:.4})", v.0, v.1);
        let scores: Vec<f64> = x.iter().map(|p| (p.0 - mx) * v.0 + (p.1 - my) * v.1).collect();
        println!("PC1 scores: {}", show(&scores));
    }
  `), r`
    eigenvalues: 1.4323, 0.0470  explained by PC1: 96.8%
    PC1 direction: (0.7182, 0.6958)
    PC1 scores: 0.357, -2.262, 0.490, -0.213, 1.206, 0.422
  `, "The first principal component is the eigenvector of the covariance matrix with the largest eigenvalue. Here the two features are strongly correlated, so a single direction explains almost all of the variance and the data can be reduced to one score per point."),

  "kernel-pca": ex(r`
    import math
    X = [0.0, 1.0, 3.0, 4.0]                   # 1-D points: two pairs
    n, gamma = len(X), 0.5
    K = [[math.exp(-gamma * (a - b) ** 2) for b in X] for a in X]
    row = [sum(r) / n for r in K]
    tot = sum(row) / n
    Kc = [[K[i][j] - row[i] - row[j] + tot for j in range(n)] for i in range(n)]   # centre in feature space
    v = [1.0, 0.5, -0.5, -1.0]
    for _ in range(200):                       # power iteration for the top eigenvector
        w = [sum(Kc[i][j] * v[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(x * x for x in w))
        v = [x / nrm for x in w]
    lam = sum(v[i] * sum(Kc[i][j] * v[j] for j in range(n)) for i in range(n))
    print(f"top eigenvalue = {lam:.4f}")
    print("kernel PC1 scores:", ", ".join(f"{math.sqrt(lam) * x:.3f}" for x in v))
  `, r`
    fn main() {
        let x = [0.0, 1.0, 3.0, 4.0_f64]; // 1-D points: two pairs
        let (n, gamma) = (x.len(), 0.5);
        let k: Vec<Vec<f64>> = x.iter().map(|a| x.iter().map(|b| (-gamma * (a - b) * (a - b)).exp()).collect()).collect();
        let row: Vec<f64> = k.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
        let tot = row.iter().sum::<f64>() / n as f64;
        // centre in feature space
        let kc: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| k[i][j] - row[i] - row[j] + tot).collect()).collect();
        let mut v = vec![1.0, 0.5, -0.5, -1.0];
        for _ in 0..200 {
            // power iteration for the top eigenvector
            let w: Vec<f64> = (0..n).map(|i| (0..n).map(|j| kc[i][j] * v[j]).sum()).collect();
            let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            v = w.iter().map(|x| x / nrm).collect();
        }
        let lam: f64 = (0..n).map(|i| v[i] * (0..n).map(|j| kc[i][j] * v[j]).sum::<f64>()).sum();
        println!("top eigenvalue = {:.4}", lam);
        let scores: Vec<String> = v.iter().map(|x| format!("{:.3}", lam.sqrt() * x)).collect();
        println!("kernel PC1 scores: {}", scores.join(", "));
    }
  `, r`
    top eigenvalue = 1.5314
    kernel PC1 scores: 0.653, 0.583, -0.583, -0.653
  `, "Kernel PCA never builds the feature space: it centres the RBF kernel matrix and takes its top eigenvector. The first kernel component gives the pair {0, 1} one sign and the pair {3, 4} the other — the two groups separate."),

  "truncated-svd": ex(r`
    import math
    A = [[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]]   # term-document style matrix
    v = [1.0, 0.0, 0.0]
    for _ in range(100):                       # power iteration on A^T A
        Av = [sum(A[i][j] * v[j] for j in range(3)) for i in range(3)]
        w = [sum(A[i][j] * Av[i] for i in range(3)) for j in range(3)]
        n = math.sqrt(sum(x * x for x in w))
        v = [x / n for x in w]
    Av = [sum(A[i][j] * v[j] for j in range(3)) for i in range(3)]
    s1 = math.sqrt(sum(x * x for x in Av))
    u = [x / s1 for x in Av]
    A1 = [[s1 * u[i] * v[j] for j in range(3)] for i in range(3)]
    err = math.sqrt(sum((A[i][j] - A1[i][j]) ** 2 for i in range(3) for j in range(3)))
    total = math.sqrt(sum(x * x for row in A for x in row))
    print(f"sigma1 = {s1:.4f}")
    print(f"rank-1 error ||A - A1||_F = {err:.4f} of ||A||_F = {total:.4f}")
  `, r`
    fn main() {
        let a = [[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0_f64]]; // term-document style matrix
        let mut v = [1.0, 0.0, 0.0_f64];
        let mv = |v: &[f64; 3]| -> [f64; 3] { let mut r = [0.0; 3]; for i in 0..3 { for j in 0..3 { r[i] += a[i][j] * v[j]; } } r };
        for _ in 0..100 {
            // power iteration on A^T A
            let av = mv(&v);
            let mut w = [0.0; 3];
            for j in 0..3 { for i in 0..3 { w[j] += a[i][j] * av[i]; } }
            let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..3 { v[j] = w[j] / n; }
        }
        let av = mv(&v);
        let s1 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u: Vec<f64> = av.iter().map(|x| x / s1).collect();
        let mut err = 0.0;
        for i in 0..3 { for j in 0..3 { err += (a[i][j] - s1 * u[i] * v[j]).powi(2); } }
        let total = a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        println!("sigma1 = {:.4}", s1);
        println!("rank-1 error ||A - A1||_F = {:.4} of ||A||_F = {:.4}", err.sqrt(), total);
    }
  `, r`
    sigma1 = 4.0000
    rank-1 error ||A - A1||_F = 2.2361 of ||A||_F = 4.5826
  `, "Power iteration finds the largest singular value σ₁ = 4 and its vectors without forming a full SVD. The rank-1 approximation keeps the dominant 'topic'; the error equals √(σ₂² + σ₃²) = √(4 + 1) ≈ 2.236 (Eckart–Young)."),

  "lda-dimensionality": ex(r`
    import math
    c0 = [(1.0, 2.0), (2.0, 3.0), (3.0, 3.0), (2.0, 1.0)]
    c1 = [(6.0, 5.0), (7.0, 8.0), (8.0, 6.0), (7.0, 5.0)]
    mean = lambda c: (sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c))
    m0, m1 = mean(c0), mean(c1)
    S = [[0.0, 0.0], [0.0, 0.0]]
    for c, m in ((c0, m0), (c1, m1)):            # within-class scatter
        for x, y in c:
            dx, dy = x - m[0], y - m[1]
            S[0][0] += dx * dx; S[0][1] += dx * dy; S[1][0] += dx * dy; S[1][1] += dy * dy
    det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
    inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
    d = (m1[0] - m0[0], m1[1] - m0[1])
    w = (inv[0][0] * d[0] + inv[0][1] * d[1], inv[1][0] * d[0] + inv[1][1] * d[1])
    n = math.hypot(*w)
    w = (w[0] / n, w[1] / n)
    print(f"w = ({w[0]:.4f}, {w[1]:.4f})")
    print("class 0 projections:", ", ".join(f"{x * w[0] + y * w[1]:.2f}" for x, y in c0))
    print("class 1 projections:", ", ".join(f"{x * w[0] + y * w[1]:.2f}" for x, y in c1))
  `, r`
    fn main() {
        let c0 = [(1.0, 2.0), (2.0, 3.0), (3.0, 3.0), (2.0, 1.0_f64)];
        let c1 = [(6.0, 5.0), (7.0, 8.0), (8.0, 6.0), (7.0, 5.0_f64)];
        let mean = |c: &[(f64, f64)]| (c.iter().map(|p| p.0).sum::<f64>() / c.len() as f64, c.iter().map(|p| p.1).sum::<f64>() / c.len() as f64);
        let (m0, m1) = (mean(&c0), mean(&c1));
        let mut s = [[0.0; 2]; 2];
        for (c, m) in [(&c0, m0), (&c1, m1)] {
            // within-class scatter
            for &(x, y) in c.iter() {
                let (dx, dy) = (x - m.0, y - m.1);
                s[0][0] += dx * dx; s[0][1] += dx * dy; s[1][0] += dx * dy; s[1][1] += dy * dy;
            }
        }
        let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        let inv = [[s[1][1] / det, -s[0][1] / det], [-s[1][0] / det, s[0][0] / det]];
        let d = (m1.0 - m0.0, m1.1 - m0.1);
        let w = (inv[0][0] * d.0 + inv[0][1] * d.1, inv[1][0] * d.0 + inv[1][1] * d.1);
        let n = w.0.hypot(w.1);
        let w = (w.0 / n, w.1 / n);
        println!("w = ({:.4}, {:.4})", w.0, w.1);
        let pr = |c: &[(f64, f64)]| c.iter().map(|p| format!("{:.2}", p.0 * w.0 + p.1 * w.1)).collect::<Vec<_>>().join(", ");
        println!("class 0 projections: {}", pr(&c0));
        println!("class 1 projections: {}", pr(&c1));
    }
  `, r`
    w = (0.9906, 0.1366)
    class 0 projections: 1.26, 2.39, 3.38, 2.12
    class 1 projections: 6.63, 8.03, 8.74, 7.62
  `, "Fisher's direction w ∝ S_W⁻¹(μ₁ − μ₀) projects each 2-D point to one number. The two classes' projections do not overlap at all, so a single threshold separates them."),

  "t-sne": ex(r`
    import math
    # squared distances from point i to its neighbours
    d2 = [1.0, 1.2, 4.0, 9.0, 16.0]
    target = math.log(3.0)                  # perplexity 3 -> entropy ln 3 (nats)
    lo, hi = 1e-6, 1e6
    for _ in range(100):                    # binary search for the precision beta = 1 / (2 sigma^2)
        beta = math.sqrt(lo * hi)
        p = [math.exp(-beta * d) for d in d2]
        z = sum(p)
        p = [x / z for x in p]
        H = -sum(x * math.log(x) for x in p if x > 0)
        if H > target: lo = beta
        else: hi = beta
    print(f"beta = {beta:.4f}, entropy = {H:.4f}, perplexity = {math.exp(H):.3f}")
    print("p_j|i:", ", ".join(f"{x:.3f}" for x in p))
  `, r`
    fn main() {
        // squared distances from point i to its neighbours
        let d2 = [1.0, 1.2, 4.0, 9.0, 16.0_f64];
        let target = 3.0_f64.ln(); // perplexity 3 -> entropy ln 3 (nats)
        let (mut lo, mut hi) = (1e-6_f64, 1e6_f64);
        let (mut beta, mut h, mut p) = (1.0, 0.0, vec![0.0; d2.len()]);
        for _ in 0..100 {
            // binary search for the precision beta = 1 / (2 sigma^2)
            beta = (lo * hi).sqrt();
            p = d2.iter().map(|d| (-beta * d).exp()).collect();
            let z: f64 = p.iter().sum();
            for x in p.iter_mut() { *x /= z; }
            h = -p.iter().filter(|&&x| x > 0.0).map(|x| x * x.ln()).sum::<f64>();
            if h > target { lo = beta; } else { hi = beta; }
        }
        println!("beta = {:.4}, entropy = {:.4}, perplexity = {:.3}", beta, h, h.exp());
        println!("p_j|i: {}", p.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    beta = 0.3739, entropy = 1.0986, perplexity = 3.000
    p_j|i: 0.433, 0.402, 0.141, 0.022, 0.002
  `, "t-SNE gives every point its own Gaussian width, found by binary search so that the neighbour distribution has the requested perplexity (here 3 — roughly 'three effective neighbours'). The two closest points get most of the probability."),

  umap: ex(r`
    import math
    dists = [0.5, 0.8, 1.1, 2.0]               # distances to the k = 4 nearest neighbours
    k = len(dists)
    rho = dists[0]                              # distance to the nearest neighbour
    lo, hi = 1e-4, 100.0
    for _ in range(100):                        # find sigma with sum exp(-(d - rho)/sigma) = log2(k)
        sigma = (lo + hi) / 2
        s = sum(math.exp(-(d - rho) / sigma) for d in dists)
        if s > math.log2(k): hi = sigma
        else: lo = sigma
    w = [math.exp(-(d - rho) / sigma) for d in dists]
    print(f"rho = {rho:.2f}, sigma = {sigma:.4f}")
    print("edge weights:", ", ".join(f"{x:.3f}" for x in w))
  `, r`
    fn main() {
        let dists = [0.5, 0.8, 1.1, 2.0_f64]; // distances to the k = 4 nearest neighbours
        let k = dists.len() as f64;
        let rho = dists[0]; // distance to the nearest neighbour
        let (mut lo, mut hi, mut sigma) = (1e-4_f64, 100.0_f64, 1.0_f64);
        for _ in 0..100 {
            // find sigma with sum exp(-(d - rho)/sigma) = log2(k)
            sigma = (lo + hi) / 2.0;
            let s: f64 = dists.iter().map(|d| (-(d - rho) / sigma).exp()).sum();
            if s > k.log2() { hi = sigma; } else { lo = sigma; }
        }
        let w: Vec<String> = dists.iter().map(|d| format!("{:.3}", (-(d - rho) / sigma).exp())).collect();
        println!("rho = {:.2}, sigma = {:.4}", rho, sigma);
        println!("edge weights: {}", w.join(", "));
    }
  `, r`
    rho = 0.50, sigma = 0.5623
    edge weights: 1.000, 0.587, 0.344, 0.069
  `, "UMAP's local edge weights: the nearest neighbour always gets weight 1 (distance minus ρ is 0), and σ is chosen so the weights sum to log₂ k = 2. These fuzzy weights form the graph that the 2-D layout then tries to reproduce."),
};

export const featureEngineeringCode: Record<string, CodeExample> = {
  "polynomial-features": ex(r`
    x1, x2 = 2.0, 3.0
    names = ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
    values = [1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2]
    for n, v in zip(names, values):
        print(f"{n:<6} = {v:.1f}")
  `, r`
    fn main() {
        let (x1, x2) = (2.0_f64, 3.0_f64);
        let names = ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"];
        let values = [1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2];
        for (n, v) in names.iter().zip(values.iter()) {
            println!("{:<6} = {:.1}", n, v);
        }
    }
  `, r`
    1      = 1.0
    x1     = 2.0
    x2     = 3.0
    x1^2   = 4.0
    x1*x2  = 6.0
    x2^2   = 9.0
  `, "A degree-2 expansion of two inputs gives 6 features, including the interaction x₁·x₂. A linear model on these features can fit curved surfaces; the number of features grows quickly with degree."),

  "cyclical-encoding": ex(r`
    import math
    def enc(hour):
        a = 2 * math.pi * hour / 24
        return math.sin(a), math.cos(a)
    for h in [0, 6, 23]:
        s, c = enc(h)
        print(f"hour {h:2d}: sin={s:+.3f} cos={c:+.3f}")
    (s1, c1), (s2, c2) = enc(23), enc(1)
    print(f"distance 23h-1h: raw={abs(23 - 1)} encoded={math.hypot(s1 - s2, c1 - c2):.3f}")
  `, r`
    fn enc(hour: f64) -> (f64, f64) {
        let a = 2.0 * std::f64::consts::PI * hour / 24.0;
        (a.sin(), a.cos())
    }

    fn main() {
        for h in [0, 6, 23] {
            let (s, c) = enc(h as f64);
            println!("hour {:2}: sin={:+.3} cos={:+.3}", h, s, c);
        }
        let ((s1, c1), (s2, c2)) = (enc(23.0), enc(1.0));
        println!("distance 23h-1h: raw={} encoded={:.3}", 23 - 1, (s1 - s2).hypot(c1 - c2));
    }
  `, r`
    hour  0: sin=+0.000 cos=+1.000
    hour  6: sin=+1.000 cos=+0.000
    hour 23: sin=-0.259 cos=+0.966
    distance 23h-1h: raw=22 encoded=0.518
  `, "Mapping the hour onto a circle makes 23:00 and 01:00 neighbours (distance 0.518) instead of 22 units apart, so models see the wrap-around of daily cycles."),

  "aggregation-features": ex(r`
    tx = [("ana", 12.0), ("ben", 40.0), ("ana", 8.0), ("cara", 95.0), ("ben", 22.0), ("ana", 30.0)]
    groups = {}
    for who, amt in tx:
        groups.setdefault(who, []).append(amt)
    for who in sorted(groups):
        a = groups[who]
        print(f"{who:<5} count={len(a)} sum={sum(a):.1f} mean={sum(a) / len(a):.2f} max={max(a):.1f}")
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let tx = [("ana", 12.0), ("ben", 40.0), ("ana", 8.0), ("cara", 95.0), ("ben", 22.0), ("ana", 30.0_f64)];
        let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for (who, amt) in tx.iter() { groups.entry(who).or_default().push(*amt); }
        for (who, a) in groups.iter() {
            let s: f64 = a.iter().sum();
            let m = a.iter().cloned().fold(f64::MIN, f64::max);
            println!("{:<5} count={} sum={:.1} mean={:.2} max={:.1}", who, a.len(), s, s / a.len() as f64, m);
        }
    }
  `, r`
    ana   count=3 sum=50.0 mean=16.67 max=30.0
    ben   count=2 sum=62.0 mean=31.00 max=40.0
    cara  count=1 sum=95.0 mean=95.00 max=95.0
  `, "Six transaction rows become one feature row per customer. These group-by statistics are what a per-customer model (churn, fraud) actually consumes."),

  "lag-features": ex(r`
    sales = [10.0, 12.0, 14.0, 13.0, 15.0, 17.0]
    print("t   y     lag1  roll2(past)")
    for t, y in enumerate(sales):
        lag1 = f"{sales[t - 1]:.1f}" if t >= 1 else "NaN"
        roll = f"{(sales[t - 1] + sales[t - 2]) / 2:.1f}" if t >= 2 else "NaN"
        print(f"{t:<2}  {y:.1f}  {lag1:>4}  {roll:>4}")
  `, r`
    fn main() {
        let sales = [10.0, 12.0, 14.0, 13.0, 15.0, 17.0_f64];
        println!("t   y     lag1  roll2(past)");
        for (t, y) in sales.iter().enumerate() {
            let lag1 = if t >= 1 { format!("{:.1}", sales[t - 1]) } else { "NaN".to_string() };
            let roll = if t >= 2 { format!("{:.1}", (sales[t - 1] + sales[t - 2]) / 2.0) } else { "NaN".to_string() };
            println!("{:<2}  {:.1}  {:>4}  {:>4}", t, y, lag1, roll);
        }
    }
  `, r`
    t   y     lag1  roll2(past)
    0   10.0   NaN   NaN
    1   12.0  10.0   NaN
    2   14.0  12.0  11.0
    3   13.0  14.0  13.0
    4   15.0  13.0  13.5
    5   17.0  15.0  14.0
  `, "lag1 is yesterday's value and roll2 the mean of the two previous days — both use only the past, so they are safe to compute at prediction time. The first rows have no history (NaN)."),

  "domain-features": ex(r`
    people = [("Asha", 60.0, 1.65), ("Bo", 90.0, 1.80), ("Chen", 50.0, 1.58)]
    for name, kg, m in people:
        bmi = kg / m ** 2                        # expert knowledge: weight relative to height squared
        print(f"{name:<5} BMI={bmi:.1f}")
  `, r`
    fn main() {
        let people = [("Asha", 60.0, 1.65), ("Bo", 90.0, 1.80), ("Chen", 50.0, 1.58_f64)];
        for (name, kg, m) in people.iter() {
            let bmi = kg / (m * m); // expert knowledge: weight relative to height squared
            println!("{:<5} BMI={:.1}", name, bmi);
        }
    }
  `, r`
    Asha  BMI=22.0
    Bo    BMI=27.8
    Chen  BMI=20.0
  `, "Body-mass index combines two raw measurements into one medically meaningful number — a relationship a linear model could not build from weight and height on its own."),

  "embedded-selection": ex(r`
    # Lasso by coordinate descent on toy data: y depends on x1 and x2 only
    X = [[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [1.0, 1.0, 0.5], [-1.0, -1.0, -0.5]]
    y = [2.0, -2.0, 4.0, -4.0]                 # y = 3*x1 + 1*x2 (x3 = x1 / 2 is redundant)
    n, p, lam = len(X), 3, 0.5
    w = [0.0] * p
    for _ in range(200):
        for j in range(p):
            r = [y[i] - sum(X[i][k] * w[k] for k in range(p) if k != j) for i in range(n)]
            rho = sum(X[i][j] * r[i] for i in range(n)) / n
            z = sum(X[i][j] ** 2 for i in range(n)) / n
            w[j] = (max(abs(rho) - lam, 0.0) * (1 if rho > 0 else -1)) / z   # soft threshold
    print("weights:", ", ".join(f"{v:.3f}" for v in w))
    print("selected:", [f"x{j + 1}" for j in range(p) if abs(w[j]) > 1e-9])
  `, r`
    fn main() {
        // Lasso by coordinate descent on toy data: y depends on x1 and x2 only
        let x = [[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [1.0, 1.0, 0.5], [-1.0, -1.0, -0.5_f64]];
        let y = [2.0, -2.0, 4.0, -4.0_f64]; // y = 3*x1 + 1*x2 (x3 = x1 / 2 is redundant)
        let (n, p, lam) = (x.len(), 3, 0.5);
        let mut w = [0.0_f64; 3];
        for _ in 0..200 {
            for j in 0..p {
                let r: Vec<f64> = (0..n).map(|i| y[i] - (0..p).filter(|&k| k != j).map(|k| x[i][k] * w[k]).sum::<f64>()).collect();
                let rho = (0..n).map(|i| x[i][j] * r[i]).sum::<f64>() / n as f64;
                let z = (0..n).map(|i| x[i][j] * x[i][j]).sum::<f64>() / n as f64;
                let sign = if rho > 0.0 { 1.0 } else { -1.0 };
                w[j] = (rho.abs() - lam).max(0.0) * sign / z; // soft threshold
            }
        }
        println!("weights: {}", w.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "));
        let sel: Vec<String> = (0..p).filter(|&j| w[j].abs() > 1e-9).map(|j| format!("'x{}'", j + 1)).collect();
        println!("selected: [{}]", sel.join(", "));
    }
  `, r`
    weights: 2.500, 0.500, 0.000
    selected: ['x1', 'x2']
  `, "The L1 penalty is applied while fitting, so selection is embedded in training: coordinate descent soft-thresholds each weight. The redundant x3 ends up exactly at zero, and the kept weights are shrunk (3 → 2.5, 1 → 0.5) — the price of the penalty."),

  "filter-methods": ex(r`
    import math
    cols = {"constant": [1, 1, 1, 1, 1, 1], "age": [23, 35, 41, 29, 52, 47], "noise": [3, 1, 4, 1, 5, 9]}
    target = [0, 1, 1, 0, 1, 1]
    def var(v):
        m = sum(v) / len(v)
        return sum((x - m) ** 2 for x in v) / len(v)
    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
        return cov / (sa * sb) if sa > 0 and sb > 0 else 0.0
    for name, v in cols.items():
        keep = var(v) > 0.1 and abs(corr(v, target)) > 0.3
        print(f"{name:<8} var={var(v):7.2f} |r|={abs(corr(v, target)):.3f} keep={keep}")
  `, r`
    fn var(v: &[f64]) -> f64 {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let sa = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let sb = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
        if sa > 0.0 && sb > 0.0 { cov / (sa * sb) } else { 0.0 }
    }

    fn main() {
        let cols: [(&str, [f64; 6]); 3] = [("constant", [1.0; 6]), ("age", [23.0, 35.0, 41.0, 29.0, 52.0, 47.0]), ("noise", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0])];
        let target = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        for (name, v) in cols.iter() {
            let keep = var(v) > 0.1 && corr(v, &target).abs() > 0.3;
            let k = if keep { "True" } else { "False" };
            println!("{:<8} var={:7.2} |r|={:.3} keep={}", name, var(v), corr(v, &target).abs(), k);
        }
    }
  `, r`
    constant var=   0.00 |r|=0.000 keep=False
    age      var= 100.14 |r|=0.836 keep=True
    noise    var=   7.47 |r|=0.474 keep=True
  `, "The constant column fails the variance threshold and age is rightly kept. But the random noise column also passes (|r| = 0.474): with only 6 rows, chance correlations are large. Filters judge each feature alone, so they need enough data and should be validated inside cross-validation."),

  "mutual-information": ex(r`
    import math
    x = [-2, -1, 0, 1, 2] * 4
    y = [v * v for v in x]                     # y is a function of x, but not a linear one
    n = len(x)
    def entropy(vals):
        counts = {}
        for v in vals: counts[v] = counts.get(v, 0) + 1
        return -sum(c / n * math.log(c / n) for c in counts.values())
    mi = entropy(x) + entropy(y) - entropy(list(zip(x, y)))
    mx, my = sum(x) / n, sum(y) / n
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    print(f"Pearson r = {r + 0.0:.3f}")
    print(f"I(X;Y) = {mi:.3f} nats, H(Y) = {entropy(y):.3f}")
  `, r`
    use std::collections::HashMap;

    fn entropy(vals: &[(i32, i32)]) -> f64 {
        let n = vals.len() as f64;
        let mut counts: HashMap<(i32, i32), usize> = HashMap::new();
        for v in vals { *counts.entry(*v).or_insert(0) += 1; }
        -counts.values().map(|&c| { let p = c as f64 / n; p * p.ln() }).sum::<f64>()
    }

    fn main() {
        let x: Vec<i32> = (0..20).map(|i| [-2, -1, 0, 1, 2][i % 5]).collect();
        let y: Vec<i32> = x.iter().map(|v| v * v).collect(); // y is a function of x, but not a linear one
        let hx = entropy(&x.iter().map(|&v| (v, 0)).collect::<Vec<_>>());
        let hy = entropy(&y.iter().map(|&v| (v, 0)).collect::<Vec<_>>());
        let hxy = entropy(&x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect::<Vec<_>>());
        let n = x.len() as f64;
        let mx = x.iter().sum::<i32>() as f64 / n;
        let my = y.iter().sum::<i32>() as f64 / n;
        let cov: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a as f64 - mx) * (b as f64 - my)).sum();
        let sx: f64 = x.iter().map(|&a| (a as f64 - mx).powi(2)).sum();
        let sy: f64 = y.iter().map(|&b| (b as f64 - my).powi(2)).sum();
        println!("Pearson r = {:.3}", (cov / (sx * sy).sqrt()).abs());
        println!("I(X;Y) = {:.3} nats, H(Y) = {:.3}", hx + hy - hxy, hy);
    }
  `, r`
    Pearson r = 0.000
    I(X;Y) = 1.055 nats, H(Y) = 1.055
  `, "Correlation is exactly 0 because the U-shaped relation is symmetric, yet mutual information equals the full entropy of Y: knowing x removes all uncertainty about y."),

  "wrapper-methods": ex(r`
    # forward selection: add the feature that most reduces training MSE of a 1-feature-at-a-time fit
    X = {"x1": [1.0, 2.0, 3.0, 4.0, 5.0], "x2": [2.0, 1.0, 4.0, 3.0, 5.0], "x3": [5.0, 3.0, 4.0, 1.0, 2.0]}
    y = [3.0, 5.0, 7.0, 9.0, 11.0]               # y = 2*x1 + 1
    def mse_simple(xs):
        n = len(xs); mx = sum(xs) / n; my = sum(y) / n
        b = sum((a - mx) * (t - my) for a, t in zip(xs, y)) / sum((a - mx) ** 2 for a in xs)
        a0 = my - b * mx
        return sum((t - a0 - b * a) ** 2 for a, t in zip(xs, y)) / n
    scores = {k: mse_simple(v) for k, v in X.items()}
    for k in sorted(scores, key=scores.get):
        print(f"{k}: MSE={scores[k]:.3f}")
    print("first feature chosen:", min(scores, key=scores.get))
  `, r`
    fn main() {
        // forward selection: add the feature that most reduces training MSE of a 1-feature-at-a-time fit
        let feats = [("x1", [1.0, 2.0, 3.0, 4.0, 5.0]), ("x2", [2.0, 1.0, 4.0, 3.0, 5.0]), ("x3", [5.0, 3.0, 4.0, 1.0, 2.0_f64])];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0_f64]; // y = 2*x1 + 1
        let mse_simple = |xs: &[f64]| {
            let n = xs.len() as f64;
            let mx = xs.iter().sum::<f64>() / n;
            let my = y.iter().sum::<f64>() / n;
            let b = xs.iter().zip(y.iter()).map(|(a, t)| (a - mx) * (t - my)).sum::<f64>() / xs.iter().map(|a| (a - mx).powi(2)).sum::<f64>();
            let a0 = my - b * mx;
            xs.iter().zip(y.iter()).map(|(a, t)| (t - a0 - b * a).powi(2)).sum::<f64>() / n
        };
        let mut scores: Vec<(&str, f64)> = feats.iter().map(|(k, v)| (*k, mse_simple(v))).collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (k, s) in scores.iter() { println!("{}: MSE={:.3}", k, s); }
        println!("first feature chosen: {}", scores[0].0);
    }
  `, r`
    x1: MSE=0.000
    x2: MSE=2.880
    x3: MSE=2.880
    first feature chosen: x1
  `, "A wrapper method judges each candidate feature by actually fitting the model: x1 explains y perfectly (MSE 0), so forward selection picks it first. Repeating with the remaining features builds the subset step by step."),
};
