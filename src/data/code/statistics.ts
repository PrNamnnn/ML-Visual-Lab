import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

/* Shared helpers repeated inside snippets so each program is self-contained. */
const PY_LCG = r`
    seed = 42
    def rand():                      # tiny linear congruential generator (same in the Rust version)
        global seed
        seed = (seed * 1103515245 + 12345) % 2**31
        return seed / 2**31`;
const RS_LCG = r`
    static mut SEED: u64 = 42;
    fn rand() -> f64 {
        // tiny linear congruential generator (same in the Python version)
        unsafe {
            SEED = (SEED * 1103515245 + 12345) % (1u64 << 31);
            SEED as f64 / (1u64 << 31) as f64
        }
    }`;
const PY_PHI = r`
    import math
    def phi(z):                      # standard normal CDF by Simpson integration (same method as Rust)
        n, a = 2000, -8.0
        h = (z - a) / n
        f = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        s = f(a) + f(z) + sum(f(a + i * h) * (4 if i % 2 else 2) for i in range(1, n))
        return s * h / 3`;
const RS_PHI = r`
    fn phi(z: f64) -> f64 {
        // standard normal CDF by Simpson integration (same method as Python)
        let (n, a) = (2000, -8.0_f64);
        let h = (z - a) / n as f64;
        let f = |x: f64| (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let mut s = f(a) + f(z);
        for i in 1..n { s += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }; }
        s * h / 3.0
    }`;

const join = (...parts: string[]) => parts.join("\n");

export const statisticsCode: Record<string, CodeExample> = {
  "sample-spaces": ex(r`
    omega = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    A = [o for o in omega if o[0] + o[1] == 7]
    print("|omega| =", len(omega))
    print("A =", A)
    print(f"P(sum = 7) = {len(A)}/{len(omega)} = {len(A) / len(omega):.4f}")
  `, r`
    fn main() {
        let mut omega = vec![];
        for a in 1..=6 { for b in 1..=6 { omega.push((a, b)); } }
        let a: Vec<(i32, i32)> = omega.iter().cloned().filter(|o| o.0 + o.1 == 7).collect();
        println!("|omega| = {}", omega.len());
        println!("A = {:?}", a);
        println!("P(sum = 7) = {}/{} = {:.4}", a.len(), omega.len(), a.len() as f64 / omega.len() as f64);
    }
  `, r`
    |omega| = 36
    A = [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]
    P(sum = 7) = 6/36 = 0.1667
  `, "List the sample space Ω (36 ordered pairs), keep the outcomes in the event, and divide counts — valid because every outcome is equally likely."),

  "probability-axioms": ex(r`
    omega = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    P = lambda ev: sum(1 for o in omega if ev(o)) / len(omega)
    A = lambda o: o[0] + o[1] > 8
    B = lambda o: o[0] == o[1]
    print(f"P(A) = {P(A):.4f}, P(B) = {P(B):.4f}")
    print(f"P(A or B) direct          = {P(lambda o: A(o) or B(o)):.4f}")
    print(f"P(A) + P(B) - P(A and B)  = {P(A) + P(B) - P(lambda o: A(o) and B(o)):.4f}")
    print(f"P(not A) = 1 - P(A)       = {1 - P(A):.4f}")
  `, r`
    fn main() {
        let mut omega = vec![];
        for a in 1..=6 { for b in 1..=6 { omega.push((a, b)); } }
        let p = |ev: &dyn Fn(&(i32, i32)) -> bool| omega.iter().filter(|o| ev(o)).count() as f64 / omega.len() as f64;
        let a = |o: &(i32, i32)| o.0 + o.1 > 8;
        let b = |o: &(i32, i32)| o.0 == o.1;
        println!("P(A) = {:.4}, P(B) = {:.4}", p(&a), p(&b));
        println!("P(A or B) direct          = {:.4}", p(&|o| a(o) || b(o)));
        println!("P(A) + P(B) - P(A and B)  = {:.4}", p(&a) + p(&b) - p(&|o| a(o) && b(o)));
        println!("P(not A) = 1 - P(A)       = {:.4}", 1.0 - p(&a));
    }
  `, r`
    P(A) = 0.2778, P(B) = 0.1667
    P(A or B) direct          = 0.3889
    P(A) + P(B) - P(A and B)  = 0.3889
    P(not A) = 1 - P(A)       = 0.7222
  `, "Counting the union directly agrees with inclusion–exclusion, and the complement rule follows from P(Ω) = 1 — both consequences of Kolmogorov's axioms."),

  "conditional-probability": ex(r`
    omega = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    B = [o for o in omega if o[0] == 6]                 # condition: first die is 6
    A_and_B = [o for o in B if o[0] + o[1] >= 10]       # sum at least 10
    print(f"P(sum >= 10 | first = 6) = {len(A_and_B)}/{len(B)} = {len(A_and_B) / len(B):.4f}")
    A = [o for o in omega if o[0] + o[1] >= 10]
    print(f"P(sum >= 10)             = {len(A)}/36 = {len(A) / 36:.4f}")
  `, r`
    fn main() {
        let mut omega = vec![];
        for a in 1..=6 { for b in 1..=6 { omega.push((a, b)); } }
        let b: Vec<&(i32, i32)> = omega.iter().filter(|o| o.0 == 6).collect(); // condition: first die is 6
        let ab = b.iter().filter(|o| o.0 + o.1 >= 10).count(); // sum at least 10
        println!("P(sum >= 10 | first = 6) = {}/{} = {:.4}", ab, b.len(), ab as f64 / b.len() as f64);
        let a = omega.iter().filter(|o| o.0 + o.1 >= 10).count();
        println!("P(sum >= 10)             = {}/36 = {:.4}", a, a as f64 / 36.0);
    }
  `, r`
    P(sum >= 10 | first = 6) = 3/6 = 0.5000
    P(sum >= 10)             = 6/36 = 0.1667
  `, "Conditioning restricts the sample space to B. Knowing the first die is 6 triples the chance of a sum ≥ 10, so the events are dependent."),

  "bayes-theorem": ex(r`
    prior = 0.01          # P(spam word appears in a message that is spam)... here: P(disease)
    sens, spec = 0.99, 0.95
    p_pos = sens * prior + (1 - spec) * (1 - prior)          # total probability
    posterior = sens * prior / p_pos                         # Bayes' theorem
    print(f"P(+) = {p_pos:.4f}")
    print(f"P(disease | +) = {posterior:.4f}")
  `, r`
    fn main() {
        let prior = 0.01_f64; // P(disease)
        let (sens, spec) = (0.99_f64, 0.95_f64);
        let p_pos = sens * prior + (1.0 - spec) * (1.0 - prior); // total probability
        let posterior = sens * prior / p_pos; // Bayes' theorem
        println!("P(+) = {:.4}", p_pos);
        println!("P(disease | +) = {:.4}", posterior);
    }
  `, r`
    P(+) = 0.0594
    P(disease | +) = 0.1667
  `, "Even with a 99%-sensitive test, a 1% base rate means only 1 in 6 positives is truly ill: P(D | +) = P(+ | D)P(D)/P(+)."),

  "conditional-independence": ex(r`
    pr, pu = 0.3, {1: 0.9, 0: 0.1}      # P(rain), P(umbrella | rain)
    pw = {1: 0.95, 0: 0.2}              # P(wet | rain)
    P = lambda r_: pr if r_ else 1 - pr
    pU = sum(P(r_) * pu[r_] for r_ in (0, 1))
    pW = sum(P(r_) * pw[r_] for r_ in (0, 1))
    pUW = sum(P(r_) * pu[r_] * pw[r_] for r_ in (0, 1))
    print(f"P(U,W) = {pUW:.4f}  vs  P(U)P(W) = {pU * pW:.4f}  -> dependent")
    print(f"P(W | U) = {pUW / pU:.4f}  vs  P(W) = {pW:.4f}")
    print(f"given rain: P(W | U, R) = P(W | R) = {pw[1]:.4f}")
  `, r`
    fn main() {
        let pr = 0.3_f64; // P(rain)
        let pu = [0.1_f64, 0.9]; // P(umbrella | rain=0/1)
        let pw = [0.2_f64, 0.95]; // P(wet | rain=0/1)
        let p = |r: usize| if r == 1 { pr } else { 1.0 - pr };
        let pu_m: f64 = (0..2).map(|r| p(r) * pu[r]).sum();
        let pw_m: f64 = (0..2).map(|r| p(r) * pw[r]).sum();
        let puw: f64 = (0..2).map(|r| p(r) * pu[r] * pw[r]).sum();
        println!("P(U,W) = {:.4}  vs  P(U)P(W) = {:.4}  -> dependent", puw, pu_m * pw_m);
        println!("P(W | U) = {:.4}  vs  P(W) = {:.4}", puw / pu_m, pw_m);
        println!("given rain: P(W | U, R) = P(W | R) = {:.4}", pw[1]);
    }
  `, r`
    P(U,W) = 0.2705  vs  P(U)P(W) = 0.1445  -> dependent
    P(W | U) = 0.7956  vs  P(W) = 0.4250
    given rain: P(W | U, R) = P(W | R) = 0.9500
  `, "Umbrella and wet ground are dependent overall (both caused by rain) but independent once rain is known — the joint factorises as P(R)P(U|R)P(W|R)."),

  "random-variables": ex(r`
    # X = number of heads in 3 fair coin tosses
    from itertools import product
    counts = {}
    for toss in product("HT", repeat=3):
        k = toss.count("H")
        counts[k] = counts.get(k, 0) + 1
    pmf = {k: counts[k] / 8 for k in sorted(counts)}
    cdf, run = {}, 0.0
    for k, p in pmf.items():
        run += p
        cdf[k] = run
    print("PMF:", pmf)
    print("CDF:", cdf)
  `, r`
    fn main() {
        // X = number of heads in 3 fair coin tosses
        let mut counts = [0; 4];
        for mask in 0..8 {
            let k = (0..3).filter(|b| mask & (1 << b) != 0).count();
            counts[k] += 1;
        }
        // print floats like Python does (1.0 rather than 1)
        let show = |v: f64| if v.fract() == 0.0 { format!("{:.1}", v) } else { format!("{}", v) };
        let pmf: Vec<String> = (0..4).map(|k| format!("{}: {}", k, show(counts[k] as f64 / 8.0))).collect();
        let mut run = 0.0;
        let cdf: Vec<String> = (0..4).map(|k| { run += counts[k] as f64 / 8.0; format!("{}: {}", k, show(run)) }).collect();
        println!("PMF: {{{}}}", pmf.join(", "));
        println!("CDF: {{{}}}", cdf.join(", "));
    }
  `, r`
    PMF: {0: 0.125, 1: 0.375, 2: 0.375, 3: 0.125}
    CDF: {0: 0.125, 1: 0.5, 2: 0.875, 3: 1.0}
  `, "A random variable maps each outcome (e.g. HTH) to a number (2 heads). The PMF gives P(X = k); the CDF accumulates it, P(X ≤ k), ending at 1."),

  "expected-value-variance": ex(r`
    marks = [55, 62, 68, 70, 71, 74, 78, 81, 88, 98]
    n = len(marks)
    mean = sum(marks) / n
    var = sum((x - mean) ** 2 for x in marks) / (n - 1)      # sample variance
    m2 = sum((x - mean) ** 2 for x in marks) / n
    m3 = sum((x - mean) ** 3 for x in marks) / n
    skew = (n * (n - 1)) ** 0.5 / (n - 2) * m3 / m2 ** 1.5
    print(f"mean = {mean:.2f}, variance = {var:.2f}, std = {var ** 0.5:.2f}, skewness = {skew:.3f}")
  `, r`
    fn main() {
        let marks = [55.0, 62.0, 68.0, 70.0, 71.0, 74.0, 78.0, 81.0, 88.0, 98.0_f64];
        let n = marks.len() as f64;
        let mean = marks.iter().sum::<f64>() / n;
        let var = marks.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0); // sample variance
        let m2 = marks.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = marks.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        let skew = (n * (n - 1.0)).sqrt() / (n - 2.0) * m3 / m2.powf(1.5);
        println!("mean = {:.2}, variance = {:.2}, std = {:.2}, skewness = {:.3}", mean, var, var.sqrt(), skew);
    }
  `, r`
    mean = 74.50, variance = 155.61, std = 12.47, skewness = 0.428
  `, "Student marks: the mean is the centre, the standard deviation the typical distance from it, and positive skewness says the high scores stretch further out than the low ones."),

  "law-of-large-numbers": ex(join(PY_LCG, r`
    total = 0.0
    for n in range(1, 100001):
        total += 1 if rand() < 0.5 else 0     # fair coin: heads = 1
        if n in (10, 100, 1000, 10000, 100000):
            print(f"n={n:>6}: fraction of heads = {total / n:.4f}")
  `), join(RS_LCG, r`
    fn main() {
        let mut total = 0.0;
        for n in 1..=100000 {
            total += if rand() < 0.5 { 1.0 } else { 0.0 }; // fair coin: heads = 1
            if [10, 100, 1000, 10000, 100000].contains(&n) {
                println!("n={:>6}: fraction of heads = {:.4}", n, total / n as f64);
            }
        }
    }
  `), r`
    n=    10: fraction of heads = 0.4000
    n=   100: fraction of heads = 0.5500
    n=  1000: fraction of heads = 0.4780
    n= 10000: fraction of heads = 0.5028
    n=100000: fraction of heads = 0.5007
  `, "The running fraction of heads wanders early and settles near the true probability 0.5 as n grows — the law of large numbers."),

  "central-limit-theorem": ex(join(PY_LCG, r`
    import math
    n, reps = 30, 20000
    means = []
    for _ in range(reps):
        sample = [-math.log(1 - rand()) for _ in range(n)]     # exponential(1): skewed
        means.append(sum(sample) / n)
    m = sum(means) / reps
    sd = (sum((x - m) ** 2 for x in means) / (reps - 1)) ** 0.5
    inside = sum(1 for x in means if abs(x - 1) < 1.96 / math.sqrt(n)) / reps
    print(f"mean of sample means = {m:.3f} (theory 1.000)")
    print(f"sd of sample means   = {sd:.3f} (theory {1 / math.sqrt(n):.3f})")
    print(f"within 1.96 SE       = {inside:.3f} (normal theory 0.950)")
  `), join(RS_LCG, r`
    fn main() {
        let (n, reps) = (30, 20000);
        let mut means = vec![];
        for _ in 0..reps {
            let s: f64 = (0..n).map(|_| -(1.0 - rand()).ln()).sum(); // exponential(1): skewed
            means.push(s / n as f64);
        }
        let m = means.iter().sum::<f64>() / reps as f64;
        let sd = (means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (reps - 1) as f64).sqrt();
        let inside = means.iter().filter(|&&x| (x - 1.0).abs() < 1.96 / (n as f64).sqrt()).count() as f64 / reps as f64;
        println!("mean of sample means = {:.3} (theory 1.000)", m);
        println!("sd of sample means   = {:.3} (theory {:.3})", sd, 1.0 / (n as f64).sqrt());
        println!("within 1.96 SE       = {:.3} (normal theory 0.950)", inside);
    }
  `), r`
    mean of sample means = 1.003 (theory 1.000)
    sd of sample means   = 0.184 (theory 0.183)
    within 1.96 SE       = 0.951 (normal theory 0.950)
  `, "Averages of 30 draws from a skewed exponential distribution behave like N(1, 1/30): the right mean, the right spread σ/√n, and about 95% within ±1.96 standard errors."),

  "covariance-correlation": ex(r`
    hours = [1, 2, 3, 4, 5, 6]
    marks = [52, 55, 61, 64, 70, 71]
    n = len(hours)
    mx, my = sum(hours) / n, sum(marks) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(hours, marks)) / (n - 1)
    sx = (sum((x - mx) ** 2 for x in hours) / (n - 1)) ** 0.5
    sy = (sum((y - my) ** 2 for y in marks) / (n - 1)) ** 0.5
    print(f"cov = {cov:.3f}, r = {cov / (sx * sy):.4f}")
  `, r`
    fn main() {
        let hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
        let marks = [52.0, 55.0, 61.0, 64.0, 70.0, 71.0_f64];
        let n = hours.len() as f64;
        let (mx, my) = (hours.iter().sum::<f64>() / n, marks.iter().sum::<f64>() / n);
        let cov = hours.iter().zip(marks.iter()).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / (n - 1.0);
        let sx = (hours.iter().map(|x| (x - mx).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let sy = (marks.iter().map(|y| (y - my).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        println!("cov = {:.3}, r = {:.4}", cov, cov / (sx * sy));
    }
  `, r`
    cov = 14.300, r = 0.9887
  `, "Hours studied vs marks: positive covariance, and dividing by both standard deviations gives r ≈ 0.99 — a strong linear association (which, by itself, says nothing about causation)."),

  "bernoulli-distribution": ex(r`
    p = 0.3                       # probability a visitor clicks
    pmf = {0: 1 - p, 1: p}
    print("P(X=0) =", pmf[0], " P(X=1) =", pmf[1])
    print(f"mean = {p}, variance = {p * (1 - p):.2f}")
  `, r`
    fn main() {
        let p = 0.3_f64; // probability a visitor clicks
        println!("P(X=0) = {}  P(X=1) = {}", 1.0 - p, p);
        println!("mean = {}, variance = {:.2}", p, p * (1.0 - p));
    }
  `, r`
    P(X=0) = 0.7  P(X=1) = 0.3
    mean = 0.3, variance = 0.21
  `, "A single click/no-click trial. The mean is p and the variance p(1−p) is largest at p = 0.5 — maximum uncertainty."),

  "binomial-distribution": ex(r`
    from math import comb
    n, p = 10, 0.5               # 10 coin tosses
    for k in [3, 5, 8]:
        print(f"P(X={k}) = {comb(n, k) * p**k * (1 - p)**(n - k):.4f}")
    print(f"mean = {n * p}, variance = {n * p * (1 - p)}")
  `, r`
    fn comb(n: u64, k: u64) -> u64 { (1..=k).fold(1, |acc, i| acc * (n - k + i) / i) }

    fn main() {
        let (n, p) = (10u64, 0.5_f64); // 10 coin tosses
        for k in [3u64, 5, 8] {
            let pk = comb(n, k) as f64 * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
            println!("P(X={}) = {:.4}", k, pk);
        }
        println!("mean = {:.1}, variance = {:.1}", n as f64 * p, n as f64 * p * (1.0 - p));
    }
  `, r`
    P(X=3) = 0.1172
    P(X=5) = 0.2461
    P(X=8) = 0.0439
    mean = 5.0, variance = 2.5
  `, "C(n, k)pᵏ(1−p)ⁿ⁻ᵏ counts the ways to get k heads times the probability of each way. 5 heads is most likely but happens only ~25% of the time."),

  "geometric-distribution": ex(r`
    p = 0.25                     # chance a sales call succeeds
    for k in [1, 2, 4]:
        print(f"P(first success on call {k}) = {(1 - p) ** (k - 1) * p:.4f}")
    print(f"expected calls = {1 / p}")
  `, r`
    fn main() {
        let p = 0.25_f64; // chance a sales call succeeds
        for k in [1, 2, 4] {
            println!("P(first success on call {}) = {:.4}", k, (1.0 - p).powi(k - 1) * p);
        }
        println!("expected calls = {:.1}", 1.0 / p);
    }
  `, r`
    P(first success on call 1) = 0.2500
    P(first success on call 2) = 0.1875
    P(first success on call 4) = 0.1055
    expected calls = 4.0
  `, "(1−p)ᵏ⁻¹p: k−1 failures, then a success. On average you need 1/p = 4 calls."),

  "poisson-distribution": ex(r`
    import math
    lam = 3.0                    # average customers per minute
    for k in [0, 2, 5]:
        print(f"P(X={k}) = {lam ** k * math.exp(-lam) / math.factorial(k):.4f}")
    print(f"mean = variance = {lam}")
  `, r`
    fn factorial(k: u32) -> f64 { (1..=k).fold(1.0, |a, i| a * i as f64) }

    fn main() {
        let lam = 3.0_f64; // average customers per minute
        for k in [0u32, 2, 5] {
            println!("P(X={}) = {:.4}", k, lam.powi(k as i32) * (-lam).exp() / factorial(k));
        }
        println!("mean = variance = {:.1}", lam);
    }
  `, r`
    P(X=0) = 0.0498
    P(X=2) = 0.2240
    P(X=5) = 0.1008
    mean = variance = 3.0
  `, "λᵏe^{−λ}/k! for counts of independent arrivals. With 3 customers a minute on average, a completely quiet minute happens about 5% of the time."),

  "discrete-uniform-distribution": ex(r`
    a, b = 1, 6                  # a fair die
    n = b - a + 1
    print(f"P(X=k) = 1/{n} = {1 / n:.4f}")
    print(f"mean = {(a + b) / 2}, variance = {(n * n - 1) / 12:.4f}")
  `, r`
    fn main() {
        let (a, b) = (1, 6); // a fair die
        let n = b - a + 1;
        println!("P(X=k) = 1/{} = {:.4}", n, 1.0 / n as f64);
        println!("mean = {:.1}, variance = {:.4}", (a + b) as f64 / 2.0, (n * n - 1) as f64 / 12.0);
    }
  `, r`
    P(X=k) = 1/6 = 0.1667
    mean = 3.5, variance = 2.9167
  `, "Every face is equally likely. The mean (a+b)/2 = 3.5 is not itself a possible roll — expectation is a long-run average."),

  "categorical-distribution": ex(r`
    import math
    logits = [2.0, 1.0, 0.1]                     # e.g. scores for cat / dog / bird
    m = max(logits)
    e = [math.exp(z - m) for z in logits]        # subtract max for stability
    p = [v / sum(e) for v in e]
    print("probabilities:", [round(v, 4) for v in p])
    print(f"sum = {sum(p):.4f}")
  `, r`
    fn main() {
        let logits = [2.0_f64, 1.0, 0.1]; // e.g. scores for cat / dog / bird
        let m = logits.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = logits.iter().map(|z| (z - m).exp()).collect(); // subtract max for stability
        let s: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|v| v / s).collect();
        let shown: Vec<String> = p.iter().map(|v| format!("{}", (v * 1e4).round() / 1e4)).collect();
        println!("probabilities: [{}]", shown.join(", "));
        println!("sum = {:.4}", p.iter().sum::<f64>());
    }
  `, r`
    probabilities: [0.659, 0.2424, 0.0986]
    sum = 1.0000
  `, "Softmax turns scores into a categorical distribution over 3 classes: non-negative probabilities that sum to 1."),

  "multinomial-distribution": ex(r`
    from math import factorial
    counts = [2, 1, 1]           # in 4 draws: 2 red, 1 green, 1 blue
    probs = [0.5, 0.3, 0.2]
    n = sum(counts)
    coef = factorial(n)
    for c in counts:
        coef //= factorial(c)
    p = coef
    for c, q in zip(counts, probs):
        p *= q ** c
    print("multinomial coefficient =", coef)
    print(f"P = {p:.4f}")
  `, r`
    fn factorial(k: u64) -> u64 { (1..=k).product() }

    fn main() {
        let counts = [2u64, 1, 1]; // in 4 draws: 2 red, 1 green, 1 blue
        let probs = [0.5_f64, 0.3, 0.2];
        let n: u64 = counts.iter().sum();
        let mut coef = factorial(n);
        for c in counts { coef /= factorial(c); }
        let mut p = coef as f64;
        for (c, q) in counts.iter().zip(probs.iter()) { p *= q.powi(*c as i32); }
        println!("multinomial coefficient = {}", coef);
        println!("P = {:.4}", p);
    }
  `, r`
    multinomial coefficient = 12
    P = 0.1800
  `, "n!/(x₁!x₂!x₃!) = 12 orderings, each with probability 0.5²·0.3·0.2 = 0.015, so P = 0.18 — the bag-of-words likelihood in multinomial Naive Bayes."),

  "uniform-distribution": ex(r`
    a, b = 2.0, 10.0             # waiting time uniformly between 2 and 10 minutes
    print(f"density = {1 / (b - a):.4f}")
    print(f"P(wait < 5) = {(5 - a) / (b - a):.4f}")
    print(f"mean = {(a + b) / 2}, variance = {(b - a) ** 2 / 12:.4f}")
  `, r`
    fn main() {
        let (a, b) = (2.0_f64, 10.0_f64); // waiting time uniformly between 2 and 10 minutes
        println!("density = {:.4}", 1.0 / (b - a));
        println!("P(wait < 5) = {:.4}", (5.0 - a) / (b - a));
        println!("mean = {:.1}, variance = {:.4}", (a + b) / 2.0, (b - a).powi(2) / 12.0);
    }
  `, r`
    density = 0.1250
    P(wait < 5) = 0.3750
    mean = 6.0, variance = 5.3333
  `, "A flat density 1/(b−a) on [a, b]. Probabilities are lengths: 3 of the 8 minutes lie below 5."),

  "normal-distribution": ex(join(PY_PHI, r`
    mu, sigma = 170.0, 8.0       # adult heights in cm
    z = (185 - mu) / sigma
    print(f"z = {z:.4f}")
    print(f"P(height > 185) = {1 - phi(z):.4f}")
    print(f"P(162 < height < 178) = {phi(1) - phi(-1):.4f}")
  `), join(RS_PHI, r`
    fn main() {
        let (mu, sigma) = (170.0_f64, 8.0_f64); // adult heights in cm
        let z = (185.0 - mu) / sigma;
        println!("z = {:.4}", z);
        println!("P(height > 185) = {:.4}", 1.0 - phi(z));
        println!("P(162 < height < 178) = {:.4}", phi(1.0) - phi(-1.0));
    }
  `), r`
    z = 1.8750
    P(height > 185) = 0.0304
    P(162 < height < 178) = 0.6827
  `, "Standardise with z = (x − μ)/σ, then use the standard normal CDF Φ. About 68% of heights fall within one σ of the mean. (Both programs integrate the density numerically for Φ.)"),

  "standard-normal-distribution": ex(join(PY_PHI, r`
    for z in [-1.96, 0.0, 1.0, 1.96]:
        print(f"Phi({z:+.2f}) = {phi(z):.4f}")
  `), join(RS_PHI, r`
    fn main() {
        for z in [-1.96_f64, 0.0, 1.0, 1.96] {
            println!("Phi({:+.2}) = {:.4}", z, phi(z));
        }
    }
  `), r`
    Phi(-1.96) = 0.0250
    Phi(+0.00) = 0.5000
    Phi(+1.00) = 0.8413
    Phi(+1.96) = 0.9750
  `, "Φ(z) = P(Z ≤ z) for Z ~ N(0, 1). The values ±1.96 cut off 2.5% in each tail — the source of '95% confidence'."),

  "exponential-distribution": ex(r`
    import math
    lam = 0.5                    # requests per second -> mean gap 2 s
    print(f"P(gap > 3 s) = {math.exp(-lam * 3):.4f}")
    # memoryless: P(gap > 5 | gap > 2) equals P(gap > 3)
    print(f"P(gap > 5 | gap > 2) = {math.exp(-lam * 5) / math.exp(-lam * 2):.4f}")
    print(f"mean = {1 / lam}, variance = {1 / lam ** 2}")
  `, r`
    fn main() {
        let lam = 0.5_f64; // requests per second -> mean gap 2 s
        println!("P(gap > 3 s) = {:.4}", (-lam * 3.0).exp());
        // memoryless: P(gap > 5 | gap > 2) equals P(gap > 3)
        println!("P(gap > 5 | gap > 2) = {:.4}", (-lam * 5.0).exp() / (-lam * 2.0).exp());
        println!("mean = {:.1}, variance = {:.1}", 1.0 / lam, 1.0 / (lam * lam));
    }
  `, r`
    P(gap > 3 s) = 0.2231
    P(gap > 5 | gap > 2) = 0.2231
    mean = 2.0, variance = 4.0
  `, "P(X > t) = e^{−λt}. Having already waited 2 s does not change the chance of waiting 3 more — the memoryless property."),

  "gamma-distribution": ex(r`
    import math
    k, theta = 3, 2.0            # total time of 3 tasks, each exponential with mean 2
    pdf = lambda x: x ** (k - 1) * math.exp(-x / theta) / (math.factorial(k - 1) * theta ** k)
    print(f"f(4) = {pdf(4.0):.4f}")
    print(f"mean = {k * theta}, variance = {k * theta ** 2}")
  `, r`
    fn main() {
        let (k, theta) = (3i32, 2.0_f64); // total time of 3 tasks, each exponential with mean 2
        let fact: f64 = (1..k).map(|i| i as f64).product(); // (k-1)! = Gamma(k) for integer k
        let pdf = |x: f64| x.powi(k - 1) * (-x / theta).exp() / (fact * theta.powi(k));
        println!("f(4) = {:.4}", pdf(4.0));
        println!("mean = {:.1}, variance = {:.1}", k as f64 * theta, k as f64 * theta * theta);
    }
  `, r`
    f(4) = 0.1353
    mean = 6.0, variance = 12.0
  `, "For integer shape k, Gamma(k, θ) is the sum of k exponential waits and Γ(k) = (k−1)!. Mean kθ, variance kθ²."),

  "beta-distribution": ex(r`
    from math import factorial
    a, b = 2, 5                  # e.g. belief about a conversion rate
    B = factorial(a - 1) * factorial(b - 1) / factorial(a + b - 1)
    pdf = lambda x: x ** (a - 1) * (1 - x) ** (b - 1) / B
    print(f"B(2,5) = {B:.6f}")
    print(f"f(0.2) = {pdf(0.2):.4f}")
    print(f"mean = {a / (a + b):.4f}, mode = {(a - 1) / (a + b - 2):.4f}")
  `, r`
    fn factorial(k: u64) -> f64 { (1..=k).map(|i| i as f64).product() }

    fn main() {
        let (a, b) = (2u64, 5u64); // e.g. belief about a conversion rate
        let beta = factorial(a - 1) * factorial(b - 1) / factorial(a + b - 1);
        let pdf = |x: f64| x.powi(a as i32 - 1) * (1.0 - x).powi(b as i32 - 1) / beta;
        println!("B(2,5) = {:.6}", beta);
        println!("f(0.2) = {:.4}", pdf(0.2));
        println!("mean = {:.4}, mode = {:.4}", a as f64 / (a + b) as f64, (a - 1) as f64 / (a + b - 2) as f64);
    }
  `, r`
    B(2,5) = 0.033333
    f(0.2) = 2.4576
    mean = 0.2857, mode = 0.2000
  `, "A density over probabilities in (0, 1). For integer α, β the beta function is B = (α−1)!(β−1)!/(α+β−1)!. A density above 1 is fine — only areas are probabilities."),

  "chi-square-distribution": ex(join(PY_LCG, r`
    import math
    k, reps = 4, 20000
    vals = []
    for _ in range(reps):
        s = 0.0
        for _ in range(k):                     # sum of k squared standard normals
            u1, u2 = rand(), rand()
            z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
            s += z * z
        vals.append(s)
    m = sum(vals) / reps
    v = sum((x - m) ** 2 for x in vals) / (reps - 1)
    print(f"sample mean = {m:.2f} (theory {k})")
    print(f"sample variance = {v:.2f} (theory {2 * k})")
  `), join(RS_LCG, r`
    fn main() {
        let (k, reps) = (4, 20000);
        let mut vals = vec![];
        for _ in 0..reps {
            let mut s = 0.0;
            for _ in 0..k {
                // sum of k squared standard normals (Box–Muller)
                let (u1, u2) = (rand(), rand());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                s += z * z;
            }
            vals.push(s);
        }
        let m = vals.iter().sum::<f64>() / reps as f64;
        let v = vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (reps - 1) as f64;
        println!("sample mean = {:.2} (theory {})", m, k);
        println!("sample variance = {:.2} (theory {})", v, 2 * k);
    }
  `), r`
    sample mean = 4.01 (theory 4)
    sample variance = 8.07 (theory 8)
  `, "Simulating a sum of k = 4 squared normals reproduces the chi-square moments: mean k and variance 2k."),

  "student-t-distribution": ex(r`
    import math
    def gamma_half(n2):           # Gamma(n2 / 2) for a positive integer n2
        if n2 == 1: return math.sqrt(math.pi)
        if n2 == 2: return 1.0
        return (n2 / 2 - 1) * gamma_half(n2 - 2)

    def t_pdf(t, nu):
        c = gamma_half(nu + 1) / (math.sqrt(nu * math.pi) * gamma_half(nu))
        return c * (1 + t * t / nu) ** (-(nu + 1) / 2)

    normal = lambda t: math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    for nu in [1, 5, 30]:
        print(f"nu={nu:>2}: f(3) = {t_pdf(3, nu):.5f}")
    print(f"normal: f(3) = {normal(3):.5f}")
  `, r`
    fn gamma_half(n2: u32) -> f64 {
        // Gamma(n2 / 2) for a positive integer n2
        if n2 == 1 { return std::f64::consts::PI.sqrt(); }
        if n2 == 2 { return 1.0; }
        (n2 as f64 / 2.0 - 1.0) * gamma_half(n2 - 2)
    }

    fn t_pdf(t: f64, nu: u32) -> f64 {
        let c = gamma_half(nu + 1) / ((nu as f64 * std::f64::consts::PI).sqrt() * gamma_half(nu));
        c * (1.0 + t * t / nu as f64).powf(-(nu as f64 + 1.0) / 2.0)
    }

    fn main() {
        for nu in [1u32, 5, 30] {
            println!("nu={:>2}: f(3) = {:.5}", nu, t_pdf(3.0, nu));
        }
        let normal = |t: f64| (-t * t / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        println!("normal: f(3) = {:.5}", normal(3.0));
    }
  `, r`
    nu= 1: f(3) = 0.03183
    nu= 5: f(3) = 0.01729
    nu=30: f(3) = 0.00678
    normal: f(3) = 0.00443
  `, "Three standard deviations out, the t density with few degrees of freedom is much higher than the normal's: heavy tails. As ν grows it approaches N(0, 1)."),

  "f-distribution": ex(r`
    import math
    d1, d2 = 4, 6
    # Beta(d1/2, d2/2) = Beta(2, 3) = 1! 2! / 4! with integer halves
    B = math.factorial(1) * math.factorial(2) / math.factorial(4)
    def f_pdf(x):
        return math.sqrt((d1 * x) ** d1 * d2 ** d2 / (d1 * x + d2) ** (d1 + d2)) / (x * B)
    print(f"f(1) = {f_pdf(1.0):.4f}")
    print(f"mean = d2/(d2-2) = {d2 / (d2 - 2)}")
  `, r`
    fn main() {
        let (d1, d2) = (4.0_f64, 6.0_f64);
        // Beta(d1/2, d2/2) = Beta(2, 3) = 1! 2! / 4!
        let b = (1.0 * 2.0) / 24.0;
        let f_pdf = |x: f64| ((d1 * x).powf(d1) * d2.powf(d2) / (d1 * x + d2).powf(d1 + d2)).sqrt() / (x * b);
        println!("f(1) = {:.4}", f_pdf(1.0));
        println!("mean = d2/(d2-2) = {}", d2 / (d2 - 2.0));
    }
  `, r`
    f(1) = 0.4147
    mean = d2/(d2-2) = 1.5
  `, "The F density for (4, 6) degrees of freedom evaluated at 1 (equal variances). Its mean d₂/(d₂−2) = 1.5 exists only when d₂ > 2."),

  "log-normal-distribution": ex(r`
    import math
    mu, sigma = 3.0, 0.8         # log of income (thousands) is normal
    median = math.exp(mu)
    mean = math.exp(mu + sigma ** 2 / 2)
    print(f"median = {median:.2f}, mean = {mean:.2f}")
    print("mean > median: right-skewed")
  `, r`
    fn main() {
        let (mu, sigma) = (3.0_f64, 0.8_f64); // log of income (thousands) is normal
        let median = mu.exp();
        let mean = (mu + sigma * sigma / 2.0).exp();
        println!("median = {:.2}, mean = {:.2}", median, mean);
        println!("mean > median: right-skewed");
    }
  `, r`
    median = 20.09, mean = 27.66
    mean > median: right-skewed
  `, "If ln X ~ N(μ, σ²), the median is e^μ but the mean e^{μ+σ²/2} is pulled up by the long right tail — typical of incomes and prices."),

  "weibull-distribution": ex(r`
    import math
    lam = 1000.0                 # characteristic life (hours)
    for k in [0.5, 1.0, 3.0]:
        surv = math.exp(-(500 / lam) ** k)
        print(f"k={k}: P(still working at 500 h) = {surv:.4f}")
  `, r`
    fn main() {
        let lam = 1000.0_f64; // characteristic life (hours)
        for (label, k) in [("0.5", 0.5_f64), ("1.0", 1.0), ("3.0", 3.0)] {
            let surv = (-(500.0 / lam).powf(k)).exp();
            println!("k={}: P(still working at 500 h) = {:.4}", label, surv);
        }
    }
  `, r`
    k=0.5: P(still working at 500 h) = 0.4931
    k=1.0: P(still working at 500 h) = 0.6065
    k=3.0: P(still working at 500 h) = 0.8825
  `, "Survival S(t) = e^{−(t/λ)ᵏ}. With k > 1 (wear-out) parts rarely fail early; with k < 1 (early failures) many fail soon."),

  "population-sample": ex(r`
    scores = [4, 7, 10]
    n = len(scores)
    m = sum(scores) / n
    biased = sum((x - m) ** 2 for x in scores) / n
    unbiased = sum((x - m) ** 2 for x in scores) / (n - 1)
    print(f"mean = {m}, divide by n: {biased}, divide by n-1: {unbiased}")
  `, r`
    fn main() {
        let scores = [4.0, 7.0, 10.0_f64];
        let n = scores.len() as f64;
        let m = scores.iter().sum::<f64>() / n;
        let ss: f64 = scores.iter().map(|x| (x - m).powi(2)).sum();
        println!("mean = {:.1}, divide by n: {:.1}, divide by n-1: {:.1}", m, ss / n, ss / (n - 1.0));
    }
  `, r`
    mean = 7.0, divide by n: 6.0, divide by n-1: 9.0
  `, "Measuring spread around the sample mean (which is fitted to the sample) underestimates the population variance; dividing by n − 1 (Bessel's correction) fixes it on average."),

  "sampling-distributions": ex(join(PY_LCG, r`
    import math
    pop_sd = math.sqrt(1 / 12)            # Uniform(0,1) population
    for n in [4, 16, 64]:
        means = [sum(rand() for _ in range(n)) / n for _ in range(5000)]
        m = sum(means) / len(means)
        sd = (sum((x - m) ** 2 for x in means) / (len(means) - 1)) ** 0.5
        print(f"n={n:>2}: SD of means = {sd:.4f}, sigma/sqrt(n) = {pop_sd / math.sqrt(n):.4f}")
  `), join(RS_LCG, r`
    fn main() {
        let pop_sd = (1.0_f64 / 12.0).sqrt(); // Uniform(0,1) population
        for n in [4, 16, 64] {
            let means: Vec<f64> = (0..5000).map(|_| (0..n).map(|_| rand()).sum::<f64>() / n as f64).collect();
            let m = means.iter().sum::<f64>() / means.len() as f64;
            let sd = (means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (means.len() - 1) as f64).sqrt();
            println!("n={:>2}: SD of means = {:.4}, sigma/sqrt(n) = {:.4}", n, sd, pop_sd / (n as f64).sqrt());
        }
    }
  `), r`
    n= 4: SD of means = 0.1437, sigma/sqrt(n) = 0.1443
    n=16: SD of means = 0.0729, sigma/sqrt(n) = 0.0722
    n=64: SD of means = 0.0361, sigma/sqrt(n) = 0.0361
  `, "The spread of sample means (the standard error) matches σ/√n: quadruple the sample size, halve the error."),

  "maximum-likelihood": ex(r`
    import math
    k, n = 7, 10                 # 7 heads in 10 tosses
    def loglik(p):
        return k * math.log(p) + (n - k) * math.log(1 - p)
    best = max((i / 1000 for i in range(1, 1000)), key=loglik)   # grid search
    print(f"grid-search MLE = {best:.3f}")
    print(f"closed form k/n = {k / n:.3f}")
  `, r`
    fn main() {
        let (k, n) = (7.0_f64, 10.0_f64); // 7 heads in 10 tosses
        let loglik = |p: f64| k * p.ln() + (n - k) * (1.0 - p).ln();
        let mut best = 0.001;
        for i in 1..1000 {
            let p = i as f64 / 1000.0;
            if loglik(p) > loglik(best) { best = p; } // grid search
        }
        println!("grid-search MLE = {:.3}", best);
        println!("closed form k/n = {:.3}", k / n);
    }
  `, r`
    grid-search MLE = 0.700
    closed form k/n = 0.700
  `, "Maximise the log-likelihood k log p + (n−k) log(1−p). A grid search and the calculus answer p̂ = k/n agree."),

  "confidence-intervals": ex(r`
    import math
    data = [12.1, 11.8, 12.6, 12.0, 11.5, 12.9, 12.3, 11.9]
    n = len(data)
    m = sum(data) / n
    s = math.sqrt(sum((x - m) ** 2 for x in data) / (n - 1))
    t = 2.365                    # t critical value for 95%, df = 7
    half = t * s / math.sqrt(n)
    print(f"mean = {m:.4f}, s = {s:.3f}")
    print(f"95% CI = [{m - half:.3f}, {m + half:.3f}]")
  `, r`
    fn main() {
        let data = [12.1, 11.8, 12.6, 12.0, 11.5, 12.9, 12.3, 11.9_f64];
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        let s = (data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let t = 2.365; // t critical value for 95%, df = 7
        let half = t * s / n.sqrt();
        println!("mean = {:.4}, s = {:.3}", m, s);
        println!("95% CI = [{:.3}, {:.3}]", m - half, m + half);
    }
  `, r`
    mean = 12.1375, s = 0.450
    95% CI = [11.761, 12.514]
  `, "x̄ ± t·s/√n with t₇,0.975 ≈ 2.365. About 95% of intervals built this way contain the true mean."),

  "hypothesis-testing": ex(join(PY_PHI, r`
    # A/B test: is a new page's conversion rate (p = 0.12) higher than the old one (p0 = 0.10)?
    n, p_hat, p0 = 2000, 0.12, 0.10
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    print(f"z = {z:.3f}")
    print(f"one-sided p-value = {1 - phi(z):.4f}")
  `), join(RS_PHI, r`
    fn main() {
        // A/B test: is a new page's conversion rate (p = 0.12) higher than the old one (p0 = 0.10)?
        let (n, p_hat, p0) = (2000.0_f64, 0.12_f64, 0.10_f64);
        let z = (p_hat - p0) / (p0 * (1.0 - p0) / n).sqrt();
        println!("z = {:.3}", z);
        println!("one-sided p-value = {:.4}", 1.0 - phi(z));
    }
  `), r`
    z = 2.981
    one-sided p-value = 0.0014
  `, "Under H₀ the observed rate is approximately normal around 0.10 with SE √(p₀(1−p₀)/n). z ≈ 3 gives p ≈ 0.001: very unlikely if nothing had changed."),

  "type-errors-power": ex(join(PY_PHI, r`
    z_crit = 1.96                     # two-sided alpha = 0.05
    for n in [25, 100, 200]:
        shift = 0.2 * math.sqrt(n)    # effect size d = 0.2
        power = 1 - phi(z_crit - shift) + phi(-z_crit - shift)
        print(f"n={n:>3}: power = {power:.3f}")
  `), join(RS_PHI, r`
    fn main() {
        let z_crit = 1.96; // two-sided alpha = 0.05
        for n in [25.0_f64, 100.0, 200.0] {
            let shift = 0.2 * n.sqrt(); // effect size d = 0.2
            let power = 1.0 - phi(z_crit - shift) + phi(-z_crit - shift);
            println!("n={:>3}: power = {:.3}", n, power);
        }
    }
  `), r`
    n= 25: power = 0.170
    n=100: power = 0.516
    n=200: power = 0.807
  `, "Power = P(reject H₀ | effect is real). A small effect (d = 0.2) needs about 200 observations for 80% power; with 25 you would miss it 83% of the time (Type II error)."),

  "t-tests": ex(r`
    import math
    before = [72, 68, 75, 80, 66, 71]
    after = [75, 70, 79, 81, 70, 74]
    d = [a - b for a, b in zip(after, before)]       # paired differences
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    t = md / (sd / math.sqrt(n))
    print(f"mean difference = {md:.3f}, t = {t:.3f}, df = {n - 1}")
  `, r`
    fn main() {
        let before = [72.0, 68.0, 75.0, 80.0, 66.0, 71.0_f64];
        let after = [75.0, 70.0, 79.0, 81.0, 70.0, 74.0_f64];
        let d: Vec<f64> = after.iter().zip(before.iter()).map(|(a, b)| a - b).collect(); // paired differences
        let n = d.len() as f64;
        let md = d.iter().sum::<f64>() / n;
        let sd = (d.iter().map(|x| (x - md).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let t = md / (sd / n.sqrt());
        println!("mean difference = {:.3}, t = {:.3}, df = {}", md, t, n - 1.0);
    }
  `, r`
    mean difference = 2.833, t = 5.937, df = 5
  `, "A paired t-test on scores before and after tutoring: the mean improvement is 2.83 points and t = d̄/(s_d/√n) = 5.94 with 5 df, far beyond the 2.57 needed for p < 0.05."),

  bootstrap: ex(join(PY_LCG, r`
    data = [3.1, 4.7, 2.2, 8.9, 5.0, 3.8, 15.2, 4.1, 2.9, 6.3]
    def median(x):
        s = sorted(x)
        m = len(s) // 2
        return (s[m - 1] + s[m]) / 2 if len(s) % 2 == 0 else s[m]
    boots = []
    for _ in range(5000):
        resample = [data[int(rand() * len(data))] for _ in data]
        boots.append(median(resample))
    boots.sort()
    print(f"sample median = {median(data):.2f}")
    print(f"95% bootstrap CI = [{boots[124]:.2f}, {boots[4874]:.2f}]")
  `), join(RS_LCG, r`
    fn median(x: &[f64]) -> f64 {
        let mut s = x.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = s.len() / 2;
        if s.len() % 2 == 0 { (s[m - 1] + s[m]) / 2.0 } else { s[m] }
    }

    fn main() {
        let data = [3.1, 4.7, 2.2, 8.9, 5.0, 3.8, 15.2, 4.1, 2.9, 6.3];
        let mut boots = vec![];
        for _ in 0..5000 {
            let resample: Vec<f64> = data.iter().map(|_| data[(rand() * data.len() as f64) as usize]).collect();
            boots.push(median(&resample));
        }
        boots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("sample median = {:.2}", median(&data));
        println!("95% bootstrap CI = [{:.2}, {:.2}]", boots[124], boots[4874]);
    }
  `), r`
    sample median = 4.40
    95% bootstrap CI = [3.10, 6.95]
  `, "Resample the data with replacement 5,000 times and recompute the median each time; the 2.5% and 97.5% points of those medians form a percentile confidence interval."),

  "bayesian-estimation": ex(r`
    a, b = 2, 2                  # Beta(2, 2) prior on a conversion rate
    k, n = 2, 2                  # 2 purchases out of 2 visits
    post_a, post_b = a + k, b + n - k
    print(f"MLE            = {k / n:.3f}")
    print(f"posterior mean = {post_a / (post_a + post_b):.3f}")
    print(f"MAP            = {(post_a - 1) / (post_a + post_b - 2):.3f}")
  `, r`
    fn main() {
        let (a, b) = (2.0_f64, 2.0_f64); // Beta(2, 2) prior on a conversion rate
        let (k, n) = (2.0_f64, 2.0_f64); // 2 purchases out of 2 visits
        let (pa, pb) = (a + k, b + n - k);
        println!("MLE            = {:.3}", k / n);
        println!("posterior mean = {:.3}", pa / (pa + pb));
        println!("MAP            = {:.3}", (pa - 1.0) / (pa + pb - 2.0));
    }
  `, r`
    MLE            = 1.000
    posterior mean = 0.667
    MAP            = 0.750
  `, "With only 2 observations the MLE claims a 100% conversion rate; the prior pulls the posterior mean to 0.667 and the MAP to 0.75 — sensible shrinkage."),

  "chi-square-tests": ex(r`
    O = [[30, 20, 10], [20, 25, 25]]       # bought / not bought  x  email / social / search
    rows = [sum(r) for r in O]
    cols = [sum(O[i][j] for i in range(2)) for j in range(3)]
    N = sum(rows)
    chi2 = 0.0
    for i in range(2):
        for j in range(3):
            E = rows[i] * cols[j] / N
            chi2 += (O[i][j] - E) ** 2 / E
    print(f"chi2 = {chi2:.3f}, df = {(2 - 1) * (3 - 1)}")
  `, r`
    fn main() {
        let o = [[30.0, 20.0, 10.0], [20.0, 25.0, 25.0_f64]]; // bought / not bought  x  email / social / search
        let rows: Vec<f64> = o.iter().map(|r| r.iter().sum()).collect();
        let cols: Vec<f64> = (0..3).map(|j| o[0][j] + o[1][j]).collect();
        let n: f64 = rows.iter().sum();
        let mut chi2 = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                let e = rows[i] * cols[j] / n;
                chi2 += (o[i][j] - e).powi(2) / e;
            }
        }
        println!("chi2 = {:.3}, df = {}", chi2, (2 - 1) * (3 - 1));
    }
  `, r`
    chi2 = 8.264, df = 2
  `, "Expected counts come from the margins (row total × column total / N); χ² = Σ(O − E)²/E = 8.26 with 2 df corresponds to p ≈ 0.016: purchase rate depends on channel."),

  anova: ex(r`
    groups = [[70, 72, 68, 75, 71], [78, 74, 80, 77, 76], [72, 70, 74, 73, 71]]
    allv = [x for g in groups for x in g]
    gm = sum(allv) / len(allv)
    ssb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups)
    ssw = sum((x - sum(g) / len(g)) ** 2 for g in groups for x in g)
    k, N = len(groups), len(allv)
    F = (ssb / (k - 1)) / (ssw / (N - k))
    print(f"SSB = {ssb:.3f}, SSW = {ssw:.3f}, F = {F:.3f}")
  `, r`
    fn main() {
        let groups = vec![vec![70.0, 72.0, 68.0, 75.0, 71.0], vec![78.0, 74.0, 80.0, 77.0, 76.0], vec![72.0, 70.0, 74.0, 73.0, 71.0_f64]];
        let allv: Vec<f64> = groups.iter().flatten().cloned().collect();
        let gm = allv.iter().sum::<f64>() / allv.len() as f64;
        let mean = |g: &Vec<f64>| g.iter().sum::<f64>() / g.len() as f64;
        let ssb: f64 = groups.iter().map(|g| g.len() as f64 * (mean(g) - gm).powi(2)).sum();
        let ssw: f64 = groups.iter().map(|g| { let m = mean(g); g.iter().map(|x| (x - m).powi(2)).sum::<f64>() }).sum();
        let (k, n) = (groups.len() as f64, allv.len() as f64);
        let f = (ssb / (k - 1.0)) / (ssw / (n - k));
        println!("SSB = {:.3}, SSW = {:.3}, F = {:.3}", ssb, ssw, f);
    }
  `, r`
    SSB = 98.800, SSW = 56.800, F = 10.437
  `, "Three teaching methods: between-group variation (SSB) is large compared with within-group noise (SSW), so F = 10.4 — well above the ~3.9 needed for p < 0.05 with (2, 12) df."),

  "multiple-testing": ex(r`
    pvals = [0.001, 0.008, 0.012, 0.041, 0.049, 0.18, 0.33, 0.62]
    m, alpha = len(pvals), 0.05
    print("no correction:", sum(p < alpha for p in pvals))
    print("Bonferroni:   ", sum(p < alpha / m for p in pvals))
    k = 0
    for i, p in enumerate(sorted(pvals), start=1):
        if p <= i / m * alpha:
            k = i
    print("Benjamini-Hochberg:", k)
  `, r`
    fn main() {
        let mut pvals = vec![0.001, 0.008, 0.012, 0.041, 0.049, 0.18, 0.33, 0.62_f64];
        let (m, alpha) = (pvals.len() as f64, 0.05);
        println!("no correction: {}", pvals.iter().filter(|&&p| p < alpha).count());
        println!("Bonferroni:    {}", pvals.iter().filter(|&&p| p < alpha / m).count());
        pvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        for (i, p) in pvals.iter().enumerate() {
            if *p <= (i + 1) as f64 / m * alpha { k = i + 1; }
        }
        println!("Benjamini-Hochberg: {}", k);
    }
  `, r`
    no correction: 5
    Bonferroni:    1
    Benjamini-Hochberg: 3
  `, "With 8 tests, raw α = 0.05 rejects 5, Bonferroni (α/m = 0.00625) only 1, and Benjamini–Hochberg 3 — a middle ground that controls the false discovery rate."),

  "permutation-testing": ex(r`
    from itertools import combinations
    A, B = [1, 2, 3], [4, 6, 8]
    pool = A + B
    obs = sum(B) / 3 - sum(A) / 3
    count = total = 0
    for idx in combinations(range(6), 3):        # every way to choose "group A"
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(6) if i not in idx]
        d = sum(b) / 3 - sum(a) / 3
        total += 1
        count += abs(d) >= abs(obs) - 1e-12
    print(f"observed difference = {obs:.3f}")
    print(f"exact p-value = {count}/{total} = {count / total:.3f}")
  `, r`
    fn main() {
        let pool = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0_f64]; // A = first three, B = last three
        let obs: f64 = (4.0 + 6.0 + 8.0) / 3.0 - (1.0 + 2.0 + 3.0) / 3.0;
        let (mut count, mut total) = (0, 0);
        for i in 0..6 { for j in i + 1..6 { for k in j + 1..6 { // every way to choose "group A"
            let a: f64 = pool[i] + pool[j] + pool[k];
            let b: f64 = pool.iter().sum::<f64>() - a;
            let d = b / 3.0 - a / 3.0;
            total += 1;
            if d.abs() >= obs.abs() - 1e-12 { count += 1; }
        } } }
        println!("observed difference = {:.3}", obs);
        println!("exact p-value = {}/{} = {:.3}", count, total, count as f64 / total as f64);
    }
  `, r`
    observed difference = 4.000
    exact p-value = 2/20 = 0.100
  `, "With only 6 values we can enumerate all 20 relabellings: only the observed split and its mirror are as extreme, so p = 0.1. No distributional assumptions needed."),

  "simpsons-paradox": ex(r`
    # (successes, trials) for two treatments in two patient groups
    data = {"mild":   {"A": (81, 87),  "B": (234, 270)},
            "severe": {"A": (192, 263), "B": (55, 80)}}
    for g, d in data.items():
        print(f"{g:>6}: A {d['A'][0] / d['A'][1]:.3f}  B {d['B'][0] / d['B'][1]:.3f}")
    tot = {t: (sum(data[g][t][0] for g in data), sum(data[g][t][1] for g in data)) for t in "AB"}
    print(f"pooled: A {tot['A'][0] / tot['A'][1]:.3f}  B {tot['B'][0] / tot['B'][1]:.3f}")
  `, r`
    fn main() {
        // (successes, trials) for two treatments in two patient groups
        let groups = [("mild", (81.0, 87.0), (234.0, 270.0)), ("severe", (192.0, 263.0), (55.0, 80.0))];
        let (mut ta, mut tb) = ((0.0, 0.0), (0.0, 0.0));
        for (g, a, b) in groups.iter() {
            println!("{:>6}: A {:.3}  B {:.3}", g, a.0 / a.1, b.0 / b.1);
            ta = (ta.0 + a.0, ta.1 + a.1);
            tb = (tb.0 + b.0, tb.1 + b.1);
        }
        println!("pooled: A {:.3}  B {:.3}", ta.0 / ta.1, tb.0 / tb.1);
    }
  `, r`
      mild: A 0.931  B 0.867
    severe: A 0.730  B 0.688
    pooled: A 0.780  B 0.826
  `, "Treatment A wins in both groups but loses overall, because A was given mostly to severe cases. The confounder (severity) reverses the pooled comparison — the classic kidney-stone data."),

  "outlier-detection": ex(r`
    x = [21, 23, 22, 24, 20, 22, 23, 21, 58, 22]
    s = sorted(x)
    def q(p):                       # linear-interpolation quantile
        pos = (len(s) - 1) * p
        lo = int(pos)
        return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (pos - lo)
    q1, q3 = q(0.25), q(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, fences = [{lo:.3f}, {hi:.3f}]")
    print("outliers:", [v for v in x if v < lo or v > hi])
  `, r`
    fn main() {
        let x = [21.0, 23.0, 22.0, 24.0, 20.0, 22.0, 23.0, 21.0, 58.0, 22.0_f64];
        let mut s = x.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| { // linear-interpolation quantile
            let pos = (s.len() - 1) as f64 * p;
            let lo = pos as usize;
            s[lo] + (s[(lo + 1).min(s.len() - 1)] - s[lo]) * (pos - lo as f64)
        };
        let (q1, q3) = (q(0.25), q(0.75));
        let iqr = q3 - q1;
        let (lo, hi) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
        println!("Q1 = {:.2}, Q3 = {:.2}, fences = [{:.3}, {:.3}]", q1, q3, lo, hi);
        let out: Vec<String> = x.iter().filter(|&&v| v < lo || v > hi).map(|v| format!("{}", v)).collect();
        println!("outliers: [{}]", out.join(", "));
    }
  `, r`
    Q1 = 21.25, Q3 = 23.00, fences = [18.625, 25.625]
    outliers: [58]
  `, "Tukey's rule: anything beyond 1.5 IQRs outside the quartiles is flagged. The 58 stands out; the IQR itself is barely affected by it (robust)."),

  "partial-correlation": ex(r`
    r_xy, r_xz, r_yz = 0.64, 0.8, 0.8           # ice cream (X), sunburn (Y), temperature (Z)
    partial = (r_xy - r_xz * r_yz) / ((1 - r_xz ** 2) * (1 - r_yz ** 2)) ** 0.5
    print(f"r_XY = {r_xy}, r_XY.Z = {abs(partial):.4f}")   # abs avoids printing -0.0000
  `, r`
    fn main() {
        let (r_xy, r_xz, r_yz) = (0.64_f64, 0.8_f64, 0.8_f64); // ice cream (X), sunburn (Y), temperature (Z)
        let partial = (r_xy - r_xz * r_yz) / ((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)).sqrt();
        println!("r_XY = {}, r_XY.Z = {:.4}", r_xy, partial.abs());
    }
  `, r`
    r_XY = 0.64, r_XY.Z = 0.0000
  `, "All of the 0.64 correlation between ice cream and sunburns is explained by temperature: removing it leaves a partial correlation of 0. (abs() avoids printing −0.0000 when rounding a tiny negative.)"),

  multicollinearity: ex(r`
    for rho in [0.0, 0.5, 0.9, 0.99]:
        vif = 1 / (1 - rho ** 2)
        print(f"correlation {rho:.2f}: VIF = {vif:.2f}")
  `, r`
    fn main() {
        for rho in [0.0_f64, 0.5, 0.9, 0.99] {
            println!("correlation {:.2}: VIF = {:.2}", rho, 1.0 / (1.0 - rho * rho));
        }
    }
  `, r`
    correlation 0.00: VIF = 1.00
    correlation 0.50: VIF = 1.33
    correlation 0.90: VIF = 5.26
    correlation 0.99: VIF = 50.25
  `, "With two features VIF = 1/(1−ρ²): at ρ = 0.99 coefficient variance is inflated 50×, making individual coefficients meaningless even if predictions stay good."),

  "bayesian-updating": ex(r`
    a, b = 1, 1                       # uniform prior on a coin's bias
    flips = [1, 1, 0, 1, 1, 0, 1, 1]  # 1 = heads
    for i, f in enumerate(flips, start=1):
        a, b = a + f, b + (1 - f)     # conjugate update
        if i in (1, 4, 8):
            print(f"after {i} flips: Beta({a}, {b}), mean = {a / (a + b):.3f}")
  `, r`
    fn main() {
        let (mut a, mut b) = (1, 1); // uniform prior on a coin's bias
        let flips = [1, 1, 0, 1, 1, 0, 1, 1]; // 1 = heads
        for (i, f) in flips.iter().enumerate() {
            a += f; // conjugate update
            b += 1 - f;
            if [1, 4, 8].contains(&(i + 1)) {
                println!("after {} flips: Beta({}, {}), mean = {:.3}", i + 1, a, b, a as f64 / (a + b) as f64);
            }
        }
    }
  `, r`
    after 1 flips: Beta(2, 1), mean = 0.667
    after 4 flips: Beta(4, 2), mean = 0.667
    after 8 flips: Beta(7, 3), mean = 0.700
  `, "Each flip adds 1 to α (heads) or β (tails): the Beta posterior after each observation becomes the prior for the next."),
};
