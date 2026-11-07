import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

export const aiEngineeringCode: Record<string, CodeExample> = {
  "data-validation": ex(r`
    rows = [{"id": 1, "age": 34, "country": "IN"}, {"id": 2, "age": -5, "country": "US"},
            {"id": 3, "age": None, "country": "DE"}, {"id": 3, "age": 41, "country": "Germany"}]
    allowed = {"IN", "US", "DE", "BR"}
    seen = set()
    for row in rows:
        errors = []
        if row["age"] is None: errors.append("age is null")
        elif not 0 <= row["age"] <= 120: errors.append("age out of range")
        if row["country"] not in allowed: errors.append("unknown country")
        if row["id"] in seen: errors.append("duplicate id")
        seen.add(row["id"])
        print(f"row {row['id']}: {'OK' if not errors else '; '.join(errors)}")
  `, r`
    use std::collections::HashSet;

    fn main() {
        let rows: [(i32, Option<i32>, &str); 4] = [(1, Some(34), "IN"), (2, Some(-5), "US"), (3, None, "DE"), (3, Some(41), "Germany")];
        let allowed = ["IN", "US", "DE", "BR"];
        let mut seen = HashSet::new();
        for (id, age, country) in rows.iter() {
            let mut errors = Vec::new();
            match age {
                None => errors.push("age is null"),
                Some(a) if *a < 0 || *a > 120 => errors.push("age out of range"),
                _ => {}
            }
            if !allowed.contains(country) { errors.push("unknown country"); }
            if seen.contains(id) { errors.push("duplicate id"); }
            seen.insert(*id);
            println!("row {}: {}", id, if errors.is_empty() { "OK".to_string() } else { errors.join("; ") });
        }
    }
  `, r`
    row 1: OK
    row 2: age out of range
    row 3: age is null
    row 3: unknown country; duplicate id
  `, "Each record is checked against explicit rules — nulls, ranges, allowed categories, uniqueness — before it can reach training. Three of the four rows fail, each for a different reason, and the report says exactly why."),

  "feature-stores": ex(r`
    feature_log = [(1, 40.0), (4, 55.0), (9, 30.0), (13, 70.0)]   # (day written, avg_spend_7d)
    labels = [5, 10, 14]                                           # days on which labels were observed
    for t in labels:
        past = [v for d, v in feature_log if d <= t]               # point-in-time: only values known at t
        print(f"label day {t:2d}: point-in-time feature={past[-1]:.1f}  naive latest={feature_log[-1][1]:.1f}")
  `, r`
    fn main() {
        let feature_log = [(1, 40.0), (4, 55.0), (9, 30.0), (13, 70.0_f64)]; // (day written, avg_spend_7d)
        let labels = [5, 10, 14]; // days on which labels were observed
        for &t in labels.iter() {
            // point-in-time: only values known at t
            let past = feature_log.iter().filter(|(d, _)| *d <= t).last().unwrap().1;
            println!("label day {:2}: point-in-time feature={:.1}  naive latest={:.1}", t, past, feature_log[3].1);
        }
    }
  `, r`
    label day  5: point-in-time feature=55.0  naive latest=70.0
    label day 10: point-in-time feature=30.0  naive latest=70.0
    label day 14: point-in-time feature=70.0  naive latest=70.0
  `, "A point-in-time join gives each training label the feature value that existed at that moment. Using the latest value (70.0, written on day 13) for the labels of days 5 and 10 would leak the future into training."),

  "training-inference-pipelines": ex(r`
    users, requests_per_day = 1_000_000, 200_000
    for interval_h in [6, 24]:
        batch = users * 24 // interval_h
        print(f"batch every {interval_h:2d} h: {batch:,} predictions/day, average staleness {interval_h / 2:.1f} h")
    print(f"online: {requests_per_day:,} predictions/day, staleness 0 h, but model latency on every request")
  `, r`
    fn commas(n: u64) -> String {
        let s = n.to_string();
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); }
            out.push(c);
        }
        out
    }

    fn main() {
        let (users, requests_per_day) = (1_000_000u64, 200_000u64);
        for interval_h in [6u64, 24] {
            let batch = users * 24 / interval_h;
            println!("batch every {:2} h: {} predictions/day, average staleness {:.1} h", interval_h, commas(batch), interval_h as f64 / 2.0);
        }
        println!("online: {} predictions/day, staleness 0 h, but model latency on every request", commas(requests_per_day));
    }
  `, r`
    batch every  6 h: 4,000,000 predictions/day, average staleness 3.0 h
    batch every 24 h: 1,000,000 predictions/day, average staleness 12.0 h
    online: 200,000 predictions/day, staleness 0 h, but model latency on every request
  `, "Batch inference scores every user on a schedule — cheap per request but stale and wasteful when most users never return. Online inference scores only what is requested, fresh, at the cost of serving latency."),

  "model-versioning": ex(r`
    def fnv1a(s):
        h = 0x811C9DC5
        for ch in s.encode():
            h = ((h ^ ch) * 0x01000193) % 2**32
        return f"{h:08x}"
    base = {"data": "v3", "commit": "a1c9e2f", "lr": "0.01", "seed": "42"}
    key = lambda cfg: "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    print("model id      :", fnv1a(key(base)))
    print("same recipe   :", fnv1a(key(dict(base))))
    print("seed changed  :", fnv1a(key({**base, "seed": "7"})))
  `, r`
    fn fnv1a(s: &str) -> String {
        let mut h: u32 = 0x811C9DC5;
        for b in s.bytes() { h = (h ^ b as u32).wrapping_mul(0x01000193); }
        format!("{:08x}", h)
    }

    fn main() {
        // keys in sorted order: commit, data, lr, seed
        let key = |seed: &str| format!("commit=a1c9e2f|data=v3|lr=0.01|seed={}", seed);
        println!("model id      : {}", fnv1a(&key("42")));
        println!("same recipe   : {}", fnv1a(&key("42")));
        println!("seed changed  : {}", fnv1a(&key("7")));
    }
  `, r`
    model id      : 7a424e5a
    same recipe   : 7a424e5a
    seed changed  : 06cd5d05
  `, "Hashing every ingredient of a training run — data version, code commit, hyperparameters, seed — gives a reproducible model id: the same recipe always yields the same id, and any change yields a new one."),

  "experiment-tracking": ex(r`
    def val_score(lr, depth):                      # stand-in for a real training run
        return 0.9 - 40 * (lr - 0.05) ** 2 - 0.01 * (depth - 6) ** 2
    runs = []
    for lr in [0.01, 0.05, 0.1]:
        for depth in [4, 6, 8]:
            runs.append({"lr": lr, "depth": depth, "val": val_score(lr, depth)})
    runs.sort(key=lambda r_: -r_["val"])
    for r_ in runs[:3]:
        print(f"lr={r_['lr']:.2f} depth={r_['depth']} val={r_['val']:.4f}")
    print(f"logged {len(runs)} runs; best: lr={runs[0]['lr']}, depth={runs[0]['depth']}")
  `, r`
    fn val_score(lr: f64, depth: i32) -> f64 {
        // stand-in for a real training run
        0.9 - 40.0 * (lr - 0.05).powi(2) - 0.01 * ((depth - 6) as f64).powi(2)
    }

    fn main() {
        let mut runs = Vec::new();
        for lr in [0.01, 0.05, 0.1] {
            for depth in [4, 6, 8] { runs.push((lr, depth, val_score(lr, depth))); }
        }
        runs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        for r in runs.iter().take(3) { println!("lr={:.2} depth={} val={:.4}", r.0, r.1, r.2); }
        println!("logged {} runs; best: lr={}, depth={}", runs.len(), runs[0].0, runs[0].1);
    }
  `, r`
    lr=0.05 depth=6 val=0.9000
    lr=0.05 depth=4 val=0.8600
    lr=0.05 depth=8 val=0.8600
    logged 9 runs; best: lr=0.05, depth=6
  `, "Every run's parameters and metric are logged, so they can be sorted and compared afterwards instead of being remembered. The record makes the winning configuration — and its runners-up — reproducible."),

  "evaluation-pipelines": ex(r`
    slices = {"mobile": (400, 0.86, 0.89), "desktop": (350, 0.90, 0.93), "non-English": (100, 0.74, 0.66)}  # n, baseline, candidate
    n = sum(v[0] for v in slices.values())
    base = sum(v[0] * v[1] for v in slices.values()) / n
    cand = sum(v[0] * v[2] for v in slices.values()) / n
    print(f"overall: baseline={base:.4f} candidate={cand:.4f}")
    ok = cand >= base
    for name, (_, b, c) in slices.items():
        passed = b - c <= 0.02
        ok = ok and passed
        print(f"{name:<12} {b:.2f} -> {c:.2f} {'pass' if passed else 'FAIL'}")
    print("release:", "promote" if ok else "blocked")
  `, r`
    fn main() {
        let slices = [("mobile", 400.0, 0.86, 0.89), ("desktop", 350.0, 0.90, 0.93), ("non-English", 100.0, 0.74, 0.66_f64)]; // n, baseline, candidate
        let n: f64 = slices.iter().map(|s| s.1).sum();
        let base = slices.iter().map(|s| s.1 * s.2).sum::<f64>() / n;
        let cand = slices.iter().map(|s| s.1 * s.3).sum::<f64>() / n;
        println!("overall: baseline={:.4} candidate={:.4}", base, cand);
        let mut ok = cand >= base;
        for (name, _, b, c) in slices.iter() {
            let passed = b - c <= 0.02;
            ok = ok && passed;
            println!("{:<12} {:.2} -> {:.2} {}", name, b, c, if passed { "pass" } else { "FAIL" });
        }
        println!("release: {}", if ok { "promote" } else { "blocked" });
    }
  `, r`
    overall: baseline=0.8624 candidate=0.8794
    mobile       0.86 -> 0.89 pass
    desktop      0.90 -> 0.93 pass
    non-English  0.74 -> 0.66 FAIL
    release: blocked
  `, "The candidate is better overall, but the small non-English slice drops 8 points. A per-slice gate catches the regression that the weighted average hides, and the release is blocked."),

  "model-serving": ex(r`
    import math
    lat = [12, 15, 14, 13, 90, 16, 14, 13, 15, 220, 14, 13, 17, 15, 16, 14, 13, 18, 15, 14]   # ms
    s = sorted(lat)
    pct = lambda p: s[min(len(s) - 1, math.ceil(p * len(s)) - 1)]                      # nearest-rank percentile
    print(f"mean={sum(lat) / len(lat):.1f} ms  p50={pct(0.5)} ms  p95={pct(0.95)} ms  p99={pct(0.99)} ms")
    t0, t1 = 20.0, 2.0                                     # ms per forward pass = t0 + t1 * batch size
    for b in [1, 8, 32]:
        print(f"batch {b:2d}: {1000 * b / (t0 + t1 * b):.0f} requests/s per GPU")
  `, r`
    fn main() {
        let lat = [12, 15, 14, 13, 90, 16, 14, 13, 15, 220, 14, 13, 17, 15, 16, 14, 13, 18, 15, 14]; // ms
        let mut s = lat.to_vec();
        s.sort();
        let pct = |p: f64| s[((p * s.len() as f64).ceil() as usize).max(1).min(s.len()) - 1]; // nearest-rank percentile
        let mean = lat.iter().sum::<i32>() as f64 / lat.len() as f64;
        println!("mean={:.1} ms  p50={} ms  p95={} ms  p99={} ms", mean, pct(0.5), pct(0.95), pct(0.99));
        let (t0, t1) = (20.0, 2.0); // ms per forward pass = t0 + t1 * batch size
        for b in [1.0, 8.0, 32.0_f64] {
            println!("batch {:2}: {:.0} requests/s per GPU", b as i32, 1000.0 * b / (t0 + t1 * b));
        }
    }
  `, r`
    mean=28.6 ms  p50=14 ms  p95=90 ms  p99=220 ms
    batch  1: 45 requests/s per GPU
    batch  8: 222 requests/s per GPU
    batch 32: 381 requests/s per GPU
  `, "Two slow requests barely move the median but dominate the tail percentiles — which is why serving SLOs are written in p95/p99. Batching amortises the fixed cost of a forward pass and multiplies throughput."),

  "data-drift": ex(r`
    import math
    bins = ["<20", "20-40", "40-60", ">60"]
    train = [0.25, 0.40, 0.25, 0.10]          # share of each age bin at training time
    live = [0.10, 0.30, 0.35, 0.25]           # share in this week's traffic
    psi = 0.0
    for b, e, a in zip(bins, train, live):
        term = (a - e) * math.log(a / e)
        psi += term
        print(f"{b:<5}: train={e:.2f} live={a:.2f} contribution={term:.4f}")
    print(f"PSI = {psi:.4f} ->", "significant drift" if psi > 0.25 else "moderate drift" if psi > 0.1 else "stable")
  `, r`
    fn main() {
        let bins = ["<20", "20-40", "40-60", ">60"];
        let train = [0.25, 0.40, 0.25, 0.10_f64]; // share of each age bin at training time
        let live = [0.10, 0.30, 0.35, 0.25_f64]; // share in this week's traffic
        let mut psi = 0.0;
        for i in 0..4 {
            let term = (live[i] - train[i]) * (live[i] / train[i]).ln();
            psi += term;
            println!("{:<5}: train={:.2} live={:.2} contribution={:.4}", bins[i], train[i], live[i], term);
        }
        let verdict = if psi > 0.25 { "significant drift" } else if psi > 0.1 { "moderate drift" } else { "stable" };
        println!("PSI = {:.4} -> {}", psi, verdict);
    }
  `, r`
    <20  : train=0.25 live=0.10 contribution=0.1374
    20-40: train=0.40 live=0.30 contribution=0.0288
    40-60: train=0.25 live=0.35 contribution=0.0336
    >60  : train=0.10 live=0.25 contribution=0.1374
    PSI = 0.3373 -> significant drift
  `, "The population stability index sums, over bins, the change in share times the log-ratio of shares. Traffic has shifted towards older users; a PSI above 0.25 is the usual rule-of-thumb threshold for significant drift."),

  "cosine-similarity": ex(r`
    import math
    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    q = [1.0, 2.0, 0.0]
    for name, v in [("same direction, longer", [2.0, 4.0, 0.0]), ("orthogonal", [-2.0, 1.0, 0.0]), ("opposite", [-1.0, -2.0, 0.0]), ("similar", [1.0, 1.5, 0.5])]:
        print(f"{name:<23} cos={cos(q, v) + 0.0:+.4f}")
  `, r`
    fn cos(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|y| y * y).sum::<f64>().sqrt())
    }

    fn main() {
        let q = [1.0, 2.0, 0.0];
        let vs: [(&str, [f64; 3]); 4] = [("same direction, longer", [2.0, 4.0, 0.0]), ("orthogonal", [-2.0, 1.0, 0.0]), ("opposite", [-1.0, -2.0, 0.0]), ("similar", [1.0, 1.5, 0.5])];
        for (name, v) in vs.iter() {
            println!("{:<23} cos={:+.4}", name, cos(&q, v) + 0.0);
        }
    }
  `, r`
    same direction, longer  cos=+1.0000
    orthogonal              cos=+0.0000
    opposite                cos=-1.0000
    similar                 cos=+0.9562
  `, "Cosine similarity compares directions only: a longer vector in the same direction scores 1, a perpendicular one 0 and an opposite one −1. That length-invariance is why it is the default for comparing embeddings."),

  "vector-databases": ex(r`
    import math
    vecs = [(0.1, 0.2), (0.2, 0.1), (0.15, 0.25), (0.9, 0.8), (0.8, 0.9), (0.85, 0.95), (0.5, 0.45)]
    centroids = [(0.15, 0.18), (0.85, 0.88)]                 # IVF cells learned by k-means
    cell = [min(range(2), key=lambda c: math.dist(v, centroids[c])) for v in vecs]
    q, k = (0.45, 0.4), 3
    exact = sorted(range(len(vecs)), key=lambda i: math.dist(vecs[i], q))[:k]
    probe = min(range(2), key=lambda c: math.dist(q, centroids[c]))   # nprobe = 1
    cand = [i for i in range(len(vecs)) if cell[i] == probe]
    approx = sorted(cand, key=lambda i: math.dist(vecs[i], q))[:k]
    print("exact top-3 :", exact)
    print("IVF top-3   :", approx, f"(scanned {len(cand)} of {len(vecs)})")
    print(f"recall@3 = {len(set(exact) & set(approx)) / k:.3f}")
  `, r`
    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 { (a.0 - b.0).hypot(a.1 - b.1) }

    fn main() {
        let vecs = [(0.1, 0.2), (0.2, 0.1), (0.15, 0.25), (0.9, 0.8), (0.8, 0.9), (0.85, 0.95), (0.5, 0.45_f64)];
        let centroids = [(0.15, 0.18), (0.85, 0.88_f64)]; // IVF cells learned by k-means
        let cell: Vec<usize> = vecs.iter().map(|&v| if dist(v, centroids[0]) <= dist(v, centroids[1]) { 0 } else { 1 }).collect();
        let (q, k) = ((0.45, 0.4), 3);
        let mut exact: Vec<usize> = (0..vecs.len()).collect();
        exact.sort_by(|&a, &b| dist(vecs[a], q).partial_cmp(&dist(vecs[b], q)).unwrap());
        exact.truncate(k);
        let probe = if dist(q, centroids[0]) <= dist(q, centroids[1]) { 0 } else { 1 }; // nprobe = 1
        let mut approx: Vec<usize> = (0..vecs.len()).filter(|&i| cell[i] == probe).collect();
        let scanned = approx.len();
        approx.sort_by(|&a, &b| dist(vecs[a], q).partial_cmp(&dist(vecs[b], q)).unwrap());
        approx.truncate(k);
        println!("exact top-3 : {:?}", exact);
        println!("IVF top-3   : {:?} (scanned {} of {})", approx, scanned, vecs.len());
        let hit = approx.iter().filter(|i| exact.contains(i)).count();
        println!("recall@3 = {:.3}", hit as f64 / k as f64);
    }
  `, r`
    exact top-3 : [6, 2, 1]
    IVF top-3   : [6, 2, 1] (scanned 4 of 7)
    recall@3 = 1.000
  `, "The IVF index only scans the cell whose centroid is closest to the query, so it touches fewer vectors. The price is recall: a true neighbour that lives in the other cell is missed. Probing more cells trades speed back for recall."),

  rag: ex(r`
    docs = ["Refunds are possible within 30 days of purchase.",
            "Shipping takes 3 to 5 business days.",
            "You can reset your password from the login page."]
    question = "How many days do I have to ask for a refund?"
    stop = {"how", "many", "do", "i", "have", "to", "ask", "for", "a", "are", "of", "you", "can", "your", "from", "the"}
    def words(s):                                   # lowercase, drop stop words, strip a plural s
        ws = (w.strip(".?,!").lower() for w in s.split())
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in ws if w not in stop}
    scores = [len(words(question) & words(d)) for d in docs]           # toy lexical retriever
    best = max(range(len(docs)), key=lambda i: scores[i])
    print("retrieval scores:", scores)
    prompt = f"Answer using only the context.\nContext: {docs[best]}\nQuestion: {question}\nAnswer:"
    print(prompt)
  `, r`
    use std::collections::HashSet;

    fn words(s: &str) -> HashSet<String> {
        // lowercase, drop stop words, strip a plural s
        let stop = ["how", "many", "do", "i", "have", "to", "ask", "for", "a", "are", "of", "you", "can", "your", "from", "the"];
        s.split_whitespace()
            .map(|w| w.trim_matches(|c: char| ".?,!".contains(c)).to_lowercase())
            .filter(|w| !stop.contains(&w.as_str()))
            .map(|w| if w.ends_with('s') && w.len() > 3 { w[..w.len() - 1].to_string() } else { w })
            .collect()
    }

    fn main() {
        let docs = ["Refunds are possible within 30 days of purchase.", "Shipping takes 3 to 5 business days.", "You can reset your password from the login page."];
        let question = "How many days do I have to ask for a refund?";
        let q = words(question);
        let scores: Vec<usize> = docs.iter().map(|d| q.intersection(&words(d)).count()).collect(); // toy lexical retriever
        let best = (0..docs.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
        println!("retrieval scores: {:?}", scores);
        println!("Answer using only the context.\nContext: {}\nQuestion: {}\nAnswer:", docs[best], question);
    }
  `, r`
    retrieval scores: [2, 1, 0]
    Answer using only the context.
    Context: Refunds are possible within 30 days of purchase.
    Question: How many days do I have to ask for a refund?
    Answer:
  `, "Retrieval-augmented generation first finds the most relevant document, then places it in the prompt so the model answers from that text instead of from memory. Even this toy retriever needed stop-word removal and plural stripping to match 'refunds' with 'refund' — real systems use embedding search over chunks (often combined with BM25) instead of raw word overlap."),

  chunking: ex(r`
    text = " ".join(f"w{i}" for i in range(1, 23))      # a 22-word document
    tokens = text.split()
    size, overlap = 8, 2
    step = size - overlap
    chunks = [tokens[i:i + size] for i in range(0, len(tokens) - overlap, step)]
    for k, c in enumerate(chunks):
        print(f"chunk {k}: {c[0]}..{c[-1]} ({len(c)} words)")
  `, r`
    fn main() {
        let tokens: Vec<String> = (1..=22).map(|i| format!("w{}", i)).collect(); // a 22-word document
        let (size, overlap) = (8usize, 2usize);
        let step = size - overlap;
        let mut k = 0;
        let mut i = 0;
        while i < tokens.len() - overlap {
            let c = &tokens[i..(i + size).min(tokens.len())];
            println!("chunk {}: {}..{} ({} words)", k, c[0], c[c.len() - 1], c.len());
            k += 1;
            i += step;
        }
    }
  `, r`
    chunk 0: w1..w8 (8 words)
    chunk 1: w7..w14 (8 words)
    chunk 2: w13..w20 (8 words)
    chunk 3: w19..w22 (4 words)
  `, "Fixed-size chunks of 8 words that overlap by 2 cover the whole document; the overlap keeps a sentence that straddles a boundary intact in at least one chunk. The last chunk is shorter."),

  "hybrid-search-reranking": ex(r`
    import math
    docs = [["reset", "password"], ["error", "e1042", "sync"], ["refund", "policy"], ["login", "password", "help"]]
    query = ["password", "e1042"]
    N, avgdl, k1, b = len(docs), sum(map(len, docs)) / len(docs), 1.2, 0.75
    def bm25(d):
        s = 0.0
        for t in query:
            n = sum(t in x for x in docs)
            if n == 0: continue
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5)); f = d.count(t)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        return s
    dense = [0.82, 0.40, 0.10, 0.75]                  # similarity scores from an embedding model
    lex = [bm25(d) for d in docs]
    rank = lambda sc: sorted(range(N), key=lambda i: -sc[i])
    rl, rd = rank(lex), rank(dense)
    rrf = [1 / (60 + rl.index(i) + 1) + 1 / (60 + rd.index(i) + 1) for i in range(N)]
    print("BM25 :", ", ".join(f"{v:.3f}" for v in lex))
    print("RRF ranking:", rank(rrf))
  `, r`
    fn main() {
        let docs: Vec<Vec<&str>> = vec![vec!["reset", "password"], vec!["error", "e1042", "sync"], vec!["refund", "policy"], vec!["login", "password", "help"]];
        let query = ["password", "e1042"];
        let n = docs.len() as f64;
        let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
        let (k1, b) = (1.2, 0.75);
        let bm25 = |d: &Vec<&str>| -> f64 {
            let mut s = 0.0;
            for t in query.iter() {
                let nt = docs.iter().filter(|x| x.contains(t)).count() as f64;
                if nt == 0.0 { continue; }
                let idf = (1.0 + (n - nt + 0.5) / (nt + 0.5)).ln();
                let f = d.iter().filter(|w| *w == t).count() as f64;
                s += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * d.len() as f64 / avgdl));
            }
            s
        };
        let dense = [0.82, 0.40, 0.10, 0.75]; // similarity scores from an embedding model
        let lex: Vec<f64> = docs.iter().map(|d| bm25(d)).collect();
        let rank = |sc: &[f64]| { let mut r: Vec<usize> = (0..sc.len()).collect(); r.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap()); r };
        let (rl, rd) = (rank(&lex), rank(&dense));
        let pos = |r: &Vec<usize>, i: usize| r.iter().position(|&x| x == i).unwrap() as f64;
        let rrf: Vec<f64> = (0..docs.len()).map(|i| 1.0 / (61.0 + pos(&rl, i)) + 1.0 / (61.0 + pos(&rd, i))).collect();
        println!("BM25 : {}", lex.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "));
        println!("RRF ranking: {:?}", rank(&rrf));
    }
  `, r`
    BM25 : 0.755, 1.113, 0.000, 0.641
    RRF ranking: [0, 1, 3, 2]
  `, "BM25 strongly favours the document containing the rare exact code 'e1042', while the embedding scores favour the password documents. Reciprocal rank fusion combines the two rankings using ranks alone, so neither score scale has to be calibrated."),

  "prompt-engineering": ex(r`
    template = "Extract fields as 'key: value' lines.\nExample:\nReview: Great pasta!\nname: unknown\nsentiment: positive\nrating: 5\n\nReview: {review}\n"
    prompt = template.format(review="Aster Cafe was lovely but pricey. 4 stars.")
    print(f"prompt length ~ {len(prompt) // 4} tokens")
    output = "name: Aster Cafe\nsentiment: positive\nrating: 4"       # the model's reply
    fields = dict(line.split(": ", 1) for line in output.splitlines())
    errors = []
    if fields.get("sentiment") not in ("positive", "neutral", "negative"): errors.append("bad sentiment")
    if not fields.get("rating", "").isdigit() or not 1 <= int(fields["rating"]) <= 5: errors.append("bad rating")
    print("parsed:", fields)
    print("valid" if not errors else errors)
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let review = "Aster Cafe was lovely but pricey. 4 stars.";
        let prompt = format!("Extract fields as 'key: value' lines.\nExample:\nReview: Great pasta!\nname: unknown\nsentiment: positive\nrating: 5\n\nReview: {}\n", review);
        println!("prompt length ~ {} tokens", prompt.chars().count() / 4);
        let output = "name: Aster Cafe\nsentiment: positive\nrating: 4"; // the model's reply
        let fields: BTreeMap<&str, &str> = output.lines().filter_map(|l| l.split_once(": ")).collect();
        let mut errors = Vec::new();
        if !["positive", "neutral", "negative"].contains(fields.get("sentiment").unwrap_or(&"")) { errors.push("bad sentiment"); }
        match fields.get("rating").and_then(|r| r.parse::<i32>().ok()) {
            Some(v) if (1..=5).contains(&v) => {}
            _ => errors.push("bad rating"),
        }
        // print in insertion order to match the Python dict
        println!("parsed: {{'name': '{}', 'sentiment': '{}', 'rating': '{}'}}", fields["name"], fields["sentiment"], fields["rating"]);
        println!("{}", if errors.is_empty() { "valid".to_string() } else { format!("{:?}", errors) });
    }
  `, r`
    prompt length ~ 41 tokens
    parsed: {'name': 'Aster Cafe', 'sentiment': 'positive', 'rating': '4'}
    valid
  `, "A clear instruction plus one worked example fixes the output format; the application then parses the reply and validates every field before using it. Never pass unvalidated model output to downstream code."),

  "tool-calling": ex(r`
    population = {"lisbon": 545000, "porto": 232000}
    def run_tool(name, arg):
        if name == "lookup_population":
            return population.get(arg, "error: unknown city")
        if name == "multiply":
            a, b = arg
            return a * b
        return "error: unknown tool"
    # the model's (scripted) tool calls for: "What is 17% of the population of Lisbon?"
    pop = run_tool("lookup_population", "lisbon")
    print("tool lookup_population('lisbon') ->", pop)
    res = run_tool("multiply", (pop, 0.17))
    print(f"tool multiply({pop}, 0.17) -> {res:.1f}")
    print("tool lookup_population('atlantis') ->", run_tool("lookup_population", "atlantis"))
  `, r`
    fn lookup_population(city: &str) -> Result<f64, String> {
        match city {
            "lisbon" => Ok(545000.0),
            "porto" => Ok(232000.0),
            _ => Err("error: unknown city".to_string()),
        }
    }

    fn main() {
        // the model's (scripted) tool calls for: "What is 17% of the population of Lisbon?"
        let pop = lookup_population("lisbon").unwrap();
        println!("tool lookup_population('lisbon') -> {}", pop as i64);
        println!("tool multiply({}, 0.17) -> {:.1}", pop as i64, pop * 0.17);
        match lookup_population("atlantis") {
            Ok(v) => println!("tool lookup_population('atlantis') -> {}", v),
            Err(e) => println!("tool lookup_population('atlantis') -> {}", e),
        }
    }
  `, r`
    tool lookup_population('lisbon') -> 545000
    tool multiply(545000, 0.17) -> 92650.0
    tool lookup_population('atlantis') -> error: unknown city
  `, "The model decides which tool to call with which arguments; the application executes it and feeds the result back. Errors are returned as data — the model must report 'unknown city' rather than invent a number."),

  "llm-evaluation": ex(r`
    def tokens(s):
        return [w.strip(".,").lower() for w in s.split()]
    def f1(pred, ref):
        p, r_ = tokens(pred), tokens(ref)
        common = sum(min(p.count(w), r_.count(w)) for w in set(p))
        if common == 0: return 0.0
        prec, rec = common / len(p), common / len(r_)
        return 2 * prec * rec / (prec + rec)
    cases = [("1998", "1998"), ("It opened in 1998.", "1998"), ("about 17 km", "12.3 km")]
    for pred, ref in cases:
        em = int(tokens(pred) == tokens(ref))
        print(f"pred={pred!r:<22} EM={em} F1={f1(pred, ref):.3f}")
  `, r`
    fn tokens(s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.trim_matches(|c| c == '.' || c == ',').to_lowercase()).collect()
    }

    fn f1(pred: &str, reference: &str) -> f64 {
        let (p, r) = (tokens(pred), tokens(reference));
        let mut uniq = p.clone();
        uniq.sort();
        uniq.dedup();
        let common: usize = uniq.iter().map(|w| p.iter().filter(|x| *x == w).count().min(r.iter().filter(|x| *x == w).count())).sum();
        if common == 0 { return 0.0; }
        let (prec, rec) = (common as f64 / p.len() as f64, common as f64 / r.len() as f64);
        2.0 * prec * rec / (prec + rec)
    }

    fn main() {
        let cases = [("1998", "1998"), ("It opened in 1998.", "1998"), ("about 17 km", "12.3 km")];
        for (pred, reference) in cases.iter() {
            let em = if tokens(pred) == tokens(reference) { 1 } else { 0 };
            println!("pred={:<22} EM={} F1={:.3}", format!("'{}'", pred), em, f1(pred, reference));
        }
    }
  `, r`
    pred='1998'                 EM=1 F1=1.000
    pred='It opened in 1998.'   EM=0 F1=0.400
    pred='about 17 km'          EM=0 F1=0.400
  `, "Exact match gives no credit to a correct but verbose answer; token F1 gives partial credit. Neither notices that '17 km' is simply wrong but shares a unit with the reference — which is why faithfulness checks and human or model judges are added."),

  "kv-caching": ex(r`
    layers, d_model, bytes_per = 32, 4096, 2               # fp16
    for n in [1024, 8192, 32768]:
        kv = 2 * layers * n * d_model * bytes_per            # K and V for every layer and position
        print(f"context {n:6d}: KV cache = {kv / 2**30:.2f} GiB")
    T = 1000
    without = T * (T + 1) // 2                               # recompute keys/values for the whole prefix every step
    print(f"generating {T} tokens: {without:,} K/V computations without cache vs {T:,} with cache")
  `, r`
    fn main() {
        let (layers, d_model, bytes_per) = (32u64, 4096u64, 2u64); // fp16
        for n in [1024u64, 8192, 32768] {
            let kv = 2 * layers * n * d_model * bytes_per; // K and V for every layer and position
            println!("context {:6}: KV cache = {:.2} GiB", n, kv as f64 / (1u64 << 30) as f64);
        }
        let t = 1000u64;
        let without = t * (t + 1) / 2; // recompute keys/values for the whole prefix every step
        let commas = |n: u64| { let s = n.to_string(); let mut o = String::new(); for (i, c) in s.chars().enumerate() { if i > 0 && (s.len() - i) % 3 == 0 { o.push(','); } o.push(c); } o };
        println!("generating {} tokens: {} K/V computations without cache vs {} with cache", t, commas(without), commas(t));
    }
  `, r`
    context   1024: KV cache = 0.50 GiB
    context   8192: KV cache = 4.00 GiB
    context  32768: KV cache = 16.00 GiB
    generating 1000 tokens: 500,500 K/V computations without cache vs 1,000 with cache
  `, "Caching each token's keys and values means every new token computes only its own K and V — 1 000 instead of 500 500 computations for 1 000 tokens. The price is memory that grows linearly with context length."),

  quantization: ex(r`
    w = [0.423, -1.27, 0.051, 0.884, -0.317]
    scale = max(abs(v) for v in w) / 127                  # symmetric int8 quantisation
    q = [round(v / scale) for v in w]
    deq = [v * scale for v in q]
    err = max(abs(a - b) for a, b in zip(w, deq))
    print(f"scale = {scale:.6f}")
    print("int8   :", q)
    print("dequant:", ", ".join(f"{v:+.4f}" for v in deq))
    print(f"max abs error = {err:.5f} (half a step = {scale / 2:.5f})")
  `, r`
    fn main() {
        let w = [0.423, -1.27, 0.051, 0.884, -0.317_f64];
        let scale = w.iter().map(|v| v.abs()).fold(0.0, f64::max) / 127.0; // symmetric int8 quantisation
        let q: Vec<i8> = w.iter().map(|v| (v / scale).round() as i8).collect();
        let deq: Vec<f64> = q.iter().map(|&v| v as f64 * scale).collect();
        let err = w.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("scale = {:.6}", scale);
        println!("int8   : {:?}", q);
        println!("dequant: {}", deq.iter().map(|v| format!("{:+.4}", v)).collect::<Vec<_>>().join(", "));
        println!("max abs error = {:.5} (half a step = {:.5})", err, scale / 2.0);
    }
  `, r`
    scale = 0.010000
    int8   : [42, -127, 5, 88, -32]
    dequant: +0.4200, -1.2700, +0.0500, +0.8800, -0.3200
    max abs error = 0.00400 (half a step = 0.00500)
  `, "Symmetric int8 quantisation maps the largest weight to ±127 and rounds everything else onto that grid. Each weight then needs 1 byte instead of 4, and the rounding error is at most half a quantisation step."),

  "context-windows": ex(r`
    window, system, reserve = 4096, 500, 800
    turns = [300] * 12                                       # tokens per past conversation turn
    docs = [600, 600, 600]                                   # retrieved chunks
    budget = window - system - reserve - sum(docs)
    kept, used = [], 0
    for t in reversed(turns):                                # keep the most recent turns that fit
        if used + t > budget: break
        kept.append(t); used += t
    print(f"budget for history = {budget} tokens")
    print(f"kept the last {len(kept)} of {len(turns)} turns ({used} tokens); dropped {len(turns) - len(kept)}")
    print(f"total = {system + sum(docs) + used + reserve} / {window}")
  `, r`
    fn main() {
        let (window, system, reserve) = (4096, 500, 800);
        let turns = [300; 12]; // tokens per past conversation turn
        let docs = [600, 600, 600]; // retrieved chunks
        let budget = window - system - reserve - docs.iter().sum::<i32>();
        let (mut kept, mut used) = (0, 0);
        for t in turns.iter().rev() {
            // keep the most recent turns that fit
            if used + t > budget { break; }
            kept += 1;
            used += t;
        }
        println!("budget for history = {} tokens", budget);
        println!("kept the last {} of {} turns ({} tokens); dropped {}", kept, turns.len(), used, turns.len() - kept);
        println!("total = {} / {}", system + docs.iter().sum::<i32>() + used + reserve, window);
    }
  `, r`
    budget for history = 996 tokens
    kept the last 3 of 12 turns (900 tokens); dropped 9
    total = 4000 / 4096
  `, "Everything — system prompt, retrieved documents, history and the space reserved for the answer — must fit in the window. After the fixed parts, only 996 tokens remain, so just the three most recent turns survive."),
};
