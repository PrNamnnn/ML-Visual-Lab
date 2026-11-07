import type { CodeExample } from "@/types/curriculum";
import { ex, r } from "./helpers";

export const computerVisionCode: Record<string, CodeExample> = {
  "image-representation": ex(r`
    img = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
           [(255, 255, 255), (128, 128, 128), (0, 0, 0)]]     # 2 rows x 3 columns x RGB
    print(f"shape: {len(img)} x {len(img[0])} x {len(img[0][0])}")
    for row in img:
        gray = [round(0.299 * r_ + 0.587 * g + 0.114 * b) for r_, g, b in row]   # luminance
        print("gray:", gray)
  `, r`
    fn main() {
        let img = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], [(255, 255, 255), (128, 128, 128), (0, 0, 0)]]; // 2 rows x 3 columns x RGB
        println!("shape: {} x {} x 3", img.len(), img[0].len());
        for row in img.iter() {
            let gray: Vec<i64> = row.iter().map(|&(r, g, b)| (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64).round() as i64).collect();
            println!("gray: {:?}", gray);
        }
    }
  `, r`
    shape: 2 x 3 x 3
    gray: [76, 150, 29]
    gray: [255, 128, 0]
  `, "A colour image is a height × width × 3 array of numbers. Converting to greyscale weights the channels by perceived brightness: pure green looks much brighter (150) than pure blue (29)."),

  "image-normalization": ex(r`
    pixel = (124, 116, 104)                  # one RGB pixel, values 0-255
    mean = (0.485, 0.456, 0.406)             # ImageNet channel statistics (on the 0-1 scale)
    std = (0.229, 0.224, 0.225)
    scaled = [v / 255 for v in pixel]
    norm = [(v - m) / s for v, m, s in zip(scaled, mean, std)]
    print("scaled to [0,1]:", ", ".join(f"{v:.4f}" for v in scaled))
    print("normalised     :", ", ".join(f"{v:+.4f}" for v in norm))
  `, r`
    fn main() {
        let pixel = [124.0, 116.0, 104.0_f64]; // one RGB pixel, values 0-255
        let mean = [0.485, 0.456, 0.406]; // ImageNet channel statistics (on the 0-1 scale)
        let std = [0.229, 0.224, 0.225];
        let scaled: Vec<f64> = pixel.iter().map(|v| v / 255.0).collect();
        let norm: Vec<f64> = (0..3).map(|c| (scaled[c] - mean[c]) / std[c]).collect();
        println!("scaled to [0,1]: {}", scaled.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));
        println!("normalised     : {}", norm.iter().map(|v| format!("{:+.4}", v)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    scaled to [0,1]: 0.4863, 0.4549, 0.4078
    normalised     : +0.0056, -0.0049, +0.0082
  `, "Pixels are first scaled to [0, 1], then each channel is standardised with the mean and standard deviation of the pretraining data. This pixel is almost exactly 'average', so all three channels end up near 0."),

  "convolution-filters": ex(r`
    img = [[10, 10, 10, 10, 10],
           [10, 50, 50, 50, 10],
           [10, 50, 90, 50, 10],
           [10, 50, 50, 50, 10],
           [10, 10, 10, 10, 10]]
    box = [[1 / 9] * 3 for _ in range(3)]
    sharpen = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    def conv_at(K, i, j):
        return sum(img[i + a - 1][j + b - 1] * K[a][b] for a in range(3) for b in range(3))
    print(f"centre pixel {img[2][2]}: blurred={conv_at(box, 2, 2):.2f} sharpened={conv_at(sharpen, 2, 2):.0f}")
    print(f"pixel (1,1) {img[1][1]}: blurred={conv_at(box, 1, 1):.2f} sharpened={conv_at(sharpen, 1, 1):.0f}")
  `, r`
    fn main() {
        let img = [[10, 10, 10, 10, 10], [10, 50, 50, 50, 10], [10, 50, 90, 50, 10], [10, 50, 50, 50, 10], [10, 10, 10, 10, 10]];
        let bx = [[1.0 / 9.0; 3]; 3];
        let sharpen = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0_f64]];
        let conv_at = |k: &[[f64; 3]; 3], i: usize, j: usize| {
            let mut s = 0.0;
            for a in 0..3 { for b in 0..3 { s += img[i + a - 1][j + b - 1] as f64 * k[a][b]; } }
            s
        };
        println!("centre pixel {}: blurred={:.2} sharpened={:.0}", img[2][2], conv_at(&bx, 2, 2), conv_at(&sharpen, 2, 2));
        println!("pixel (1,1) {}: blurred={:.2} sharpened={:.0}", img[1][1], conv_at(&bx, 1, 1), conv_at(&sharpen, 1, 1));
    }
  `, r`
    centre pixel 90: blurred=54.44 sharpened=250
    pixel (1,1) 50: blurred=32.22 sharpened=130
  `, "The same image under two kernels: the box filter averages each 3×3 neighbourhood (smoothing the bright centre down), while the sharpening kernel amplifies the difference from the neighbours (the centre jumps from 90 to 250)."),

  "edge-detection": ex(r`
    import math
    img = [[0, 0, 0, 9, 9],
           [0, 0, 0, 9, 9],
           [0, 0, 0, 9, 9],
           [0, 0, 0, 9, 9]]                  # a vertical edge between columns 2 and 3
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    for j in range(1, 4):
        gx = sum(img[1 + a - 1][j + b - 1] * Kx[a][b] for a in range(3) for b in range(3))
        gy = sum(img[1 + a - 1][j + b - 1] * Ky[a][b] for a in range(3) for b in range(3))
        print(f"row 1, col {j}: Gx={gx:3d} Gy={gy:3d} |G|={math.hypot(gx, gy):.1f}")
  `, r`
    fn main() {
        let img = [[0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9]]; // a vertical edge between columns 2 and 3
        let kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
        for j in 1..4 {
            let (mut gx, mut gy) = (0, 0);
            for a in 0..3 { for b in 0..3 { gx += img[a][j + b - 1] * kx[a][b]; gy += img[a][j + b - 1] * ky[a][b]; } }
            println!("row 1, col {}: Gx={:3} Gy={:3} |G|={:.1}", j, gx, gy, (gx as f64).hypot(gy as f64));
        }
    }
  `, r`
    row 1, col 1: Gx=  0 Gy=  0 |G|=0.0
    row 1, col 2: Gx= 36 Gy=  0 |G|=36.0
    row 1, col 3: Gx= 36 Gy=  0 |G|=36.0
  `, "The Sobel x-kernel responds strongly where brightness changes from left to right, so the gradient magnitude is large only in the columns next to the edge and 0 in the flat region; G_y is 0 because nothing changes vertically."),

  "image-gradients": ex(r`
    import math
    img = [[1, 2, 4], [2, 4, 8], [4, 8, 16]]
    i, j = 1, 1
    gx = (img[i][j + 1] - img[i][j - 1]) / 2      # central differences
    gy = (img[i + 1][j] - img[i - 1][j]) / 2
    print(f"gx={gx:.1f} gy={gy:.1f} magnitude={math.hypot(gx, gy):.3f} direction={math.degrees(math.atan2(gy, gx)):.1f} deg")
  `, r`
    fn main() {
        let img = [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0_f64]];
        let (i, j) = (1, 1);
        let gx = (img[i][j + 1] - img[i][j - 1]) / 2.0; // central differences
        let gy = (img[i + 1][j] - img[i - 1][j]) / 2.0;
        println!("gx={:.1} gy={:.1} magnitude={:.3} direction={:.1} deg", gx, gy, gx.hypot(gy), gy.atan2(gx).to_degrees());
    }
  `, r`
    gx=3.0 gy=3.0 magnitude=4.243 direction=45.0 deg
  `, "The image gradient at a pixel points in the direction of fastest brightness increase. Brightness grows equally fast rightwards and downwards here, so the gradient points at 45° with magnitude ≈ 4.24."),

  "cnn-feature-extraction": ex(r`
    def conv(img, K):
        n = len(img) - 2
        return [[max(0, sum(img[i + a][j + b] * K[a][b] for a in range(3) for b in range(3))) for j in range(n)] for i in range(n)]
    vertical = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    horizontal = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    bar_v = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
    bar_h = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
    for name, img in [("vertical bar", bar_v), ("horizontal bar", bar_h)]:
        feats = [max(max(row) for row in conv(img, K)) for K in (vertical, horizontal)]   # ReLU + global max pool
        print(f"{name:<15} -> features [vertical={feats[0]}, horizontal={feats[1]}]")
  `, r`
    fn conv(img: &[[i32; 4]; 4], k: &[[i32; 3]; 3]) -> i32 {
        // valid 3x3 convolution, ReLU, then global max pooling
        let mut best = 0;
        for i in 0..2 { for j in 0..2 {
            let mut s = 0;
            for a in 0..3 { for b in 0..3 { s += img[i + a][j + b] * k[a][b]; } }
            best = best.max(s.max(0));
        } }
        best
    }

    fn main() {
        let vertical = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
        let horizontal = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]];
        let bar_v = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]];
        let bar_h = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]];
        for (name, img) in [("vertical bar", bar_v), ("horizontal bar", bar_h)] {
            println!("{:<15} -> features [vertical={}, horizontal={}]", name, conv(&img, &vertical), conv(&img, &horizontal));
        }
    }
  `, r`
    vertical bar    -> features [vertical=3, horizontal=0]
    horizontal bar  -> features [vertical=0, horizontal=3]
  `, "Two filters, ReLU and global max pooling turn each image into a two-number feature vector. The vertical bar excites only the vertical-edge detector and the horizontal bar only the horizontal one — a classifier can separate them from these features alone."),

  iou: ex(r`
    def iou(a, b):                          # boxes as (x1, y1, x2, y2)
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter, union, inter / union
    pred, truth = (2, 2, 6, 6), (3, 3, 7, 8)
    inter, union, v = iou(pred, truth)
    print(f"intersection={inter} union={union} IoU={v:.4f}")
    print("counts as a detection at IoU >= 0.5:", v >= 0.5)
  `, r`
    fn iou(a: [i32; 4], b: [i32; 4]) -> (i32, i32, f64) {
        // boxes as (x1, y1, x2, y2)
        let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0);
        let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0);
        let inter = ix * iy;
        let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        (inter, union, inter as f64 / union as f64)
    }

    fn main() {
        let (inter, union, v) = iou([2, 2, 6, 6], [3, 3, 7, 8]);
        println!("intersection={} union={} IoU={:.4}", inter, union, v);
        println!("counts as a detection at IoU >= 0.5: {}", if v >= 0.5 { "True" } else { "False" });
    }
  `, r`
    intersection=9 union=27 IoU=0.3333
    counts as a detection at IoU >= 0.5: False
  `, "The boxes overlap in a 3 × 3 square, but their union is 27, so IoU is only 0.33 — not enough for the usual 0.5 detection threshold even though the boxes clearly overlap."),

  "non-max-suppression": ex(r`
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    boxes = [(10, 10, 50, 50, 0.9), (12, 12, 52, 52, 0.8), (60, 60, 90, 90, 0.75), (11, 9, 49, 51, 0.6)]
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):          # highest score first
        if all(iou(b, k) <= 0.5 for k in keep):
            keep.append(b)
    for k in keep:
        print("kept", k[:4], "score", k[4])
  `, r`
    fn iou(a: &[f64; 5], b: &[f64; 5]) -> f64 {
        let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let inter = ix * iy;
        inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    }

    fn main() {
        let mut boxes = vec![[10.0, 10.0, 50.0, 50.0, 0.9], [12.0, 12.0, 52.0, 52.0, 0.8], [60.0, 60.0, 90.0, 90.0, 0.75], [11.0, 9.0, 49.0, 51.0, 0.6]];
        boxes.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap()); // highest score first
        let mut keep: Vec<[f64; 5]> = Vec::new();
        for b in boxes.iter() {
            if keep.iter().all(|k| iou(b, k) <= 0.5) { keep.push(*b); }
        }
        for k in keep.iter() {
            println!("kept ({}, {}, {}, {}) score {}", k[0], k[1], k[2], k[3], k[4]);
        }
    }
  `, r`
    kept (10, 10, 50, 50) score 0.9
    kept (60, 60, 90, 90) score 0.75
  `, "NMS keeps the highest-scoring box and removes any box that overlaps a kept one by more than the IoU threshold. Two duplicates of the first object are suppressed; the separate object at (60, 60) survives."),

  "r-cnn-family": ex(r`
    import math
    anchor = (20.0, 20.0, 40.0, 30.0)        # (cx, cy, w, h)
    truth = (24.0, 18.0, 50.0, 27.0)
    t = ((truth[0] - anchor[0]) / anchor[2], (truth[1] - anchor[1]) / anchor[3],
         math.log(truth[2] / anchor[2]), math.log(truth[3] / anchor[3]))
    print("regression targets tx, ty, tw, th =", ", ".join(f"{v:+.4f}" for v in t))
    decoded = (anchor[0] + t[0] * anchor[2], anchor[1] + t[1] * anchor[3], anchor[2] * math.exp(t[2]), anchor[3] * math.exp(t[3]))
    print("decoded box =", ", ".join(f"{v:.1f}" for v in decoded))
  `, r`
    fn main() {
        let anchor: (f64, f64, f64, f64) = (20.0, 20.0, 40.0, 30.0); // (cx, cy, w, h)
        let truth: (f64, f64, f64, f64) = (24.0, 18.0, 50.0, 27.0);
        let t = ((truth.0 - anchor.0) / anchor.2, (truth.1 - anchor.1) / anchor.3, (truth.2 / anchor.2).ln(), (truth.3 / anchor.3).ln());
        println!("regression targets tx, ty, tw, th = {:+.4}, {:+.4}, {:+.4}, {:+.4}", t.0, t.1, t.2, t.3);
        let d = (anchor.0 + t.0 * anchor.2, anchor.1 + t.1 * anchor.3, anchor.2 * t.2.exp(), anchor.3 * t.3.exp());
        println!("decoded box = {:.1}, {:.1}, {:.1}, {:.1}", d.0, d.1, d.2, d.3);
    }
  `, r`
    regression targets tx, ty, tw, th = +0.1000, -0.0667, +0.2231, -0.1054
    decoded box = 24.0, 18.0, 50.0, 27.0
  `, "Detectors in the R-CNN family regress offsets relative to an anchor or proposal, normalised by its size, and log-ratios for width and height. Decoding the targets recovers the ground-truth box exactly."),

  yolo: ex(r`
    S, img = 7, 448
    cell = img / S
    for name, (cx, cy, w, h) in [("dog", (200.0, 100.0, 120.0, 90.0)), ("bike", (330.0, 300.0, 200.0, 150.0))]:
        col, row = int(cx // cell), int(cy // cell)
        print(f"{name}: cell (row={row}, col={col}) x_off={(cx % cell) / cell:.3f} y_off={(cy % cell) / cell:.3f} w={w / img:.3f} h={h / img:.3f}")
    print(f"output tensor: {S} x {S} x (2*5 + 20) = {S * S * 30} numbers")
  `, r`
    fn main() {
        let (s, img) = (7.0_f64, 448.0_f64);
        let cell = img / s;
        for (name, (cx, cy, w, h)) in [("dog", (200.0, 100.0, 120.0, 90.0_f64)), ("bike", (330.0, 300.0, 200.0, 150.0))] {
            let (col, row) = ((cx / cell).floor() as i32, (cy / cell).floor() as i32);
            println!("{}: cell (row={}, col={}) x_off={:.3} y_off={:.3} w={:.3} h={:.3}", name, row, col, (cx % cell) / cell, (cy % cell) / cell, w / img, h / img);
        }
        println!("output tensor: 7 x 7 x (2*5 + 20) = {} numbers", 7 * 7 * 30);
    }
  `, r`
    dog: cell (row=1, col=3) x_off=0.125 y_off=0.562 w=0.268 h=0.201
    bike: cell (row=4, col=5) x_off=0.156 y_off=0.688 w=0.446 h=0.335
    output tensor: 7 x 7 x (2*5 + 20) = 1470 numbers
  `, "YOLO makes the grid cell containing an object's centre responsible for it and predicts the centre offset inside that cell plus the size relative to the whole image — all in one forward pass."),

  "semantic-segmentation": ex(r`
    truth = [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 2], [2, 2, 2, 2]]
    pred = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 0, 2]]
    ious = []
    for c in range(3):
        inter = sum(t == c and p == c for rt, rp in zip(truth, pred) for t, p in zip(rt, rp))
        union = sum(t == c or p == c for rt, rp in zip(truth, pred) for t, p in zip(rt, rp))
        ious.append(inter / union)
        print(f"class {c}: IoU = {inter}/{union} = {inter / union:.3f}")
    print(f"mIoU = {sum(ious) / 3:.3f}")
  `, r`
    fn main() {
        let truth = [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 2], [2, 2, 2, 2]];
        let pred = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 0, 2]];
        let mut total = 0.0;
        for c in 0..3 {
            let (mut inter, mut union) = (0, 0);
            for i in 0..4 { for j in 0..4 {
                let (t, p) = (truth[i][j] == c, pred[i][j] == c);
                if t && p { inter += 1; }
                if t || p { union += 1; }
            } }
            total += inter as f64 / union as f64;
            println!("class {}: IoU = {}/{} = {:.3}", c, inter, union, inter as f64 / union as f64);
        }
        println!("mIoU = {:.3}", total / 3.0);
    }
  `, r`
    class 0: IoU = 4/7 = 0.571
    class 1: IoU = 5/7 = 0.714
    class 2: IoU = 4/5 = 0.800
    mIoU = 0.695
  `, "Segmentation is scored per pixel and per class: IoU compares the predicted and true masks of each class, and mIoU averages over classes so small classes count as much as large ones."),

  "vision-transformers": ex(r`
    img = [[r_ * 4 + c for c in range(4)] for r_ in range(4)]      # a 4x4 'image'
    p = 2
    patches = [[img[i + a][j + b] for a in range(p) for b in range(p)] for i in range(0, 4, p) for j in range(0, 4, p)]
    for k, patch in enumerate(patches):
        print(f"token {k}: {patch}")
    print(f"224x224 image, 16x16 patches -> {(224 // 16) ** 2} tokens of length {16 * 16 * 3}")
  `, r`
    fn main() {
        let img: Vec<Vec<i32>> = (0..4).map(|r| (0..4).map(|c| r * 4 + c).collect()).collect(); // a 4x4 'image'
        let p = 2;
        let mut k = 0;
        for i in (0..4).step_by(p) {
            for j in (0..4).step_by(p) {
                let patch: Vec<i32> = (0..p).flat_map(|a| { let row = &img[i + a]; (0..p).map(move |b| row[j + b]) }).collect();
                println!("token {}: {:?}", k, patch);
                k += 1;
            }
        }
        println!("224x224 image, 16x16 patches -> {} tokens of length {}", (224 / 16) * (224 / 16), 16 * 16 * 3);
    }
  `, r`
    token 0: [0, 1, 4, 5]
    token 1: [2, 3, 6, 7]
    token 2: [8, 9, 12, 13]
    token 3: [10, 11, 14, 15]
    224x224 image, 16x16 patches -> 196 tokens of length 768
  `, "A Vision Transformer cuts the image into non-overlapping patches and flattens each into one token vector. A 224 × 224 RGB image with 16 × 16 patches becomes a sequence of 196 tokens of 768 numbers each."),

  "image-embeddings": ex(r`
    import math
    def unit(v):
        n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]
    imgs = [unit(v) for v in ([0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.2, 0.9])]   # image encoder outputs
    txts = [unit(v) for v in ([1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [0.1, 0.3, 1.0])]   # matching caption encodings
    tau = 0.1
    S = [[sum(a * b for a, b in zip(i, t)) for t in txts] for i in imgs]
    loss = 0.0
    for k, row in enumerate(S):
        z = [s / tau for s in row]; m = max(z)
        loss += -(z[k] - m - math.log(sum(math.exp(v - m) for v in z)))
        print(f"image {k}:", " ".join(f"{s:.3f}" for s in row), "-> best caption", row.index(max(row)))
    print(f"image-to-text InfoNCE loss = {loss / 3:.4f}")
  `, r`
    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn main() {
        let imgs = [unit([0.9, 0.1, 0.2]), unit([0.1, 0.8, 0.3]), unit([0.2, 0.2, 0.9])]; // image encoder outputs
        let txts = [unit([1.0, 0.0, 0.1]), unit([0.0, 1.0, 0.2]), unit([0.1, 0.3, 1.0])]; // matching caption encodings
        let tau = 0.1;
        let mut loss = 0.0;
        for (k, i) in imgs.iter().enumerate() {
            let row: Vec<f64> = txts.iter().map(|t| i.iter().zip(t.iter()).map(|(a, b)| a * b).sum()).collect();
            let z: Vec<f64> = row.iter().map(|s| s / tau).collect();
            let m = z.iter().cloned().fold(f64::MIN, f64::max);
            loss += -(z[k] - m - z.iter().map(|v| (v - m).exp()).sum::<f64>().ln());
            let best = (0..3).fold(0, |b, j| if row[j] > row[b] { j } else { b });
            println!("image {}: {} -> best caption {}", k, row.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>().join(" "), best);
        }
        println!("image-to-text InfoNCE loss = {:.4}", loss / 3.0);
    }
  `, r`
    image 0: 0.987 0.148 0.329 -> best caption 0
    image 1: 0.150 0.980 0.610 -> best caption 1
    image 2: 0.306 0.395 0.990 -> best caption 2
    image-to-text InfoNCE loss = 0.0099
  `, "With normalised embeddings, cosine similarity is a dot product. Each image is most similar to its own caption, and the contrastive (InfoNCE) loss is low; training pushes the diagonal up and everything else down."),
};

export const nlpCode: Record<string, CodeExample> = {
  "text-preprocessing": ex(r`
    import re
    text = "The Cats are RUNNING, and the dogs ran too!"
    stop = {"the", "are", "and", "too"}
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    tokens = [t for t in tokens if t not in stop]
    def stem(w):
        for suf in ("ning", "ing", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w
    print("tokens :", tokens)
    print("stemmed:", [stem(t) for t in tokens])
  `, r`
    fn stem(w: &str) -> String {
        for suf in ["ning", "ing", "s"] {
            if w.ends_with(suf) && w.len() - suf.len() >= 3 { return w[..w.len() - suf.len()].to_string(); }
        }
        w.to_string()
    }

    fn main() {
        let text = "The Cats are RUNNING, and the dogs ran too!";
        let stop = ["the", "are", "and", "too"];
        let cleaned: String = text.to_lowercase().chars().map(|c| if c.is_ascii_lowercase() || c.is_whitespace() { c } else { ' ' }).collect();
        let tokens: Vec<&str> = cleaned.split_whitespace().filter(|t| !stop.contains(t)).collect();
        let show = |v: &[String]| format!("[{}]", v.iter().map(|t| format!("'{}'", t)).collect::<Vec<_>>().join(", "));
        println!("tokens : {}", show(&tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>()));
        println!("stemmed: {}", show(&tokens.iter().map(|t| stem(t)).collect::<Vec<_>>()));
    }
  `, r`
    tokens : ['cats', 'running', 'dogs', 'ran']
    stemmed: ['cat', 'run', 'dog', 'ran']
  `, "Lowercasing, removing punctuation and stop words, and stripping suffixes merge surface variants: 'RUNNING' becomes 'run' and 'Cats' becomes 'cat'. Irregular forms like 'ran' are not merged — stemmers are crude."),

  tokenization: ex(r`
    from collections import Counter
    words = ["low", "lower", "lowest", "newer", "wider"]
    seqs = [list(w) + ["_"] for w in words]          # characters plus an end-of-word marker
    for step in range(4):                             # learn 4 BPE merges
        pairs = Counter((s[i], s[i + 1]) for s in seqs for i in range(len(s) - 1))
        best = max(sorted(pairs), key=lambda p: pairs[p])   # most frequent pair (ties: alphabetical)
        print(f"merge {step + 1}: {best[0]} + {best[1]} (count {pairs[best]})")
        merged = best[0] + best[1]
        new = []
        for s in seqs:
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(merged); i += 2
                else:
                    out.append(s[i]); i += 1
            new.append(out)
        seqs = new
    print("lowest ->", seqs[2])
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let words = ["low", "lower", "lowest", "newer", "wider"];
        // characters plus an end-of-word marker
        let mut seqs: Vec<Vec<String>> = words.iter().map(|w| { let mut v: Vec<String> = w.chars().map(|c| c.to_string()).collect(); v.push("_".to_string()); v }).collect();
        for step in 0..4 {
            // learn 4 BPE merges
            let mut pairs: BTreeMap<(String, String), usize> = BTreeMap::new();
            for s in seqs.iter() { for i in 0..s.len() - 1 { *pairs.entry((s[i].clone(), s[i + 1].clone())).or_insert(0) += 1; } }
            let mut best = pairs.keys().next().unwrap().clone(); // most frequent pair (ties: alphabetical)
            for (p, c) in pairs.iter() { if *c > pairs[&best] { best = p.clone(); } }
            println!("merge {}: {} + {} (count {})", step + 1, best.0, best.1, pairs[&best]);
            let merged = format!("{}{}", best.0, best.1);
            seqs = seqs.iter().map(|s| {
                let (mut out, mut i) = (Vec::new(), 0);
                while i < s.len() {
                    if i + 1 < s.len() && s[i] == best.0 && s[i + 1] == best.1 { out.push(merged.clone()); i += 2; } else { out.push(s[i].clone()); i += 1; }
                }
                out
            }).collect();
        }
        println!("lowest -> [{}]", seqs[2].iter().map(|t| format!("'{}'", t)).collect::<Vec<_>>().join(", "));
    }
  `, r`
    merge 1: e + r (count 3)
    merge 2: er + _ (count 3)
    merge 3: l + o (count 3)
    merge 4: lo + w (count 3)
    lowest -> ['low', 'e', 's', 't', '_']
  `, "Byte-pair encoding starts from characters and repeatedly merges the most frequent adjacent pair. Frequent fragments like 'low' and 'er_' become single tokens, so common words need few tokens and rare words are spelled from pieces."),

  vocabulary: ex(r`
    from collections import Counter
    corpus = "the cat sat on the mat the dog sat on the log".split()
    vocab = ["[PAD]", "[UNK]"] + [w for w, _ in sorted(Counter(corpus).items(), key=lambda x: (-x[1], x[0]))[:4]]
    ids = {w: i for i, w in enumerate(vocab)}
    sentence = "the cat ate the fish".split()
    encoded = [ids.get(w, ids["[UNK]"]) for w in sentence]
    print("vocab:", vocab)
    print("encoded:", encoded)
    print(f"OOV rate: {sum(w not in ids for w in sentence) / len(sentence):.0%}")
  `, r`
    use std::collections::BTreeMap;

    fn main() {
        let corpus: Vec<&str> = "the cat sat on the mat the dog sat on the log".split(' ').collect();
        let mut counts: BTreeMap<&str, i32> = BTreeMap::new();
        for w in corpus.iter() { *counts.entry(w).or_insert(0) += 1; }
        let mut sorted: Vec<(&str, i32)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let mut vocab = vec!["[PAD]", "[UNK]"];
        vocab.extend(sorted.iter().take(4).map(|x| x.0));
        let sentence = ["the", "cat", "ate", "the", "fish"];
        let id = |w: &str| vocab.iter().position(|v| *v == w).unwrap_or(1);
        let encoded: Vec<usize> = sentence.iter().map(|w| id(w)).collect();
        let oov = sentence.iter().filter(|w| !vocab.contains(w)).count();
        println!("vocab: [{}]", vocab.iter().map(|v| format!("'{}'", v)).collect::<Vec<_>>().join(", "));
        println!("encoded: {:?}", encoded);
        println!("OOV rate: {:.0}%", 100.0 * oov as f64 / sentence.len() as f64);
    }
  `, r`
    vocab: ['[PAD]', '[UNK]', 'the', 'on', 'sat', 'cat']
    encoded: [2, 5, 1, 2, 1]
    OOV rate: 40%
  `, "The vocabulary keeps special tokens and the most frequent words; everything else maps to [UNK] (id 1). With only four real words, 'ate' and 'fish' are unknown — a 40 % out-of-vocabulary rate on this sentence."),

  "word-embeddings": ex(r`
    import math
    E = {"king": [0.8, 0.9, 0.1, 0.0], "queen": [0.8, 0.1, 0.9, 0.0], "man": [0.2, 0.9, 0.1, 0.0], "woman": [0.2, 0.1, 0.9, 0.0], "apple": [0.0, 0.1, 0.1, 0.9]}
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    print(f"cos(king, queen) = {cos(E['king'], E['queen']):.3f}")
    print(f"cos(king, apple) = {cos(E['king'], E['apple']):.3f}")
    target = [k - m + w for k, m, w in zip(E["king"], E["man"], E["woman"])]
    best = max((w for w in E if w not in ("king", "man", "woman")), key=lambda w: cos(E[w], target))
    print("king - man + woman ~", best)
  `, r`
    fn cos(a: &[f64; 4], b: &[f64; 4]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|y| y * y).sum::<f64>().sqrt())
    }

    fn main() {
        let e = [("king", [0.8, 0.9, 0.1, 0.0]), ("queen", [0.8, 0.1, 0.9, 0.0]), ("man", [0.2, 0.9, 0.1, 0.0]), ("woman", [0.2, 0.1, 0.9, 0.0]), ("apple", [0.0, 0.1, 0.1, 0.9_f64])];
        let get = |w: &str| e.iter().find(|p| p.0 == w).unwrap().1;
        println!("cos(king, queen) = {:.3}", cos(&get("king"), &get("queen")));
        println!("cos(king, apple) = {:.3}", cos(&get("king"), &get("apple")));
        let (k, m, w) = (get("king"), get("man"), get("woman"));
        let target = [k[0] - m[0] + w[0], k[1] - m[1] + w[1], k[2] - m[2] + w[2], k[3] - m[3] + w[3]];
        let best = e.iter().filter(|p| !["king", "man", "woman"].contains(&p.0)).max_by(|a, b| cos(&a.1, &target).partial_cmp(&cos(&b.1, &target)).unwrap()).unwrap().0;
        println!("king - man + woman ~ {}", best);
    }
  `, r`
    cos(king, queen) = 0.562
    cos(king, apple) = 0.091
    king - man + woman ~ queen
  `, "Embeddings place words as vectors so that related words have high cosine similarity. With dimensions loosely meaning 'royal', 'male', 'female' and 'fruit', king and queen are far more similar than king and apple, and vector arithmetic king − man + woman lands on queen. (Hand-made 4-D vectors; real ones are learned and have hundreds of dimensions.)"),

  "cbow-skipgram": ex(r`
    sentence = "the quick brown fox jumps".split()
    window = 1
    print("skip-gram (centre -> context):")
    for i, w in enumerate(sentence):
        ctx = [sentence[j] for j in range(max(0, i - window), min(len(sentence), i + window + 1)) if j != i]
        print("  " + ", ".join(f"({w} -> {c})" for c in ctx))
    print("CBOW (context -> centre):")
    for i, w in enumerate(sentence):
        ctx = [sentence[j] for j in range(max(0, i - window), min(len(sentence), i + window + 1)) if j != i]
        print(f"  {ctx} -> {w}")
  `, r`
    fn main() {
        let sentence = ["the", "quick", "brown", "fox", "jumps"];
        let window = 1usize;
        let ctx = |i: usize| -> Vec<&str> {
            (i.saturating_sub(window)..(i + window + 1).min(sentence.len())).filter(|&j| j != i).map(|j| sentence[j]).collect()
        };
        println!("skip-gram (centre -> context):");
        for (i, w) in sentence.iter().enumerate() {
            println!("  {}", ctx(i).iter().map(|c| format!("({} -> {})", w, c)).collect::<Vec<_>>().join(", "));
        }
        println!("CBOW (context -> centre):");
        for (i, w) in sentence.iter().enumerate() {
            println!("  [{}] -> {}", ctx(i).iter().map(|c| format!("'{}'", c)).collect::<Vec<_>>().join(", "), w);
        }
    }
  `, r`
    skip-gram (centre -> context):
      (the -> quick)
      (quick -> the), (quick -> brown)
      (brown -> quick), (brown -> fox)
      (fox -> brown), (fox -> jumps)
      (jumps -> fox)
    CBOW (context -> centre):
      ['quick'] -> the
      ['the', 'brown'] -> quick
      ['quick', 'fox'] -> brown
      ['brown', 'jumps'] -> fox
      ['fox'] -> jumps
  `, "The same window produces two kinds of training examples: skip-gram predicts each context word from the centre word, CBOW predicts the centre word from all its context words together."),

  glove: ex(r`
    import math
    from collections import defaultdict
    text = "ice is cold ice is solid steam is hot steam is gas".split()
    X = defaultdict(float)
    for i, w in enumerate(text):                     # symmetric window of 2, weighted by 1/distance
        for j in range(max(0, i - 2), min(len(text), i + 3)):
            if j != i: X[(w, text[j])] += 1 / abs(i - j)
    for pair in [("ice", "cold"), ("ice", "solid"), ("steam", "hot"), ("ice", "hot")]:
        print(f"X{pair} = {X[pair]:.2f}")
    f = lambda x, xmax=10, a=0.75: min(1.0, (x / xmax) ** a)
    x = X[("ice", "cold")]
    print(f"weight f(X) = {f(x):.4f}, regression target log X = {math.log(x):.4f}")
  `, r`
    use std::collections::HashMap;

    fn main() {
        let text: Vec<&str> = "ice is cold ice is solid steam is hot steam is gas".split(' ').collect();
        let mut x: HashMap<(&str, &str), f64> = HashMap::new();
        for i in 0..text.len() {
            // symmetric window of 2, weighted by 1/distance
            for j in i.saturating_sub(2)..(i + 3).min(text.len()) {
                if j != i { *x.entry((text[i], text[j])).or_insert(0.0) += 1.0 / (i as f64 - j as f64).abs(); }
            }
        }
        for pair in [("ice", "cold"), ("ice", "solid"), ("steam", "hot"), ("ice", "hot")] {
            println!("X('{}', '{}') = {:.2}", pair.0, pair.1, x.get(&pair).copied().unwrap_or(0.0));
        }
        let f = |v: f64| (v / 10.0).powf(0.75).min(1.0);
        let v = x[&("ice", "cold")];
        println!("weight f(X) = {:.4}, regression target log X = {:.4}", f(v), v.ln());
    }
  `, r`
    X('ice', 'cold') = 1.50
    X('ice', 'solid') = 0.50
    X('steam', 'hot') = 1.50
    X('ice', 'hot') = 0.00
    weight f(X) = 0.2410, regression target log X = 0.4055
  `, "GloVe starts from distance-weighted co-occurrence counts: 'ice' appears near 'cold' and 'solid' but never near 'hot'. Each non-zero count becomes a weighted regression target log X for the dot product of two word vectors."),

  "positional-encoding": ex(r`
    import math
    d = 4
    for pos in range(4):
        pe = []
        for i in range(0, d, 2):
            angle = pos / 10000 ** (i / d)
            pe += [math.sin(angle), math.cos(angle)]
        print(f"pos {pos}:", ", ".join(f"{v:+.4f}" for v in pe))
  `, r`
    fn main() {
        let d = 4;
        for pos in 0..4 {
            let mut pe = Vec::new();
            for i in (0..d).step_by(2) {
                let angle = pos as f64 / 10000f64.powf(i as f64 / d as f64);
                pe.push(angle.sin());
                pe.push(angle.cos());
            }
            println!("pos {}: {}", pos, pe.iter().map(|v| format!("{:+.4}", v)).collect::<Vec<_>>().join(", "));
        }
    }
  `, r`
    pos 0: +0.0000, +1.0000, +0.0000, +1.0000
    pos 1: +0.8415, +0.5403, +0.0100, +1.0000
    pos 2: +0.9093, -0.4161, +0.0200, +0.9998
    pos 3: +0.1411, -0.9900, +0.0300, +0.9996
  `, "Each position gets a fixed vector of sines and cosines at geometrically spaced frequencies: the first pair changes quickly with position, the second slowly. Added to token embeddings, it tells the order-blind attention where each token is."),
};

export const transformersCode: Record<string, CodeExample> = {
  "self-attention": ex(r`
    import math
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]          # 3 tokens, d = 2 (Q = K = V = X for clarity)
    d = 2
    out = []
    for q in X:
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d) for k in X]
        m = max(scores); e = [math.exp(s - m) for s in scores]; w = [v / sum(e) for v in e]
        out.append([sum(w[j] * X[j][c] for j in range(3)) for c in range(d)])
        print("weights:", ", ".join(f"{v:.3f}" for v in w), "-> output", ", ".join(f"{v:.3f}" for v in out[-1]))
  `, r`
    fn main() {
        let x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0_f64]]; // 3 tokens, d = 2 (Q = K = V = X for clarity)
        let d = 2.0_f64;
        for q in x.iter() {
            let scores: Vec<f64> = x.iter().map(|k| (q[0] * k[0] + q[1] * k[1]) / d.sqrt()).collect();
            let m = scores.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            let w: Vec<f64> = e.iter().map(|v| v / sum).collect();
            let out: Vec<f64> = (0..2).map(|c| (0..3).map(|j| w[j] * x[j][c]).sum()).collect();
            println!("weights: {} -> output {}", w.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "), out.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "));
        }
    }
  `, r`
    weights: 0.401, 0.198, 0.401 -> output 0.802, 0.599
    weights: 0.198, 0.401, 0.401 -> output 0.599, 0.802
    weights: 0.248, 0.248, 0.503 -> output 0.752, 0.752
  `, "Each token compares its query with every key, turns the scores into weights with softmax, and outputs the weighted mix of values. Every output is a context-dependent blend of the whole sequence."),

  "scaled-dot-product-attention": ex(r`
    import math
    def softmax(z):
        m = max(z); e = [math.exp(v - m) for v in z]; return [v / sum(e) for v in e]
    d = 64
    raw = [12.0, 8.0, 4.0]                    # q.k grows like sqrt(d) for random unit-variance vectors
    print("without scaling:", ", ".join(f"{p:.5f}" for p in softmax(raw)))
    print("scaled by sqrt(d):", ", ".join(f"{p:.5f}" for p in softmax([s / math.sqrt(d) for s in raw])))
  `, r`
    fn softmax(z: &[f64]) -> Vec<f64> {
        let m = z.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    fn main() {
        let d = 64.0_f64;
        let raw = [12.0, 8.0, 4.0]; // q.k grows like sqrt(d) for random unit-variance vectors
        let show = |p: Vec<f64>| p.iter().map(|v| format!("{:.5}", v)).collect::<Vec<_>>().join(", ");
        println!("without scaling: {}", show(softmax(&raw)));
        println!("scaled by sqrt(d): {}", show(softmax(&raw.iter().map(|s| s / d.sqrt()).collect::<Vec<_>>())));
    }
  `, r`
    without scaling: 0.98169, 0.01798, 0.00033
    scaled by sqrt(d): 0.50648, 0.30720, 0.18632
  `, "Dot products of d-dimensional vectors grow with √d. Unscaled, the softmax becomes almost one-hot (98 % on one key), which gives vanishing gradients; dividing by √d = 8 keeps the distribution soft and trainable."),

  "multi-head-attention": ex(r`
    import math
    def attend(Q, K, V):
        d = len(Q[0]); out = []
        for q in Q:
            s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d) for k in K]
            m = max(s); e = [math.exp(v - m) for v in s]; w = [v / sum(e) for v in e]
            out.append([sum(w[j] * V[j][c] for j in range(len(V))) for c in range(len(V[0]))])
        return out
    X = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    heads = [attend([x[h * 2:h * 2 + 2] for x in X], [x[h * 2:h * 2 + 2] for x in X], [x[h * 2:h * 2 + 2] for x in X]) for h in range(2)]
    for t in range(3):                         # concatenate the two heads' outputs
        print(f"token {t}:", ", ".join(f"{v:.3f}" for v in heads[0][t] + heads[1][t]))
  `, r`
    fn attend(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // single-head attention with Q = K = V = x
        let d = x[0].len() as f64;
        x.iter().map(|q| {
            let s: Vec<f64> = x.iter().map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() / d.sqrt()).collect();
            let m = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = e.iter().sum();
            (0..x[0].len()).map(|c| (0..x.len()).map(|j| e[j] / z * x[j][c]).sum()).collect()
        }).collect()
    }

    fn main() {
        let x = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0, 0.0]];
        let heads: Vec<Vec<Vec<f64>>> = (0..2).map(|h| attend(&x.iter().map(|t| t[h * 2..h * 2 + 2].to_vec()).collect::<Vec<_>>())).collect();
        for t in 0..3 {
            // concatenate the two heads' outputs
            let cat: Vec<String> = heads[0][t].iter().chain(heads[1][t].iter()).map(|v| format!("{:.3}", v)).collect();
            println!("token {}: {}", t, cat.join(", "));
        }
    }
  `, r`
    token 0: 0.802, 0.599, 0.248, 0.503
    token 1: 0.599, 0.802, 0.503, 0.248
    token 2: 0.752, 0.752, 0.333, 0.333
  `, "The 4-dimensional embedding is split into two heads of size 2; each head attends independently over its own subspace, so the heads can focus on different relationships, and their outputs are concatenated."),

  "cross-attention": ex(r`
    import math
    encoder = {"le": [1.0, 0.0], "chat": [0.0, 1.0], "noir": [0.7, 0.7]}    # source tokens (keys = values)
    query = [0.1, 1.0]                                                        # decoder state while writing "cat"
    s = {w: sum(a * b for a, b in zip(query, k)) / math.sqrt(2) for w, k in encoder.items()}
    m = max(s.values()); e = {w: math.exp(v - m) for w, v in s.items()}
    z = sum(e.values())
    for w in encoder:
        print(f"attention to '{w}': {e[w] / z:.3f}")
  `, r`
    fn main() {
        let encoder = [("le", [1.0, 0.0]), ("chat", [0.0, 1.0]), ("noir", [0.7, 0.7_f64])]; // source tokens (keys = values)
        let query = [0.1, 1.0_f64]; // decoder state while writing "cat"
        let s: Vec<f64> = encoder.iter().map(|(_, k)| (query[0] * k[0] + query[1] * k[1]) / 2f64.sqrt()).collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for (i, (w, _)) in encoder.iter().enumerate() {
            println!("attention to '{}': {:.3}", w, e[i] / z);
        }
    }
  `, r`
    attention to 'le': 0.222
    attention to 'chat': 0.420
    attention to 'noir': 0.357
  `, "In cross-attention the query comes from the decoder and the keys/values from the encoder. While producing 'cat', the decoder's query aligns best with the source word 'chat'."),

  "transformer-architecture": ex(r`
    d, ff, layers, vocab = 512, 2048, 6, 32000
    attn = 4 * (d * d + d)                    # W_Q, W_K, W_V, W_O with biases
    mlp = d * ff + ff + ff * d + d
    norms = 2 * 2 * d                         # two LayerNorms (gamma, beta)
    block = attn + mlp + norms
    print(f"attention {attn:,} + MLP {mlp:,} + norms {norms:,} = {block:,} per block")
    print(f"{layers} blocks = {layers * block:,}; embeddings = {vocab * d:,}")
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
        let (d, ff, layers, vocab) = (512u64, 2048u64, 6u64, 32000u64);
        let attn = 4 * (d * d + d); // W_Q, W_K, W_V, W_O with biases
        let mlp = d * ff + ff + ff * d + d;
        let norms = 2 * 2 * d; // two LayerNorms (gamma, beta)
        let block = attn + mlp + norms;
        println!("attention {} + MLP {} + norms {} = {} per block", commas(attn), commas(mlp), commas(norms), commas(block));
        println!("{} blocks = {}; embeddings = {}", layers, commas(layers * block), commas(vocab * d));
    }
  `, r`
    attention 1,050,624 + MLP 2,099,712 + norms 2,048 = 3,152,384 per block
    6 blocks = 18,914,304; embeddings = 16,384,000
  `, "A Transformer block is multi-head attention plus a feed-forward MLP, each wrapped with a residual connection and layer norm. The MLP holds about twice the attention's parameters; the token-embedding table alone (16.4 M parameters) is nearly as large as all six blocks together (18.9 M)."),

  "autoregressive-generation": ex(r`
    nxt = {"<s>": {"I": 0.6, "you": 0.4}, "I": {"like": 0.7, "see": 0.3}, "like": {"tea": 0.5, "rain": 0.3, "</s>": 0.2},
           "tea": {"</s>": 0.9, "and": 0.1}, "you": {"like": 0.8, "</s>": 0.2}}
    tok, out, logp = "<s>", [], 0.0
    import math
    while tok != "</s>" and len(out) < 10:   # each step conditions on everything generated so far
        tok = max(nxt[tok], key=nxt[tok].get)  # greedy: take the most likely next token
        logp += math.log(nxt[out[-1] if out else "<s>"][tok])
        out.append(tok)
    print("generated:", " ".join(out))
    print(f"sequence probability = {math.exp(logp):.4f}")
  `, r`
    fn main() {
        let nxt = |t: &str| -> Vec<(&'static str, f64)> {
            match t {
                "<s>" => vec![("I", 0.6), ("you", 0.4)],
                "I" => vec![("like", 0.7), ("see", 0.3)],
                "like" => vec![("tea", 0.5), ("rain", 0.3), ("</s>", 0.2)],
                "tea" => vec![("</s>", 0.9), ("and", 0.1)],
                _ => vec![("like", 0.8), ("</s>", 0.2)],
            }
        };
        let (mut tok, mut out, mut logp) = ("<s>", Vec::new(), 0.0_f64);
        while tok != "</s>" && out.len() < 10 {
            // each step conditions on everything generated so far; greedy picks the most likely token
            let (best, p) = nxt(tok).into_iter().fold(("", 0.0), |a, x| if x.1 > a.1 { x } else { a });
            logp += p.ln();
            tok = best;
            out.push(best);
        }
        println!("generated: {}", out.join(" "));
        println!("sequence probability = {:.4}", logp.exp());
    }
  `, r`
    generated: I like tea </s>
    sequence probability = 0.1890
  `, "An autoregressive model generates one token at a time, feeding each choice back in as context; the sequence probability is the product of the step probabilities (0.6 × 0.7 × 0.5 × 0.9)."),

  "sampling-strategies": ex(r`
    import math
    words = ["blue", "green", "red", "violet", "plaid"]
    logits = [2.0, 1.5, 1.0, 0.0, -1.0]
    def softmax(z, T=1.0):
        e = [math.exp(v / T) for v in z]; return [v / sum(e) for v in e]
    for T in [0.5, 1.0, 2.0]:
        print(f"T={T}:", ", ".join(f"{w}={p:.3f}" for w, p in zip(words, softmax(logits, T))))
    p = softmax(logits)
    top2 = sorted(range(5), key=lambda i: -p[i])[:2]                 # top-k with k = 2
    print("top-k (k=2):", ", ".join(f"{words[i]}={p[i] / sum(p[j] for j in top2):.3f}" for i in top2))
    order, cum, nucleus = sorted(range(5), key=lambda i: -p[i]), 0.0, []
    for i in order:                                                   # top-p (nucleus) with p = 0.8
        nucleus.append(i); cum += p[i]
        if cum >= 0.8: break
    print("top-p (p=0.8) keeps:", [words[i] for i in nucleus])
  `, r`
    fn softmax(z: &[f64], t: f64) -> Vec<f64> {
        let e: Vec<f64> = z.iter().map(|v| (v / t).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    fn main() {
        let words = ["blue", "green", "red", "violet", "plaid"];
        let logits = [2.0, 1.5, 1.0, 0.0, -1.0];
        for t in [0.5, 1.0, 2.0] {
            let p = softmax(&logits, t);
            println!("T={:.1}: {}", t, words.iter().zip(p.iter()).map(|(w, q)| format!("{}={:.3}", w, q)).collect::<Vec<_>>().join(", "));
        }
        let p = softmax(&logits, 1.0);
        let mut order: Vec<usize> = (0..5).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let z = p[order[0]] + p[order[1]]; // top-k with k = 2
        println!("top-k (k=2): {}", order[..2].iter().map(|&i| format!("{}={:.3}", words[i], p[i] / z)).collect::<Vec<_>>().join(", "));
        let (mut cum, mut nucleus) = (0.0, Vec::new());
        for &i in order.iter() {
            // top-p (nucleus) with p = 0.8
            nucleus.push(format!("'{}'", words[i]));
            cum += p[i];
            if cum >= 0.8 { break; }
        }
        println!("top-p (p=0.8) keeps: [{}]", nucleus.join(", "));
    }
  `, r`
    T=0.5: blue=0.656, green=0.241, red=0.089, violet=0.012, plaid=0.002
    T=1.0: blue=0.463, green=0.281, red=0.170, violet=0.063, plaid=0.023
    T=2.0: blue=0.336, green=0.262, red=0.204, violet=0.124, plaid=0.075
    top-k (k=2): blue=0.622, green=0.378
    top-p (p=0.8) keeps: ['blue', 'green', 'red']
  `, "Temperature sharpens (T < 1) or flattens (T > 1) the distribution. Top-k keeps a fixed number of candidates and renormalises; top-p keeps the smallest set whose probability reaches p, so it adapts to how confident the model is."),

  "beam-search": ex(r`
    import math
    LM = {"<s>": {"the": 0.5, "a": 0.4, "one": 0.1}, "the": {"dog": 0.4, "cat": 0.35, "end": 0.25}, "a": {"very": 0.9, "dog": 0.1},
          "one": {"cat": 0.9, "dog": 0.1}, "very": {"good": 0.95, "bad": 0.05}, "dog": {"barks": 0.6, "</s>": 0.4},
          "cat": {"sleeps": 0.7, "</s>": 0.3}, "good": {"dog": 0.8, "</s>": 0.2}, "bad": {"</s>": 1.0}, "end": {"</s>": 1.0},
          "barks": {"</s>": 1.0}, "sleeps": {"</s>": 1.0}}
    def search(k):
        beams = [(["<s>"], 0.0)]
        for _ in range(6):
            cand = []
            for seq, lp in beams:
                if seq[-1] == "</s>": cand.append((seq, lp)); continue
                for w, p in LM[seq[-1]].items(): cand.append((seq + [w], lp + math.log(p)))
            beams = sorted(cand, key=lambda c: -c[1])[:k]
        return beams[0]
    for k in [1, 2]:
        seq, lp = search(k)
        print(f"beam {k}: {' '.join(seq[1:-1])} (p={math.exp(lp):.4f})")
  `, r`
    fn lm(t: &str) -> Vec<(&'static str, f64)> {
        match t {
            "<s>" => vec![("the", 0.5), ("a", 0.4), ("one", 0.1)],
            "the" => vec![("dog", 0.4), ("cat", 0.35), ("end", 0.25)],
            "a" => vec![("very", 0.9), ("dog", 0.1)],
            "one" => vec![("cat", 0.9), ("dog", 0.1)],
            "very" => vec![("good", 0.95), ("bad", 0.05)],
            "dog" => vec![("barks", 0.6), ("</s>", 0.4)],
            "cat" => vec![("sleeps", 0.7), ("</s>", 0.3)],
            "good" => vec![("dog", 0.8), ("</s>", 0.2)],
            _ => vec![("</s>", 1.0)],
        }
    }

    fn search(k: usize) -> (Vec<&'static str>, f64) {
        let mut beams: Vec<(Vec<&'static str>, f64)> = vec![(vec!["<s>"], 0.0)];
        for _ in 0..6 {
            let mut cand = Vec::new();
            for (seq, lp) in beams.iter() {
                let last = *seq.last().unwrap();
                if last == "</s>" { cand.push((seq.clone(), *lp)); continue; }
                for (w, p) in lm(last) { let mut s = seq.clone(); s.push(w); cand.push((s, lp + p.ln())); }
            }
            cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            cand.truncate(k);
            beams = cand;
        }
        beams[0].clone()
    }

    fn main() {
        for k in [1, 2] {
            let (seq, lp) = search(k);
            println!("beam {}: {} (p={:.4})", k, seq[1..seq.len() - 1].join(" "), lp.exp());
        }
    }
  `, r`
    beam 1: the dog barks (p=0.1200)
    beam 2: a very good dog barks (p=0.1642)
  `, "Greedy decoding (beam 1) commits to 'the' because it is the most likely first word, ending with probability 0.12. Keeping two hypotheses finds 'a very good dog barks', which is more likely overall (0.164) despite its less likely first word."),

  "fine-tuning": ex(r`
    xs, ys = [1.0, 2.0, 3.0], [3.1, 5.9, 9.2]       # small new task: y is about 3x
    w_pre = 2.0                                     # weight from pre-training (old task: y = 2x)
    for lam in [0.0, 1.0]:                          # lam > 0: L2-SP pull towards the pretrained weight
        w = w_pre
        for _ in range(200):
            g = sum(2 * (w * x - y) * x for x, y in zip(xs, ys)) / len(xs) + lam * (w - w_pre)
            w -= 0.02 * g
        print(f"lambda={lam:.1f}: fine-tuned w={w:.4f} (moved {w - w_pre:+.4f} from pre-trained)")
  `, r`
    fn main() {
        let (xs, ys) = ([1.0, 2.0, 3.0_f64], [3.1, 5.9, 9.2_f64]); // small new task: y is about 3x
        let w_pre = 2.0; // weight from pre-training (old task: y = 2x)
        for lam in [0.0, 1.0] {
            // lam > 0: L2-SP pull towards the pretrained weight
            let mut w = w_pre;
            for _ in 0..200 {
                let g = xs.iter().zip(ys.iter()).map(|(x, y)| 2.0 * (w * x - y) * x).sum::<f64>() / 3.0 + lam * (w - w_pre);
                w -= 0.02 * g;
            }
            println!("lambda={:.1}: fine-tuned w={:.4} (moved {:+.4} from pre-trained)", lam, w, w - w_pre);
        }
    }
  `, r`
    lambda=0.0: fine-tuned w=3.0357 (moved +1.0357 from pre-trained)
    lambda=1.0: fine-tuned w=2.9355 (moved +0.9355 from pre-trained)
  `, "Fine-tuning starts from the pretrained weight and adapts it to the new data. A penalty towards the pretrained value (L2-SP) limits how far it moves — a simple guard against forgetting the original task."),

  lora: ex(r`
    d, r_ = 4096, 8
    full = d * d
    lora_params = d * r_ + r_ * d
    print(f"full fine-tune of one {d}x{d} matrix: {full:,} parameters")
    print(f"LoRA rank {r_}: {lora_params:,} parameters ({lora_params / full:.2%})")
    # tiny numerical example: W' = W + B A with rank-1 update
    W = [[1.0, 0.0], [0.0, 1.0]]
    B, A = [[0.5], [-0.5]], [[2.0, 1.0]]
    Wn = [[W[i][j] + B[i][0] * A[0][j] for j in range(2)] for i in range(2)]
    print("W + BA =", Wn)
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
        let (d, r) = (4096u64, 8u64);
        let full = d * d;
        let lora_params = d * r + r * d;
        println!("full fine-tune of one {}x{} matrix: {} parameters", d, d, commas(full));
        println!("LoRA rank {}: {} parameters ({:.2}%)", r, commas(lora_params), 100.0 * lora_params as f64 / full as f64);
        // tiny numerical example: W' = W + B A with rank-1 update
        let w = [[1.0, 0.0], [0.0, 1.0_f64]];
        let (b, a) = ([0.5, -0.5_f64], [2.0, 1.0_f64]);
        let wn: Vec<String> = (0..2).map(|i| format!("[{:?}, {:?}]", w[i][0] + b[i] * a[0], w[i][1] + b[i] * a[1])).collect();
        println!("W + BA = [{}]", wn.join(", "));
    }
  `, r`
    full fine-tune of one 4096x4096 matrix: 16,777,216 parameters
    LoRA rank 8: 65,536 parameters (0.39%)
    W + BA = [[2.0, 0.5], [-1.0, 0.5]]
  `, "LoRA freezes W and learns a low-rank update BA. For a 4096 × 4096 matrix a rank-8 update needs 0.39 % of the parameters, and after training BA can be merged into W so inference costs nothing extra."),

  "knowledge-distillation": ex(r`
    import math
    def softmax(z, T):
        e = [math.exp(v / T) for v in z]; return [v / sum(e) for v in e]
    teacher, student, label, T, alpha = [5.0, 2.0, 0.5], [3.0, 2.5, 0.0], 0, 4.0, 0.5
    pt, ps = softmax(teacher, T), softmax(student, T)
    kl = sum(p * math.log(p / q) for p, q in zip(pt, ps))
    ce = -math.log(softmax(student, 1.0)[label])
    print("teacher soft targets:", ", ".join(f"{v:.3f}" for v in pt))
    print("student at T       :", ", ".join(f"{v:.3f}" for v in ps))
    print(f"loss = {alpha}*CE {ce:.4f} + {1 - alpha}*T^2*KL {T * T * kl:.4f} = {alpha * ce + (1 - alpha) * T * T * kl:.4f}")
  `, r`
    fn softmax(z: &[f64], t: f64) -> Vec<f64> {
        let e: Vec<f64> = z.iter().map(|v| (v / t).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    fn main() {
        let (teacher, student, label, t, alpha) = ([5.0, 2.0, 0.5], [3.0, 2.5, 0.0], 0, 4.0_f64, 0.5_f64);
        let (pt, ps) = (softmax(&teacher, t), softmax(&student, t));
        let kl: f64 = pt.iter().zip(ps.iter()).map(|(p, q)| p * (p / q).ln()).sum();
        let ce = -softmax(&student, 1.0)[label].ln();
        let show = |v: &Vec<f64>| v.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", ");
        println!("teacher soft targets: {}", show(&pt));
        println!("student at T       : {}", show(&ps));
        println!("loss = {}*CE {:.4} + {}*T^2*KL {:.4} = {:.4}", alpha, ce, 1.0 - alpha, t * t * kl, alpha * ce + (1.0 - alpha) * t * t * kl);
    }
  `, r`
    teacher soft targets: 0.556, 0.263, 0.181
    student at T       : 0.425, 0.375, 0.201
    loss = 0.5*CE 0.5046 + 0.5*T^2*KL 0.6131 = 0.5588
  `, "At temperature 4 the teacher's soft targets reveal how it ranks the wrong classes. The student is trained on a mix of the usual hard-label cross-entropy and the T²-scaled KL divergence to those soft targets."),
};
