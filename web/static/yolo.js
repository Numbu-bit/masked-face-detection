/**
 * YOLOv8 pre/post-processing for onnxruntime-web (and Node, for tests).
 *
 * Pure functions - no DOM access - so the exact same code runs in the browser
 * and in the Node test-suite (web/tests/yolo.test.mjs).
 *
 * Model contract (Ultralytics ONNX export, no NMS in graph):
 *   input : float32 [1, 3, H, W], RGB, 0-1, letterboxed with grey (114) padding
 *   output: float32 [1, 4 + numClasses, N]  ->  cx, cy, w, h (input px), class scores
 */

/**
 * Letterbox geometry: scale ratio and padding offsets to fit (w0,h0) into (W,H).
 * @returns {{ratio:number, dx:number, dy:number, nw:number, nh:number}}
 */
export function letterboxParams(w0, h0, W, H) {
  const ratio = Math.min(W / w0, H / h0);
  const nw = Math.max(1, Math.round(w0 * ratio));
  const nh = Math.max(1, Math.round(h0 * ratio));
  return { ratio, dx: Math.floor((W - nw) / 2), dy: Math.floor((H - nh) / 2), nw, nh };
}

/**
 * Convert an RGBA byte buffer (W*H*4, e.g. from canvas.getImageData) into the
 * planar CHW float32 tensor the model expects.
 */
export function rgbaToCHW(rgba, W, H) {
  const size = W * H;
  const out = new Float32Array(3 * size);
  for (let i = 0, p = 0; i < size; i++, p += 4) {
    out[i] = rgba[p] / 255;
    out[i + size] = rgba[p + 1] / 255;
    out[i + 2 * size] = rgba[p + 2] / 255;
  }
  return out;
}

/** IoU of two [x1,y1,x2,y2] boxes. */
export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / Math.max(areaA + areaB - inter, 1e-9);
}

/**
 * Greedy per-class non-maximum suppression.
 * @param {Array<{bbox:number[], confidence:number, classId:number}>} dets
 * @returns the kept detections sorted by confidence (desc)
 */
export function nms(dets, iouThresh) {
  const byClass = new Map();
  for (const d of dets) {
    if (!byClass.has(d.classId)) byClass.set(d.classId, []);
    byClass.get(d.classId).push(d);
  }
  const keep = [];
  for (const group of byClass.values()) {
    group.sort((a, b) => b.confidence - a.confidence);
    const suppressed = new Uint8Array(group.length);
    for (let i = 0; i < group.length; i++) {
      if (suppressed[i]) continue;
      keep.push(group[i]);
      for (let j = i + 1; j < group.length; j++) {
        if (!suppressed[j] && iou(group[i].bbox, group[j].bbox) >= iouThresh) suppressed[j] = 1;
      }
    }
  }
  return keep.sort((a, b) => b.confidence - a.confidence);
}

/**
 * Decode the raw model output into detections in ORIGINAL image pixels.
 * @param {Float32Array} data  flat output tensor
 * @param {number[]} dims      [1, 4+nc, N]
 * @param {object} p           {confThresh, iouThresh, ratio, dx, dy, origW, origH, classNames, maxDet}
 */
export function decodeOutput(data, dims, p) {
  const [, C, N] = dims;
  const nc = C - 4;
  const classNames = p.classNames || [];
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, bestId = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * N + i];
      if (s > best) { best = s; bestId = c; }
    }
    if (best < p.confThresh) continue;
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    const x1 = Math.min(Math.max((cx - w / 2 - p.dx) / p.ratio, 0), p.origW);
    const y1 = Math.min(Math.max((cy - h / 2 - p.dy) / p.ratio, 0), p.origH);
    const x2 = Math.min(Math.max((cx + w / 2 - p.dx) / p.ratio, 0), p.origW);
    const y2 = Math.min(Math.max((cy + h / 2 - p.dy) / p.ratio, 0), p.origH);
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;
    cand.push({ bbox: [x1, y1, x2, y2], confidence: best, classId: bestId,
                className: classNames[bestId] ?? String(bestId) });
  }
  return nms(cand, p.iouThresh).slice(0, p.maxDet || 300);
}

/** Per-class counts + total, e.g. {with_mask: 2, without_mask: 1, ..., total: 3}. */
export function summarise(dets, classNames) {
  const counts = Object.fromEntries(classNames.map((n) => [n, 0]));
  for (const d of dets) counts[d.className] = (counts[d.className] || 0) + 1;
  counts.total = dets.length;
  return counts;
}

/** Banner text identical to the Python side: "Faces: 5 | Masked: 3 | Unmasked: 2 | Incorrect: 0". */
export function bannerText(counts) {
  return `Faces: ${counts.total} | Masked: ${counts.with_mask ?? 0} | ` +
         `Unmasked: ${counts.without_mask ?? 0} | Incorrect: ${counts.mask_worn_incorrectly ?? 0}`;
}
