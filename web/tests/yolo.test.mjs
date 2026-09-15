/**
 * Unit + parity tests for web/static/yolo.js.
 *
 *   node --test web/tests/            (from the repo root, after `npm install` in web/)
 *
 * The parity test needs fixtures produced by Python (see web/tests/README.md):
 *   $FIXTURES/coco.onnx, letterboxed.rgba, python_result.json
 * It is skipped when FIXTURES is not set.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { letterboxParams, rgbaToCHW, iou, nms, decodeOutput, summarise, bannerText } from "../static/yolo.js";

test("letterboxParams keeps aspect ratio and centres", () => {
  const p = letterboxParams(400, 267, 416, 416);
  assert.equal(p.nw, 416);
  assert.equal(p.nh, Math.round(267 * (416 / 400)));
  assert.equal(p.dx, 0);
  assert.equal(p.dy, Math.floor((416 - p.nh) / 2));
});

test("rgbaToCHW is planar and normalised", () => {
  const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 128, 0, 255]); // 2 px
  const t = rgbaToCHW(rgba, 2, 1);
  assert.deepEqual([...t].map((v) => +v.toFixed(3)), [1, 0, 0, 0.502, 0, 0]);
});

test("iou / nms", () => {
  assert.equal(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
  assert.equal(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  const kept = nms([
    { bbox: [0, 0, 10, 10], confidence: 0.9, classId: 0 },
    { bbox: [1, 1, 11, 11], confidence: 0.8, classId: 0 },   // overlaps -> suppressed
    { bbox: [1, 1, 11, 11], confidence: 0.7, classId: 1 },   // other class -> kept
    { bbox: [50, 50, 60, 60], confidence: 0.6, classId: 0 }, // far away -> kept
  ], 0.5);
  assert.deepEqual(kept.map((d) => d.confidence), [0.9, 0.7, 0.6]);
});

test("decodeOutput maps letterboxed xywh back to original pixels", () => {
  // one anchor, 2 classes; box centred at (208,208) size 100 in a 416 input for a 400x267 image
  const N = 1, dims = [1, 6, N];
  const data = new Float32Array([208, 208, 100, 100, 0.1, 0.9]);
  const p = letterboxParams(400, 267, 416, 416);
  const dets = decodeOutput(data, dims, { confThresh: 0.5, iouThresh: 0.5, ...p, origW: 400, origH: 267, classNames: ["a", "b"] });
  assert.equal(dets.length, 1);
  assert.equal(dets[0].className, "b");
  const [x1, y1, x2, y2] = dets[0].bbox;
  assert.ok(Math.abs((x2 - x1) - 100 / p.ratio) < 0.01);
  assert.ok(Math.abs(((x1 + x2) / 2) - (208 - p.dx) / p.ratio) < 0.01);
  assert.ok(Math.abs(((y1 + y2) / 2) - (208 - p.dy) / p.ratio) < 0.01);
});

test("summarise / bannerText", () => {
  const names = ["with_mask", "without_mask", "mask_worn_incorrectly"];
  const c = summarise([{ className: "with_mask" }, { className: "with_mask" }, { className: "without_mask" }], names);
  assert.deepEqual(c, { with_mask: 2, without_mask: 1, mask_worn_incorrectly: 0, total: 3 });
  assert.equal(bannerText(c), "Faces: 3 | Masked: 2 | Unmasked: 1 | Incorrect: 0");
});

test("parity with the Python server on a real image", { skip: !process.env.FIXTURES }, async () => {
  const dir = process.env.FIXTURES;
  const ort = await import("onnxruntime-node");
  const ref = JSON.parse(readFileSync(join(dir, "python_result.json"), "utf8"));
  const rgba = new Uint8Array(readFileSync(join(dir, "letterboxed.rgba")));
  const session = await ort.InferenceSession.create(join(dir, "coco.onnx"));
  const tensor = new ort.Tensor("float32", rgbaToCHW(rgba, ref.W, ref.H), [1, 3, ref.H, ref.W]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const o = out[session.outputNames[0]];
  const dets = decodeOutput(o.data, o.dims, { confThresh: 0.25, iouThresh: 0.5, ratio: ref.ratio, dx: ref.dx, dy: ref.dy,
                                              origW: ref.orig[0], origH: ref.orig[1], classNames: [] });
  assert.equal(dets.length, ref.dets.length, `count js=${dets.length} py=${ref.dets.length}`);
  for (let i = 0; i < dets.length; i++) {
    assert.equal(dets[i].classId, ref.dets[i].class_id);
    assert.ok(Math.abs(dets[i].confidence - ref.dets[i].confidence) < 1e-3);
    assert.ok(iou(dets[i].bbox, ref.dets[i].bbox_xyxy) > 0.99, `box ${i} iou too low`);
  }
  console.log(`  parity OK: ${dets.length} detections match the Python server`);
});
