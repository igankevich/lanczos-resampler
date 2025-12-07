import assert from "node:assert";
import * as lanczos from "./pkg/lanczos_resampler_js.js";

const input = new Float32Array(1024);
input.fill(0.1);
const output = lanczos.resample(input, 44100, 48000);

const outputLength = lanczos.outputLength(1024, 44100, 48000);
assert.equal(output.length, outputLength);

const output2 = new Float32Array(outputLength);
const numRead = lanczos.resampleInto(input, 44100, 48000, output2);
assert.equal(numRead, input.length);
assert.ok(output.every((y, i) => y === output2[i]));
