import assert from "node:assert";
import {
    ChunkedResampler,
    WholeResampler,
    outputLength,
} from "./pkg/lanczos_resampler.js";

let origOutput;

{
    const chunk = new Float32Array(1024);
    chunk.fill(0.1);
    const resampler = new ChunkedResampler(44100, 48000);
    const output = new Float32Array(
        resampler.maxOutputChunkLength(chunk.length),
    );
    const numProcessed = resampler.resampleChunk(chunk, output);
    assert.equal(chunk.length, numProcessed);
    origOutput = output;
}

{
    const whole = new Float32Array(1024);
    whole.fill(0.1);
    const outputLen = outputLength(1024, 44100, 48000);
    const output = new Float32Array(outputLen);
    const resampler = new WholeResampler();
    const numProcessed = resampler.resampleWholeInto(whole, output);
    assert.equal(numProcessed, whole.length);
    assert.ok(output.every((y, i) => y === origOutput[i]));
}

{
    const n = 44100 * 6;
    const whole = new Float32Array(n);
    whole.fill(0.1);
    const outputLen = outputLength(n, 44100, 48000);
    const output = new Float32Array(outputLen);
    const resampler = new WholeResampler();
    const iterations = 100
    for (let i = 0; i < iterations; ++i) {
        resampler.resampleWholeInto(whole, output);
    }
    const t0 = performance.now();
    for (let i = 0; i < iterations; ++i) {
        resampler.resampleWholeInto(whole, output);
    }
    const t1 = performance.now();
    console.log(`WholeResampler.resampleWholeInto ${(t1 - t0) / iterations} ms`);
}
