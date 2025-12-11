import assert from "node:assert";
import {
    ChunkedResampler,
    ChunkedInterleavedResampler,
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
    const numProcessed = resampler.resample(chunk, output);
    assert.equal(chunk.length, numProcessed);
    origOutput = output;
    const resampler2 = new ChunkedInterleavedResampler(44100, 48000, 1);
    const output2 = new Float32Array(
        resampler2.maxOutputChunkLength(chunk.length),
    );
    const numProcessed2 = resampler2.resample(chunk, output2);
    assert.equal(chunk.length, numProcessed2);
    assert.ok(output2.every((y, i) => y === origOutput[i]));
}

{
    const whole = new Float32Array(1024);
    whole.fill(0.1);
    const outputLen = outputLength(1024, 44100, 48000);
    const output = new Float32Array(outputLen);
    const resampler = new WholeResampler();
    const numProcessed = resampler.resampleInto(whole, output);
    assert.equal(numProcessed, whole.length);
    assert.ok(output.every((y, i) => y === origOutput[i]));
    const output2 = new Float32Array(outputLen);
    const numProcessed2 = resampler.resampleInterleavedInto(whole, 1, output2);
    assert.equal(numProcessed2, whole.length);
    assert.ok(output2.every((y, i) => y === origOutput[i]));
}

function benchmark(name, callback, iterations) {
    // Warm-up run.
    for (let i = 0; i < iterations; ++i) {
        callback();
    }
    // Real run.
    const t0 = performance.now();
    for (let i = 0; i < iterations; ++i) {
        callback();
    }
    const t1 = performance.now();
    console.log(`${name}: ${(t1 - t0) / iterations} ms`);
}

{
    const n = 44100 * 6;
    const whole = new Float32Array(n);
    whole.fill(0.1);
    const outputLen = outputLength(n, 44100, 48000);
    const output = new Float32Array(outputLen);
    const resampler = new WholeResampler();
    benchmark(
        "WholeResampler.resampleInto",
        () => resampler.resampleInto(whole, output),
        100,
    );
}

{
    const chunk = new Float32Array(1024);
    chunk.fill(0.1);
    const resampler = new ChunkedResampler(44100, 48000);
    const output = new Float32Array(
        resampler.maxOutputChunkLength(chunk.length),
    );
    benchmark(
        "ChunkedResampler.resample",
        () => resampler.resample(chunk, output),
        100,
    );
}
