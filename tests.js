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
