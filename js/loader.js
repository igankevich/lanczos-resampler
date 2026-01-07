/**
 * This module provides functions to asynchronously load the library.
 *
 * This can be useful inside audio worklets and web workers.
 * @module loader
 */
import './text-decoder.js'
import { __wbg_set_wasm } from './lanczos_resampler_bg.js'

/**
 * Instantiate WebAssembly module using hard-coded BASE64 string.
 *
 * Use this function to asynchronously initialize the library inside AudioWorklet.
 *
 * #### Example
 *
 * ```javascript
 * // Inside audio worklet.
 * import { ChunkedResampler, initWithBase64 } from 'lanczos-resampler/loader'
 *
 * (async () => {
 *     // N.B. Top-level `await`s aren't reliable in audio worklets.
 *     // This code should be moved to a more appropriate place.
 *     await initWithBase64()
 *     const resampler = new ChunkedResampler(44100, 48000)
 *     console.debug(resampler)
 * })()
 *
 * class MyAudioProcessor extends AudioWorkletProcessor {
 *     // ...
 * }
 *
 * ```
 */
export async function initWithBase64() {
    const result = await WebAssembly.instantiate(decodeBase64(CODE_BASE64), IMPORT_OBJECT)
    __wbg_set_wasm(result.instance.exports)
}

/**
 * Instantiate WebAssembly module by fetching the code from the provided URL.
 *
 * Use this function to asynchronously initialize the library inside Web Worker
 * or in the main thread.
 *
 * @param url - the URL where the code can be downlaoded
 * @param options - `fetch` options
 *
 * #### Example
 *
 * ```javascript
 * // Inside web worker.
 * import codeURL from 'lanczos-resampler/code.wasm?url' // Vite.
 * import { ChunkedResampler, initWithFetch } from 'lanczos-resampler/loader';
 *
 * (async () => {
 *     // N.B. Any messages that are sent to the worker during top-level `await` are lost.
 *     // Hence this code should be moved to a different scope.
 *     await initWithFetch(codeURL);
 *     const resampler = new ChunkedResampler(44100, 48000);
 *     console.log(resampler);
 * })()
 * ```
 */
export async function initWithFetch(url, options) {
    const response = await fetch(url, options)
    const result = await WebAssembly.instantiateStreaming(response, IMPORT_OBJECT)
    __wbg_set_wasm(result.instance.exports)
}

function decodeBase64(string) {
    if (typeof Uint8Array === 'function' && typeof Uint8Array.fromBase64 === 'function') {
        return Uint8Array.fromBase64(CODE_BASE64)
    }
    if (typeof Buffer === 'function' && typeof Buffer.from === 'function') {
        return Buffer.from(CODE_BASE64, 'base64')
    }
    if (typeof atob === 'function') {
        return Uint8Array.from(atob(string), (m) => m.codePointAt(0))
    }
    throw new Error('Unable to decode BASE64 string')
}
