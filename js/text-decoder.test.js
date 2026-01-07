import { getRandomValues } from 'node:crypto'
import { test } from 'node:test'
import { strictEqual } from 'node:assert'
import { TextDecoder as TextDecoderPolyfill } from './text-decoder.js'

function randomU32(a, b) {
    const i = getRandomValues(new Uint32Array(1))[0]
    return a + Math.floor((b - a + 1) * (i / (2 ** 32 - 1)))
}

function randomString(minCodePoint, maxCodePoint) {
    const LEN = 10
    let s = ''
    for (let i = 0; i < LEN; ++i) {
        s += String.fromCodePoint(randomU32(minCodePoint, maxCodePoint))
    }
    return s
}

test('1-byte', (t) => {
    for (let i = 0; i < 1000; ++i) {
        const string = randomString(0, 0x7f)
        const encoder = new TextEncoder()
        const bytes = encoder.encode(string)
        const decoder = new TextDecoder()
        const expected = decoder.decode(bytes)
        const decoder2 = new TextDecoderPolyfill()
        const actual = decoder2.decode(bytes)
        strictEqual(actual, expected)
    }
})

test('2-byte', (t) => {
    for (let i = 0; i < 1000; ++i) {
        const string = randomString(0x80, 0x7ff)
        const encoder = new TextEncoder()
        const bytes = encoder.encode(string)
        const decoder = new TextDecoder()
        const expected = decoder.decode(bytes)
        const decoder2 = new TextDecoderPolyfill()
        const actual = decoder2.decode(bytes)
        strictEqual(actual, expected)
    }
})

test('3-4-byte', (t) => {
    for (let i = 0; i < 1000; ++i) {
        const string = randomString(0x800, 0x10ffff)
        const encoder = new TextEncoder()
        const bytes = encoder.encode(string)
        const decoder = new TextDecoder()
        const expected = decoder.decode(bytes)
        try {
            const decoder2 = new TextDecoderPolyfill()
            const actual = decoder2.decode(bytes)
            strictEqual(actual, expected)
        } catch (e) {
            const codePoints = Uint32Array.from(expected, (m) => m.codePointAt(0))
            console.debug(
                `Expected string ${JSON.stringify(expected)}, code points = [${codePoints}]`,
            )
            throw e
        }
    }
})
