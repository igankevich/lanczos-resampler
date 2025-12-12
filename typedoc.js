export default {
    compilerOptions: {
        lib: ["esnext"],
    },
    highlightLanguages: ["rust", "javascript", "bash"],
    excludeNotDocumented: true,
    entryPoints: ["pkg/lanczos_resampler.d.ts"],
};
