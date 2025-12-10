use clap::Parser;
use core::mem::size_of;
use core::mem::size_of_val;
use core::ptr;
use core::slice;
use lanczos_resampler::ChunkedResampler;
use lanczos_resampler::WholeResampler;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Audio chunk size.
    #[clap(long = "chunk-size")]
    chunk_size: Option<usize>,

    /// The number of channels.
    #[clap(long = "num-channels")]
    num_channels: usize,

    /// Input sample rate in Hz.
    #[clap(long = "input-sample-rate")]
    input_sample_rate: usize,

    /// Output sample rate in Hz.
    #[clap(long = "output-sample-rate")]
    output_sample_rate: usize,

    /// Input file.
    #[clap(value_name = "FILE")]
    input_file: Option<PathBuf>,

    /// Output file.
    #[clap(value_name = "FILE")]
    output_file: Option<PathBuf>,
}

fn main() -> Result<(), std::io::Error> {
    let args = Args::parse();
    match (args.input_file, args.output_file) {
        (Some(input_file), Some(output_file)) => {
            let input_bytes = std::fs::read(&input_file)?;
            let (prefix, samples, suffix) = unsafe { input_bytes.align_to::<f32>() };
            assert_eq!(0, prefix.len());
            assert_eq!(0, suffix.len());
            let resampler = WholeResampler::new();
            let num_input_frames = samples.len() / args.num_channels;
            let num_output_frames = lanczos_resampler::output_len(
                num_input_frames,
                args.input_sample_rate,
                args.output_sample_rate,
            );
            let mut output = Vec::with_capacity(num_output_frames * args.num_channels);
            let num_processed = resampler.resample_interleaved_into(
                samples,
                args.num_channels,
                &mut output.spare_capacity_mut(),
            );
            assert_eq!(num_processed, num_input_frames);
            unsafe { output.set_len(output.capacity()) };
            let output_bytes =
                unsafe { slice::from_raw_parts(output.as_ptr().cast(), size_of_val(&output[..])) };
            std::fs::write(&output_file, output_bytes)?;
            Ok(())
        }
        (None, None) => {
            let chunk_size = args.chunk_size.unwrap_or(args.input_sample_rate);
            let mut resampler =
                ChunkedResampler::new(args.input_sample_rate, args.output_sample_rate);
            let mut buf: Vec<u8> = vec![0_u8; chunk_size * size_of::<f32>()];
            let mut offset = 0;
            loop {
                let n = std::io::stdin().read(&mut buf[offset..])?;
                if n == 0 {
                    break;
                }
                let (prefix, samples, suffix) = unsafe { buf[..n].align_to::<f32>() };
                assert_eq!(0, prefix.len());
                resampler.resample(
                    samples,
                    &mut Writer {
                        inner: std::io::stdout(),
                        frame: Vec::new(),
                    },
                );
                let suffix_len = suffix.len();
                if suffix_len != 0 {
                    buf[..suffix_len].copy_within(n - suffix_len..n, 0);
                }
                offset = suffix_len;
            }
            Ok(())
        }
        _ => Err(std::io::Error::other(
            "You need either specify both input and output files or neither of them",
        )),
    }
}

struct Writer<W: Write> {
    inner: W,
    frame: Vec<f32>,
}

impl<W: Write> lanczos_resampler::Output for Writer<W> {
    fn remaining(&self) -> Option<usize> {
        None
    }

    fn write(&mut self, sample: f32) {
        let bytes =
            unsafe { slice::from_raw_parts(ptr::from_ref(&sample).cast(), size_of::<f32>()) };
        self.inner.write_all(bytes).unwrap();
    }

    fn write_slice(&mut self, samples: &[f32]) {
        let bytes = unsafe { slice::from_raw_parts(samples.as_ptr().cast(), size_of_val(samples)) };
        self.inner.write_all(bytes).unwrap();
    }

    fn write_frame(&mut self, num_channels: usize, write: impl FnOnce(&mut [f32])) {
        self.frame.resize(num_channels, 0.0);
        self.frame.fill(0.0);
        write(&mut self.frame[..]);
        let bytes = unsafe {
            slice::from_raw_parts(self.frame.as_ptr().cast(), size_of_val(&self.frame[..]))
        };
        self.inner.write_all(bytes).unwrap();
    }
}
