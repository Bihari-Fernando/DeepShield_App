import 'dart:io' show Platform;
import 'dart:math' as math;

import 'package:tflite_flutter/tflite_flutter.dart';

/// What exactly is inside the timed region.
enum BenchmarkMode {
  /// `Interpreter.run(input, output)`: includes tflite_flutter copying the
  /// input into, and the output out of, the native tensors.
  runWithCopy,

  /// `Interpreter.invoke()` only: the input tensor is filled once beforehand
  /// and nothing is copied inside the timed region — pure model execution.
  invokeOnly,
}

/// Latency statistics for one benchmark run of one model / thread-count.
///
/// Every number here is measured on the device the app is running on;
/// nothing is estimated or carried over from another machine.
class BenchmarkResult {
  BenchmarkResult({
    required this.modelAsset,
    required this.threads,
    required this.warmupRuns,
    required this.latenciesMs,
    required this.inputShape,
    required this.inputType,
    required this.outputShape,
    required this.outputType,
    required this.mode,
  }) : timestamp = DateTime.now();

  final String modelAsset;
  final int threads;
  final int warmupRuns;
  final BenchmarkMode mode;

  /// One entry per timed run, in milliseconds, in execution order.
  final List<double> latenciesMs;

  final List<int> inputShape;
  final String inputType;
  final List<int> outputShape;
  final String outputType;
  final DateTime timestamp;

  int get timedRuns => latenciesMs.length;

  double get mean =>
      latenciesMs.reduce((a, b) => a + b) / latenciesMs.length;

  /// Population standard deviation (ddof = 0) — the same definition as
  /// `np.std`, which the Colab benchmark in `exp08` uses, so the two
  /// sets of numbers are directly comparable.
  double get std {
    final m = mean;
    final variance = latenciesMs
            .map((v) => (v - m) * (v - m))
            .reduce((a, b) => a + b) /
        latenciesMs.length;
    return math.sqrt(variance);
  }

  double get min => latenciesMs.reduce(math.min);
  double get max => latenciesMs.reduce(math.max);

  double get median {
    final sorted = [...latenciesMs]..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /// Plain-text report, including every raw latency, so the figures can
  /// be pasted into the thesis notes and re-derived independently.
  String toReport() {
    final raw = latenciesMs.map((v) => v.toStringAsFixed(2)).join(', ');
    return [
      'DeepShield on-device benchmark',
      'Timestamp: ${timestamp.toIso8601String()}',
      'Model asset: $modelAsset',
      'Input: $inputType $inputShape   Output: $outputType $outputShape',
      'Timed call: ${mode == BenchmarkMode.invokeOnly ? 'Interpreter.invoke() only (input set once, no copy in timed region)' : 'Interpreter.run() (includes input/output copy)'}',
      'Threads: $threads (no extra delegate configured)',
      'Warm-up runs (discarded): $warmupRuns',
      'Timed runs: $timedRuns',
      'Mean: ${mean.toStringAsFixed(2)} ms',
      'Std (population): ${std.toStringAsFixed(2)} ms',
      'Median: ${median.toStringAsFixed(2)} ms',
      'Min: ${min.toStringAsFixed(2)} ms',
      'Max: ${max.toStringAsFixed(2)} ms',
      'OS: ${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
      'Raw latencies (ms): $raw',
    ].join('\n');
  }
}

/// Measures pure model-inference latency on this device.
///
/// Two modes (see [BenchmarkMode]): the default measures `invoke()` alone,
/// the other measures `run()` including tflite_flutter's tensor copies.
///
/// Methodology mirrors `benchmark_inference_speed` in
/// `exp08_tflite_conversion_benchmarking.ipynb`: a random [0, 1) input
/// tensor, 5 warm-up runs that are discarded, then 50 timed runs. The
/// input tensor is built once, outside the timed region, so image
/// decoding / resizing / UI work is NOT part of the measurement. What is
/// timed is a single `Interpreter.run` call (which includes copying the
/// input into, and the output out of, the native tensors — the
/// equivalent of `set_tensor` + `invoke` + `get_tensor` in Colab).
///
/// A fresh [Interpreter] is created per call and closed afterwards, so
/// the benchmark never touches the interpreter used for real
/// detections.
class BenchmarkService {
  static const int defaultWarmupRuns = 5;
  static const int defaultTimedRuns = 50;

  /// [onProgress] receives a label and a 0..1 fraction of total runs.
  Future<BenchmarkResult> run({
    required String modelAsset,
    required int threads,
    int warmupRuns = defaultWarmupRuns,
    int timedRuns = defaultTimedRuns,
    BenchmarkMode mode = BenchmarkMode.invokeOnly,
    void Function(String label, double fraction)? onProgress,
  }) async {
    final options = InterpreterOptions()..threads = threads;
    final interpreter =
        await Interpreter.fromAsset(modelAsset, options: options);

    try {
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      final input = _buildInput(inputTensor);
      final output = _buildOutput(outputTensor);

      final total = warmupRuns + timedRuns;
      var done = 0;
      final invokeOnly = mode == BenchmarkMode.invokeOnly;

      if (invokeOnly) {
        // Copies the input into the native input tensor once. The tensor
        // keeps its contents, so later invoke() calls reuse it without any
        // further copying. Not timed and not counted as a warm-up.
        interpreter.run(input, output);
      }

      void callModel() {
        if (invokeOnly) {
          interpreter.invoke();
        } else {
          interpreter.run(input, output);
        }
      }

      for (var i = 0; i < warmupRuns; i++) {
        callModel();
        done++;
        onProgress?.call('Warm-up $done/$warmupRuns', done / total);
        // Yield so the progress bar can repaint. Outside any timed region.
        await Future<void>.delayed(Duration.zero);
      }

      final latencies = <double>[];
      final stopwatch = Stopwatch();
      for (var i = 0; i < timedRuns; i++) {
        stopwatch
          ..reset()
          ..start();
        callModel();
        stopwatch.stop();
        latencies.add(stopwatch.elapsedMicroseconds / 1000.0);

        done++;
        onProgress?.call('Timed run ${i + 1}/$timedRuns', done / total);
        await Future<void>.delayed(Duration.zero);
      }

      return BenchmarkResult(
        modelAsset: modelAsset,
        threads: threads,
        warmupRuns: warmupRuns,
        latenciesMs: latencies,
        inputShape: List<int>.from(inputTensor.shape),
        inputType: _typeName(inputTensor.type),
        outputShape: List<int>.from(outputTensor.shape),
        outputType: _typeName(outputTensor.type),
        mode: mode,
      );
    } finally {
      interpreter.close();
    }
  }

  static String _typeName(TensorType type) =>
      type.toString().replaceFirst('TensorType.', '');

  /// Random values in [0, 1) from a fixed seed (same distribution as the
  /// Colab benchmark's `np.random.rand`), quantized with the tensor's own
  /// scale / zero-point if the model has an int8 / uint8 input.
  dynamic _buildInput(Tensor tensor) {
    final rng = math.Random(42);
    final params = tensor.params;
    final scale = params.scale == 0 ? 1.0 : params.scale;
    final zeroPoint = params.zeroPoint;

    num next() {
      final v = rng.nextDouble();
      switch (tensor.type) {
        case TensorType.int8:
          return (v / scale + zeroPoint).round().clamp(-128, 127);
        case TensorType.uint8:
          return (v / scale + zeroPoint).round().clamp(0, 255);
        default:
          return v;
      }
    }

    return _nested(tensor.shape, next);
  }

  dynamic _buildOutput(Tensor tensor) {
    final isFloat = tensor.type == TensorType.float32;
    List<dynamic> build(List<int> shape) {
      if (shape.length == 1) {
        return isFloat
            ? List<double>.filled(shape[0], 0.0)
            : List<int>.filled(shape[0], 0);
      }
      return List.generate(
        shape[0],
        (_) => build(shape.sublist(1)),
        growable: false,
      );
    }

    return build(tensor.shape);
  }

  /// Plain nested Lists (not TypedData + `reshape`, which throws on
  /// Float32List in tflite_flutter — see deepfake_detection_service.dart).
  dynamic _nested(List<int> shape, num Function() next) {
    if (shape.length == 1) {
      return List<num>.generate(shape[0], (_) => next(), growable: false);
    }
    return List.generate(
      shape[0],
      (_) => _nested(shape.sublist(1), next),
      growable: false,
    );
  }
}
