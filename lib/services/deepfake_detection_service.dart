import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection_result.dart';

/// Runs deepfake detection fully on-device using a quantized INT8
/// TensorFlow Lite model.
///
/// This is the single place that knows about TFLite. Everything else
/// in the app (UI, navigation) only ever talks to [DetectionResult],
/// so the underlying model can be swapped for a different INT8
/// classifier without touching any other file — see
/// MODEL_INTEGRATION.md for exactly what to change and where.
///
/// No network calls are made anywhere in this class.
class DeepfakeDetectionService {
  DeepfakeDetectionService({
    this.modelAssetPath = 'assets/models/deepshield_int8.tflite',
  });

  /// Path of the bundled .tflite model inside Flutter assets.
  final String modelAssetPath;

  Interpreter? _interpreter;

  int _inputHeight = 224;
  int _inputWidth = 224;
  int _inputChannels = 3;
  TensorType _inputType = TensorType.uint8;
  TensorType _outputType = TensorType.uint8;
  List<int> _outputShape = const [1, 2];

  double _inputScale = 1.0;
  int _inputZeroPoint = 0;
  double _outputScale = 1.0;
  int _outputZeroPoint = 0;

  bool get isReady => _interpreter != null;

  /// Loads the model and reads its input/output quantization
  /// parameters. Safe to call more than once — subsequent calls are
  /// no-ops while a model is already loaded.
  Future<void> loadModel() async {
    if (_interpreter != null) return;

    try {
      final interpreterOptions = InterpreterOptions()..threads = 4;

      final interpreter = await Interpreter.fromAsset(
        modelAssetPath,
        options: interpreterOptions,
      );

      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      final inputShape = inputTensor.shape; // [1, H, W, C] typically
      if (inputShape.length < 3) {
        throw const DeepfakeDetectionException(
          'The bundled model has an unexpected input shape. '
          'It must accept a single [1, height, width, channels] image tensor.',
        );
      }

      _inputHeight = inputShape[inputShape.length - 3];
      _inputWidth = inputShape[inputShape.length - 2];
      _inputChannels = inputShape.last;
      _inputType = inputTensor.type;
      _outputType = outputTensor.type;
      _outputShape = outputTensor.shape;

      final inputParams = inputTensor.params;
      _inputScale = inputParams.scale == 0 ? 1.0 : inputParams.scale;
      _inputZeroPoint = inputParams.zeroPoint;

      final outputParams = outputTensor.params;
      _outputScale = outputParams.scale == 0 ? 1.0 : outputParams.scale;
      _outputZeroPoint = outputParams.zeroPoint;

      _interpreter = interpreter;
    } on DeepfakeDetectionException {
      rethrow;
    } catch (e) {
      throw DeepfakeDetectionException(
        'Could not load the on-device model ($modelAssetPath). '
        'Make sure a valid INT8 .tflite file is bundled at that path. ($e)',
      );
    }
  }

  /// Preprocesses [imageFile], runs local inference, and returns the
  /// classification with confidence and measured latency.
  Future<DetectionResult> analyzeImage(File imageFile) async {
    if (_interpreter == null) {
      await loadModel();
    }
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw const DeepfakeDetectionException(
        'The detection model is not available.',
      );
    }

    final bytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const DeepfakeDetectionException(
        'Could not read the selected image. Please choose a JPEG or PNG photo.',
      );
    }

    final resized = img.copyResize(
      decoded,
      width: _inputWidth,
      height: _inputHeight,
      interpolation: img.Interpolation.linear,
    );

    final input = _buildInputBuffer(resized);
    final output = _buildOutputBuffer();

    final stopwatch = Stopwatch()..start();
    interpreter.run(input, output);
    stopwatch.stop();

    return _interpretOutput(output, stopwatch.elapsedMilliseconds);
  }

  /// Builds an input tensor shaped [1, H, W, C] as plain nested Lists.
  ///
  /// Values are normalized to [0, 1] the same way the model was
  /// trained. For an int8/uint8 model that's then quantized with the
  /// model's own scale/zero-point; for a float32 model (e.g. one
  /// converted with `inference_input_type = tf.float32` so only the
  /// weights are int8-quantized internally) the normalized [0,1]
  /// value is used directly, matching how the model was trained and
  /// evaluated.
  ///
  /// This builds nested `List`s by hand instead of a flat TypedData
  /// buffer reshaped via tflite_flutter's `reshape` extension: that
  /// extension throws `NoSuchMethodError` on `Float32List` at runtime
  /// (a limitation of the package, not of the model), even though it
  /// compiles fine. Plain nested Lists work for every tensor type.
  dynamic _buildInputBuffer(img.Image image) {
    final bool isFloat = _inputType == TensorType.float32;
    final bool isInt8 = _inputType == TensorType.int8;

    final List<List<List<num>>> rows = List.generate(
      _inputHeight,
      (y) => List.generate(
        _inputWidth,
        (x) {
          final pixel = image.getPixel(x, y);
          final List<num> channelValues = _inputChannels == 1
              ? [img.getLuminance(pixel)]
              : [pixel.r, pixel.g, pixel.b];

          return channelValues.map((raw) {
            final normalized = raw / 255.0;
            if (isFloat) {
              return normalized.toDouble();
            }
            final quantized =
                (normalized / _inputScale + _inputZeroPoint).round();
            return isInt8
                ? quantized.clamp(-128, 127)
                : quantized.clamp(0, 255);
          }).toList(growable: false);
        },
        growable: false,
      ),
      growable: false,
    );

    return [rows];
  }

  /// Builds an output buffer matching [_outputShape] as plain nested
  /// Lists (see [_buildInputBuffer] for why this avoids `reshape`).
  dynamic _buildOutputBuffer() {
    List<dynamic> build(List<int> shape) {
      if (shape.length == 1) {
        return _outputType == TensorType.float32
            ? List<double>.filled(shape[0], 0.0)
            : List<int>.filled(shape[0], 0);
      }
      return List.generate(
        shape[0],
        (_) => build(shape.sublist(1)),
        growable: false,
      );
    }

    return build(_outputShape);
  }

  /// Dequantizes (or, for a float32 model, simply reads) the raw
  /// output and turns it into a [DetectionResult].
  ///
  /// Two output layouts are supported out of the box:
  ///  * shape [1, 2]  -> [realScore, fakeScore] logits, softmax'd here.
  ///  * shape [1, 1]  -> a single sigmoid-style FAKE probability.
  /// If your model uses a different layout, adjust this method —
  /// see MODEL_INTEGRATION.md.
  DetectionResult _interpretOutput(dynamic output, int inferenceTimeMs) {
    final List<dynamic> row = (output as List).first as List;
    final bool isFloat = _outputType == TensorType.float32;
    final List<double> dequantized = row
        .map(
          (v) => isFloat
              ? (v as num).toDouble()
              : _outputScale * ((v as num).toInt() - _outputZeroPoint),
        )
        .toList();

    if (dequantized.length >= 2) {
      final realScore = dequantized[0];
      final fakeScore = dequantized[1];
      final isFake = fakeScore >= realScore;

      final maxScore = math.max(realScore, fakeScore);
      final expReal = math.exp(realScore - maxScore);
      final expFake = math.exp(fakeScore - maxScore);
      final sum = expReal + expFake;
      final confidence = isFake ? expFake / sum : expReal / sum;

      return DetectionResult(
        label: isFake ? DetectionLabel.fake : DetectionLabel.real,
        confidence: confidence.isFinite ? confidence : 0.5,
        inferenceTimeMs: inferenceTimeMs,
      );
    }

    final fakeProbability = dequantized.first.clamp(0.0, 1.0);
    final isFake = fakeProbability >= 0.5;

    return DetectionResult(
      label: isFake ? DetectionLabel.fake : DetectionLabel.real,
      confidence: isFake ? fakeProbability : 1 - fakeProbability,
      inferenceTimeMs: inferenceTimeMs,
    );
  }

  /// Releases the interpreter's native resources.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
