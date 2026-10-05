/// Helpers for the on-device accuracy evaluation (developer tool).
library;

/// 32-bit FNV-1a hash of [bytes].
///
/// Used to identify each evaluated image by its exact file content, so the
/// phone's results can be joined to the Colab manifest regardless of how
/// the file was renamed on the way through the gallery / photo picker. It is
/// a plain checksum, not a security hash. Standard test vectors:
/// "" -> 0x811c9dc5, "a" -> 0xe40c292c, "foobar" -> 0xbf9cf968.
int fnv1a32(List<int> bytes) {
  var hash = 0x811c9dc5;
  for (final b in bytes) {
    hash ^= b & 0xff;
    // Dart ints are 64-bit on Android; 32-bit hash * 24-bit prime < 2^63.
    hash = (hash * 0x01000193) & 0xFFFFFFFF;
  }
  return hash;
}

/// [fnv1a32] as 8 lower-case hex digits (zero-padded).
String fnv1a32Hex(List<int> bytes) =>
    fnv1a32(bytes).toRadixString(16).padLeft(8, '0');

/// Quotes a CSV field only when it needs it (comma, quote, newline).
String csvField(String value) {
  if (value.contains(',') ||
      value.contains('"') ||
      value.contains('\n') ||
      value.contains('\r')) {
    return '"${value.replaceAll('"', '""')}"';
  }
  return value;
}

/// One image evaluated on-device with one model / resize configuration.
class EvalRecord {
  const EvalRecord({
    required this.model,
    required this.resize,
    required this.batchLabel,
    required this.hash,
    required this.name,
    required this.pReal,
    required this.predictedFake,
    required this.inferenceMs,
  });

  /// 'int8' or 'float32'.
  final String model;

  /// 'linear' or 'nearest'.
  final String resize;

  /// Ground truth the operator selected for the batch: 'REAL' or 'FAKE'.
  final String batchLabel;

  /// [fnv1a32Hex] of the image file's bytes.
  final String hash;

  /// File name as reported by the picker (may differ from the original).
  final String name;

  /// Model's raw P(real); null if unavailable.
  final double? pReal;

  /// The app's own decision at its 0.5 threshold.
  final bool predictedFake;

  final int inferenceMs;

  bool get truthIsFake => batchLabel == 'FAKE';
  bool get correct => predictedFake == truthIsFake;

  /// Identifies the configuration this record belongs to.
  String get configKey => '$model|$resize';

  static const String csvHeader =
      'model,resize,batch_label,fnv1a32,name,p_real,pred,ms';

  String toCsvRow() => [
        model,
        resize,
        batchLabel,
        hash,
        csvField(name),
        pReal == null ? '' : pReal!.toStringAsFixed(6),
        predictedFake ? 'FAKE' : 'REAL',
        inferenceMs.toString(),
      ].join(',');
}

/// Whole CSV (header + one row per record).
String recordsToCsv(List<EvalRecord> records) =>
    [EvalRecord.csvHeader, ...records.map((r) => r.toCsvRow())].join('\n');

/// Confusion counts for one configuration, with REAL / FAKE as ground truth.
class EvalTally {
  int realAsReal = 0;
  int realAsFake = 0;
  int fakeAsFake = 0;
  int fakeAsReal = 0;

  void add(EvalRecord r) {
    if (r.truthIsFake) {
      if (r.predictedFake) {
        fakeAsFake++;
      } else {
        fakeAsReal++;
      }
    } else {
      if (r.predictedFake) {
        realAsFake++;
      } else {
        realAsReal++;
      }
    }
  }

  int get total => realAsReal + realAsFake + fakeAsFake + fakeAsReal;
  int get correct => realAsReal + fakeAsFake;
  double get accuracy => total == 0 ? 0 : correct / total;
}
