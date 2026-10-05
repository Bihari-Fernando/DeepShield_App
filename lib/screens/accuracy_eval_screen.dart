import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:share_plus/share_plus.dart';

import '../services/deepfake_detection_service.dart';
import '../services/eval_utils.dart';

/// Developer-only screen (long-press the shield icon on the home screen,
/// then "Accuracy evaluation") that measures how the bundled model
/// classifies a labelled set of images ON THIS DEVICE.
///
/// Each image goes through the exact same code path as a normal scan
/// ([DeepfakeDetectionService.analyzeImage]: decode, resize, model,
/// interpret) — only the resize interpolation is selectable. The operator
/// picks the ground-truth label for a batch (e.g. every image from the
/// `real` folder), then picks the images. Results accumulate across
/// batches and can be exported as CSV for offline analysis against the
/// Colab manifest (scripts/analyze_phone_eval.py).
class AccuracyEvalScreen extends StatefulWidget {
  const AccuracyEvalScreen({super.key});

  @override
  State<AccuracyEvalScreen> createState() => _AccuracyEvalScreenState();
}

class _ModelChoice {
  const _ModelChoice(this.key, this.label, this.asset);

  final String key;
  final String label;
  final String asset;
}

class _AccuracyEvalScreenState extends State<AccuracyEvalScreen> {
  static const List<_ModelChoice> _models = [
    _ModelChoice('int8', 'Int8', 'assets/models/deepshield_int8.tflite'),
    _ModelChoice(
        'float32', 'Float32', 'assets/models/deepshield_float32.tflite'),
  ];

  final ImagePicker _picker = ImagePicker();
  final List<EvalRecord> _records = [];
  final Set<String> _seen = {};

  int _modelIndex = 0;
  bool _nearest = false; // false = linear (app default), true = nearest
  bool _batchIsFake = false;

  bool _running = false;
  double _progress = 0;
  String _status = '';
  String? _error;
  int _failures = 0;
  int _duplicatesSkipped = 0;

  String get _resizeKey => _nearest ? 'nearest' : 'linear';

  Future<void> _pickAndEvaluate() async {
    final model = _models[_modelIndex];

    List<XFile> picked;
    try {
      picked = await _picker.pickMultiImage();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Could not open the image picker: $e');
      return;
    }
    if (picked.isEmpty || !mounted) return;

    setState(() {
      _running = true;
      _error = null;
      _progress = 0;
      _status = 'Loading ${model.label} model…';
    });

    final service = DeepfakeDetectionService(
      modelAssetPath: model.asset,
      interpolation:
          _nearest ? img.Interpolation.nearest : img.Interpolation.linear,
    );

    final batchLabel = _batchIsFake ? 'FAKE' : 'REAL';
    var failedThisBatch = 0;
    var dupThisBatch = 0;
    final added = <EvalRecord>[];

    try {
      await service.loadModel();

      for (var i = 0; i < picked.length; i++) {
        final file = File(picked[i].path);
        try {
          final bytes = await file.readAsBytes();
          final hash = fnv1a32Hex(bytes);
          final seenKey = '${model.key}|$_resizeKey|$hash';

          if (_seen.contains(seenKey)) {
            dupThisBatch++;
          } else {
            final result = await service.analyzeImage(file);
            _seen.add(seenKey);
            added.add(EvalRecord(
              model: model.key,
              resize: _resizeKey,
              batchLabel: batchLabel,
              hash: hash,
              name: picked[i].name,
              pReal: result.realProbability,
              predictedFake: result.isFake,
              inferenceMs: result.inferenceTimeMs,
            ));
          }
        } catch (_) {
          failedThisBatch++;
        }

        if (!mounted) return;
        setState(() {
          _progress = (i + 1) / picked.length;
          _status = 'Image ${i + 1}/${picked.length}';
        });
        // Let the progress bar repaint between images.
        await Future<void>.delayed(Duration.zero);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Evaluation failed for ${model.asset}.\n'
            'If this is the Float32 model, make sure it was bundled in '
            'assets/models/.\n\nDetails: $e';
      });
    } finally {
      service.dispose();
      if (mounted) {
        setState(() {
          _records.addAll(added);
          _failures += failedThisBatch;
          _duplicatesSkipped += dupThisBatch;
          _running = false;
          _status = '';
        });
      }
    }
  }

  Future<void> _copyCsv() async {
    await Clipboard.setData(ClipboardData(text: recordsToCsv(_records)));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Copied ${_records.length} rows as CSV')),
    );
  }

  Future<void> _shareCsv() async {
    await SharePlus.instance.share(
      ShareParams(
        text: recordsToCsv(_records),
        subject: 'DeepShield on-device accuracy evaluation (CSV)',
      ),
    );
  }

  void _clear() {
    setState(() {
      _records.clear();
      _seen.clear();
      _failures = 0;
      _duplicatesSkipped = 0;
      _error = null;
    });
  }

  Map<String, EvalTally> _tallies() {
    final map = <String, EvalTally>{};
    for (final r in _records) {
      map.putIfAbsent(r.configKey, EvalTally.new).add(r);
    }
    return map;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final tallies = _tallies();

    return Scaffold(
      appBar: AppBar(title: const Text('On-device accuracy')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Uses the same detection code as a normal scan. Pick the '
              'ground truth for the batch, then pick the images (about 100 '
              'at a time is safest). Results accumulate until you clear '
              'them. The thesis analysis takes ground truth from the Colab '
              'manifest, not from this selector, and flags any mismatch.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),
            Text('Ground truth of the images you will pick',
                style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<bool>(
              segments: const [
                ButtonSegment<bool>(value: false, label: Text('REAL')),
                ButtonSegment<bool>(value: true, label: Text('FAKE')),
              ],
              selected: {_batchIsFake},
              onSelectionChanged:
                  _running ? null : (s) => setState(() => _batchIsFake = s.first),
            ),
            const SizedBox(height: 16),
            Text('Model', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<int>(
              segments: [
                for (var i = 0; i < _models.length; i++)
                  ButtonSegment<int>(value: i, label: Text(_models[i].label)),
              ],
              selected: {_modelIndex},
              onSelectionChanged:
                  _running ? null : (s) => setState(() => _modelIndex = s.first),
            ),
            const SizedBox(height: 16),
            Text('Resize method', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<bool>(
              segments: const [
                ButtonSegment<bool>(
                    value: false, label: Text('Linear (app default)')),
                ButtonSegment<bool>(
                    value: true, label: Text('Nearest (Keras default)')),
              ],
              selected: {_nearest},
              onSelectionChanged:
                  _running ? null : (s) => setState(() => _nearest = s.first),
            ),
            const SizedBox(height: 20),
            FilledButton.icon(
              onPressed: _running ? null : _pickAndEvaluate,
              icon: const Icon(Icons.photo_library_outlined),
              label: Text(_running ? 'Evaluating…' : 'Pick images & evaluate'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),
            if (_running) ...[
              const SizedBox(height: 16),
              LinearProgressIndicator(value: _progress),
              const SizedBox(height: 8),
              Text(_status, style: theme.textTheme.bodySmall),
            ],
            if (_error != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  _error!,
                  style: TextStyle(color: theme.colorScheme.onErrorContainer),
                ),
              ),
            ],
            const SizedBox(height: 24),
            Text('Running tally (${_records.length} images)',
                style: theme.textTheme.titleMedium),
            if (_failures > 0 || _duplicatesSkipped > 0)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  'Could not process: $_failures · Duplicates skipped: '
                  '$_duplicatesSkipped',
                  style: theme.textTheme.bodySmall,
                ),
              ),
            if (tallies.isEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text('Nothing evaluated yet.',
                    style: theme.textTheme.bodySmall),
              ),
            for (final entry in tallies.entries)
              _TallyCard(configKey: entry.key, tally: entry.value),
            if (_records.isNotEmpty) ...[
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  FilledButton.tonalIcon(
                    onPressed: _copyCsv,
                    icon: const Icon(Icons.copy),
                    label: const Text('Copy CSV'),
                  ),
                  FilledButton.tonalIcon(
                    onPressed: _shareCsv,
                    icon: const Icon(Icons.share),
                    label: const Text('Share CSV'),
                  ),
                  TextButton.icon(
                    onPressed: _running ? null : _clear,
                    icon: const Icon(Icons.delete_outline),
                    label: const Text('Clear all'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _TallyCard extends StatelessWidget {
  const _TallyCard({required this.configKey, required this.tally});

  final String configKey;
  final EvalTally tally;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final parts = configKey.split('|');
    final title = '${parts[0]} · resize ${parts[1]}';

    return Card(
      margin: const EdgeInsets.only(top: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: theme.textTheme.titleSmall),
            const SizedBox(height: 6),
            Text(
              'Accuracy ${(tally.accuracy * 100).toStringAsFixed(1)}%  '
              '(${tally.correct}/${tally.total})',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'REAL → REAL ${tally.realAsReal} · REAL → FAKE '
              '${tally.realAsFake}\n'
              'FAKE → FAKE ${tally.fakeAsFake} · FAKE → REAL '
              '${tally.fakeAsReal}',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
