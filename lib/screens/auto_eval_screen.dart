import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';

import '../services/deepfake_detection_service.dart';
import '../services/eval_utils.dart';

/// Developer-only: evaluates EVERY image in a folder on this device with all
/// four configurations (Int8 / Float32 x linear / nearest resize) in one go,
/// writes the complete CSV to a file and opens the share sheet so it can be
/// saved (Files, Drive, ...).
///
/// The ground truth of each image is taken from its folder name (`real` /
/// `fake`) or, failing that, from `_real_` / `_fake_` in its file name — the
/// layout produced by notebooks/on_device_eval/exp12_build_phone_eval_set.ipynb.
/// The thesis analysis still takes the truth from the Colab manifest and
/// flags any mismatch.
class AutoEvalScreen extends StatefulWidget {
  const AutoEvalScreen({super.key});

  @override
  State<AutoEvalScreen> createState() => _AutoEvalScreenState();
}

class _Config {
  const _Config(this.model, this.asset, this.resize, this.interpolation);

  final String model;
  final String asset;
  final String resize;
  final img.Interpolation interpolation;

  String get key => '$model|$resize';
}

class _AutoEvalScreenState extends State<AutoEvalScreen> {
  static const List<_Config> _configs = [
    _Config('int8', 'assets/models/deepshield_int8.tflite', 'linear',
        img.Interpolation.linear),
    _Config('int8', 'assets/models/deepshield_int8.tflite', 'nearest',
        img.Interpolation.nearest),
    _Config('float32', 'assets/models/deepshield_float32.tflite', 'linear',
        img.Interpolation.linear),
    _Config('float32', 'assets/models/deepshield_float32.tflite', 'nearest',
        img.Interpolation.nearest),
  ];

  final TextEditingController _folder = TextEditingController(
    text: '/storage/emulated/0/Pictures/phone_eval_set',
  );

  bool _running = false;
  bool _cancel = false;
  double _progress = 0;
  String _status = '';
  String? _error;
  String? _csvPath;
  int _rowsWritten = 0;
  int _failures = 0;
  final Map<String, EvalTally> _tallies = {};
  final List<String> _log = [];

  @override
  void dispose() {
    _folder.dispose();
    super.dispose();
  }

  Future<bool> _ensureAccess() async {
    final photos = await Permission.photos.request();
    if (photos.isGranted || photos.isLimited) return true;
    final storage = await Permission.storage.request();
    return storage.isGranted;
  }

  /// Returns (path, truthIsFake) for every labelled image under [root].
  Future<List<(File, bool)>> _scan(Directory root) async {
    final out = <(File, bool)>[];
    await for (final e in root.list(recursive: true, followLinks: false)) {
      if (e is! File) continue;
      final lower = e.path.toLowerCase();
      if (!(lower.endsWith('.jpg') ||
          lower.endsWith('.jpeg') ||
          lower.endsWith('.png'))) {
        continue;
      }
      final segs = lower.replaceAll('\\', '/').split('/');
      final dirs = segs.sublist(0, segs.length - 1);
      final name = segs.last;
      bool? fake;
      if (dirs.contains('fake')) {
        fake = true;
      } else if (dirs.contains('real')) {
        fake = false;
      } else if (name.contains('_fake_')) {
        fake = true;
      } else if (name.contains('_real_')) {
        fake = false;
      }
      if (fake == null) continue;
      out.add((e, fake));
    }
    out.sort((a, b) => a.$1.path.compareTo(b.$1.path));
    return out;
  }

  Future<void> _run() async {
    setState(() {
      _running = true;
      _cancel = false;
      _error = null;
      _progress = 0;
      _rowsWritten = 0;
      _failures = 0;
      _tallies.clear();
      _log.clear();
      _csvPath = null;
      _status = 'Checking access…';
    });

    IOSink? sink;
    try {
      if (!await _ensureAccess()) {
        throw 'Photo/storage access was denied. Allow it in system settings '
            'and try again.';
      }
      final root = Directory(_folder.text.trim());
      if (!await root.exists()) {
        throw 'Folder not found: ${root.path}';
      }

      setState(() => _status = 'Scanning folder…');
      final images = await _scan(root);
      if (images.isEmpty) {
        throw 'No labelled images found under ${root.path}. If the files '
            'are there, the gallery may not have indexed them yet (restart '
            'the phone) or access was limited.';
      }
      final nReal = images.where((i) => !i.$2).length;
      final nFake = images.length - nReal;
      _log.add('Found ${images.length} images ($nReal real, $nFake fake).');

      final file = File('${Directory.systemTemp.path}/'
          'deepshield_eval_full_${DateTime.now().millisecondsSinceEpoch}.csv');
      final out = file.openWrite();
      sink = out;
      out.writeln(EvalRecord.csvHeader);
      _csvPath = file.path;

      final total = images.length * _configs.length;
      var done = 0;

      for (final cfg in _configs) {
        if (_cancel) break;
        final service = DeepfakeDetectionService(
          modelAssetPath: cfg.asset,
          interpolation: cfg.interpolation,
        );
        try {
          await service.loadModel();
          final tally = _tallies.putIfAbsent(cfg.key, EvalTally.new);
          for (final item in images) {
            if (_cancel) break;
            try {
              final bytes = await item.$1.readAsBytes();
              final hash = fnv1a32Hex(bytes);
              final result = await service.analyzeImage(item.$1);
              final rec = EvalRecord(
                model: cfg.model,
                resize: cfg.resize,
                batchLabel: item.$2 ? 'FAKE' : 'REAL',
                hash: hash,
                name: item.$1.path.split('/').last,
                pReal: result.realProbability,
                predictedFake: result.isFake,
                inferenceMs: result.inferenceTimeMs,
              );
              out.writeln(rec.toCsvRow());
              tally.add(rec);
              _rowsWritten++;
            } catch (_) {
              _failures++;
            }
            done++;
            if (done % 5 == 0 || done == total) {
              if (!mounted) return;
              setState(() {
                _progress = done / total;
                _status = '${cfg.key}  ($done/$total)';
              });
              await out.flush();
              await Future<void>.delayed(Duration.zero);
            }
          }
          _log.add('Finished ${cfg.key}: ${tally.correct}/${tally.total} '
              'correct.');
        } finally {
          service.dispose();
        }
      }

      await out.flush();
      await out.close();
      sink = null;
      if (!mounted) return;
      setState(() {
        _status = _cancel ? 'Cancelled — partial CSV saved.' : 'Done.';
      });
      await _share();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      try {
        await sink?.close();
      } catch (_) {}
      if (mounted) setState(() => _running = false);
    }
  }

  Future<void> _share() async {
    final path = _csvPath;
    if (path == null) return;
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(path, mimeType: 'text/csv')],
        subject: 'DeepShield full on-device evaluation (CSV)',
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Full on-device evaluation')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Runs every image in the folder through all four '
              'configurations (Int8/Float32 x linear/nearest), saves one '
              'complete CSV and opens the share sheet so you can save it '
              '(e.g. to Files or Drive). Keep the app open and the screen '
              'on; it takes a while (roughly 4 x the number of images x '
              '0.4 s).',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _folder,
              enabled: !_running,
              decoration: const InputDecoration(
                labelText: 'Image folder (contains real/ and fake/)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _running ? null : _run,
              icon: const Icon(Icons.play_arrow),
              label: Text(_running ? 'Running…' : 'Run full evaluation'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),
            if (_running) ...[
              const SizedBox(height: 8),
              TextButton(
                onPressed: () => setState(() => _cancel = true),
                child: const Text('Cancel (keeps rows so far)'),
              ),
              const SizedBox(height: 8),
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
            const SizedBox(height: 16),
            Text('Rows written: $_rowsWritten · failed: $_failures',
                style: theme.textTheme.bodyMedium),
            for (final l in _log) Text(l, style: theme.textTheme.bodySmall),
            for (final e in _tallies.entries)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  '${e.key}: ${(e.value.accuracy * 100).toStringAsFixed(1)}% '
                  '(${e.value.correct}/${e.value.total})',
                  style: theme.textTheme.bodySmall,
                ),
              ),
            if (_csvPath != null && !_running) ...[
              const SizedBox(height: 16),
              FilledButton.tonalIcon(
                onPressed: _share,
                icon: const Icon(Icons.share),
                label: const Text('Share / save CSV again'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
