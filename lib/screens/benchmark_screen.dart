import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/benchmark_service.dart';

/// Developer-only screen (long-press the shield icon on the home screen)
/// that measures real on-device inference latency for the bundled
/// models. Used to produce the Poco M3 numbers for the thesis.
class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _ModelOption {
  const _ModelOption(this.label, this.asset);

  final String label;
  final String asset;
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  static const List<_ModelOption> _models = [
    _ModelOption('Int8', 'assets/models/deepshield_int8.tflite'),
    _ModelOption('Float32', 'assets/models/deepshield_float32.tflite'),
  ];

  final BenchmarkService _service = BenchmarkService();
  final List<BenchmarkResult> _results = [];

  int _modelIndex = 0;
  int _threads = 4;
  BenchmarkMode _mode = BenchmarkMode.invokeOnly;
  bool _running = false;
  String _status = '';
  double _progress = 0;
  String? _error;

  Future<void> _run() async {
    final model = _models[_modelIndex];
    setState(() {
      _running = true;
      _error = null;
      _progress = 0;
      _status = 'Loading ${model.label} model…';
    });

    try {
      final result = await _service.run(
        modelAsset: model.asset,
        threads: _threads,
        mode: _mode,
        onProgress: (label, fraction) {
          if (!mounted) return;
          setState(() {
            _status = label;
            _progress = fraction;
          });
        },
      );
      if (!mounted) return;
      setState(() => _results.insert(0, result));
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Benchmark failed for ${model.asset}.\n'
            'If this is the Float32 model, make sure the file was added to '
            'assets/models/ and the app was rebuilt.\n\nDetails: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
          _status = '';
        });
      }
    }
  }

  /// 3 rounds x {Int8, Float32} x {1, 4 threads} = 12 reports, interleaved
  /// (round-robin) so thermal drift is spread across configurations.
  Future<void> _runAll() async {
    setState(() {
      _running = true;
      _error = null;
      _progress = 0;
      _status = 'Starting…';
    });
    const rounds = 3;
    const threadOptions = [4, 1];
    final totalJobs = rounds * _models.length * threadOptions.length;
    var job = 0;
    try {
      for (var round = 1; round <= rounds; round++) {
        for (final threads in threadOptions) {
          for (final model in _models) {
            job++;
            final label = 'Job $job/$totalJobs · ${model.label} · '
                '$threads thread${threads == 1 ? '' : 's'}';
            final result = await _service.run(
              modelAsset: model.asset,
              threads: threads,
              mode: _mode,
              onProgress: (l, f) {
                if (!mounted) return;
                setState(() {
                  _status = '$label · $l';
                  _progress = ((job - 1) + f) / totalJobs;
                });
              },
            );
            if (!mounted) return;
            setState(() => _results.insert(0, result));
          }
        }
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Run-all failed at job $job.\n\nDetails: $e');
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
          _status = '';
        });
      }
    }
  }

  Future<void> _copy(String text) async {
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('On-device benchmark')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              '5 warm-up runs (discarded) + 50 timed runs on a fixed random '
              'input. Image decoding and UI are not included. For stable '
              'numbers: close other apps, keep the phone cool, and repeat '
              'each configuration a few times.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
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
              onSelectionChanged: _running
                  ? null
                  : (s) => setState(() => _modelIndex = s.first),
            ),
            const SizedBox(height: 16),
            Text('Threads', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<int>(
              segments: const [
                ButtonSegment<int>(value: 1, label: Text('1')),
                ButtonSegment<int>(value: 4, label: Text('4')),
              ],
              selected: {_threads},
              onSelectionChanged: _running
                  ? null
                  : (s) => setState(() => _threads = s.first),
            ),
            const SizedBox(height: 16),
            Text('Timed call', style: theme.textTheme.titleSmall),
            const SizedBox(height: 8),
            SegmentedButton<BenchmarkMode>(
              segments: const [
                ButtonSegment<BenchmarkMode>(
                    value: BenchmarkMode.invokeOnly,
                    label: Text('invoke() only')),
                ButtonSegment<BenchmarkMode>(
                    value: BenchmarkMode.runWithCopy,
                    label: Text('run() + copy')),
              ],
              selected: {_mode},
              onSelectionChanged:
                  _running ? null : (s) => setState(() => _mode = s.first),
            ),
            const SizedBox(height: 20),
            FilledButton.icon(
              onPressed: _running ? null : _run,
              icon: const Icon(Icons.speed),
              label: Text(_running ? 'Running…' : 'Run benchmark'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: _running ? null : _runAll,
              icon: const Icon(Icons.auto_mode),
              label: const Text('Run all 12 configs'),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size.fromHeight(48),
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
            if (_results.isNotEmpty) ...[
              const SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: Text(
                      'Results (${_results.length})',
                      style: theme.textTheme.titleMedium,
                    ),
                  ),
                  TextButton.icon(
                    onPressed: () => _copy(
                      _results.map((r) => r.toReport()).join('\n\n----\n\n'),
                    ),
                    icon: const Icon(Icons.copy_all),
                    label: const Text('Copy all'),
                  ),
                ],
              ),
              for (final r in _results) _ResultCard(result: r, onCopy: _copy),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result, required this.onCopy});

  final BenchmarkResult result;
  final Future<void> Function(String) onCopy;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final name = result.modelAsset.split('/').last;

    return Card(
      margin: const EdgeInsets.only(top: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '$name · ${result.threads} thread${result.threads == 1 ? '' : 's'}'
              ' · ${result.mode == BenchmarkMode.invokeOnly ? 'invoke()' : 'run()'}',
              style: theme.textTheme.titleSmall,
            ),
            const SizedBox(height: 6),
            Text(
              'Mean ${result.mean.toStringAsFixed(1)} ± '
              '${result.std.toStringAsFixed(1)} ms   '
              '(n = ${result.timedRuns})',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Median ${result.median.toStringAsFixed(1)} · '
              'Min ${result.min.toStringAsFixed(1)} · '
              'Max ${result.max.toStringAsFixed(1)} ms',
              style: theme.textTheme.bodySmall,
            ),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: () => onCopy(result.toReport()),
                icon: const Icon(Icons.copy, size: 18),
                label: const Text('Copy full report'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
