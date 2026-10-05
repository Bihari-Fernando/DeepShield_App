import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/detection_result.dart';
import '../services/deepfake_detection_service.dart';
import '../widgets/loading_overlay.dart';
import '../widgets/primary_button.dart';
import 'result_screen.dart';

/// Landing screen: lets the user pick an image (gallery or camera)
/// and kicks off on-device detection.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.detectionService});

  final DeepfakeDetectionService detectionService;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

  bool _isAnalyzing = false;
  String? _errorMessage;

  Future<bool> _ensurePermission(ImageSource source) async {
    // On modern Android, image_picker uses the system photo picker for
    // ImageSource.gallery, which needs no runtime permission. Only the
    // camera needs an explicit request here.
    if (source == ImageSource.camera) {
      final status = await Permission.camera.request();
      return status.isGranted;
    }
    return true;
  }

  Future<void> _pickImage(ImageSource source) async {
    setState(() => _errorMessage = null);

    final granted = await _ensurePermission(source);
    if (!granted) {
      setState(() {
        _errorMessage = source == ImageSource.camera
            ? 'Camera permission was denied. Enable it in system settings to take a photo.'
            : 'Permission was denied. Enable photo access in system settings.';
      });
      return;
    }

    try {
      final XFile? picked = await _picker.pickImage(
        source: source,
        imageQuality: 100,
      );
      if (picked == null) return;
      await _runDetection(File(picked.path));
    } catch (e) {
      setState(() {
        _errorMessage =
            'Could not open the ${source == ImageSource.camera ? 'camera' : 'gallery'}. '
            'Please try again.';
      });
    }
  }

  Future<void> _runDetection(File imageFile) async {
    setState(() => _isAnalyzing = true);

    try {
      final DetectionResult result =
          await widget.detectionService.analyzeImage(imageFile);

      if (!mounted) return;
      setState(() => _isAnalyzing = false);

      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultScreen(imageFile: imageFile, result: result),
        ),
      );
    } on DeepfakeDetectionException catch (e) {
      if (!mounted) return;
      setState(() {
        _isAnalyzing = false;
        _errorMessage = e.message;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isAnalyzing = false;
        _errorMessage =
            'Something went wrong while analyzing the image. Please try another photo.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Stack(
        children: [
          SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight,
                    ),
                    child: IntrinsicHeight(
                      child: Column(
                        children: [
                          const Spacer(),
                          Icon(
                            Icons.shield_outlined,
                            size: 88,
                            color: theme.colorScheme.primary,
                          ),
                          const SizedBox(height: 24),
                          Text(
                            'DeepShield',
                            style: theme.textTheme.headlineMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'On-device deepfake detection.\nYour images never leave your phone.',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 40),
                          if (_errorMessage != null) ...[
                            _ErrorBanner(message: _errorMessage!),
                            const SizedBox(height: 24),
                          ],
                          PrimaryButton(
                            label: 'Choose from Gallery',
                            icon: Icons.photo_library_outlined,
                            onPressed: () => _pickImage(ImageSource.gallery),
                          ),
                          const SizedBox(height: 16),
                          OutlinedButton.icon(
                            onPressed: () => _pickImage(ImageSource.camera),
                            icon: const Icon(Icons.camera_alt_outlined),
                            label: const Text('Take a Photo'),
                            style: OutlinedButton.styleFrom(
                              minimumSize: const Size.fromHeight(52),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(14),
                              ),
                            ),
                          ),
                          const Spacer(flex: 2),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.lock_outline,
                                size: 16,
                                color: theme.colorScheme.onSurfaceVariant,
                              ),
                              const SizedBox(width: 6),
                              Flexible(
                                child: Text(
                                  'All analysis runs locally. Nothing is uploaded. No account needed.',
                                  style: theme.textTheme.bodySmall?.copyWith(
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 24),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          if (_isAnalyzing)
            const LoadingOverlay(message: 'Analyzing image locally…'),
        ],
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: theme.colorScheme.onErrorContainer),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: TextStyle(color: theme.colorScheme.onErrorContainer),
            ),
          ),
        ],
      ),
    );
  }
}
