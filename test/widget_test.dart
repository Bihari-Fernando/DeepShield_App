import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:deepshield/app.dart';
import 'package:deepshield/models/detection_result.dart';
import 'package:deepshield/services/eval_utils.dart';

void main() {
  testWidgets('Home screen shows title and image source buttons',
      (WidgetTester tester) async {
    await tester.pumpWidget(const DeepShieldApp());

    expect(find.text('DeepShield'), findsOneWidget);
    expect(find.byIcon(Icons.shield_outlined), findsOneWidget);
    expect(find.text('Choose from Gallery'), findsOneWidget);
    expect(find.text('Take a Photo'), findsOneWidget);

    // No error banner or loading overlay before the user picks an image.
    expect(find.byIcon(Icons.error_outline), findsNothing);
    expect(find.byType(CircularProgressIndicator), findsNothing);
  });

  group('DetectionResult', () {
    test('formats a FAKE result', () {
      const result = DetectionResult(
        label: DetectionLabel.fake,
        confidence: 0.924,
        inferenceTimeMs: 42,
      );

      expect(result.isFake, isTrue);
      expect(result.labelText, 'FAKE');
      expect(result.confidencePercentText, '92.4%');
    });

    test('formats a REAL result and clamps confidence', () {
      const result = DetectionResult(
        label: DetectionLabel.real,
        confidence: 1.5,
        inferenceTimeMs: 10,
      );

      expect(result.isFake, isFalse);
      expect(result.labelText, 'REAL');
      expect(result.confidencePercentText, '100.0%');
    });

    test('copyWith replaces only the given fields', () {
      const original = DetectionResult(
        label: DetectionLabel.real,
        confidence: 0.8,
        inferenceTimeMs: 5,
      );
      final copy = original.copyWith(label: DetectionLabel.fake);

      expect(copy.label, DetectionLabel.fake);
      expect(copy.confidence, 0.8);
      expect(copy.inferenceTimeMs, 5);
    });
  });

  group('eval_utils', () {
    test('fnv1a32 matches the published FNV-1a test vectors', () {
      expect(fnv1a32(<int>[]), 0x811c9dc5);
      expect(fnv1a32('a'.codeUnits), 0xe40c292c);
      expect(fnv1a32('foobar'.codeUnits), 0xbf9cf968);
    });

    test('fnv1a32Hex is zero-padded to 8 digits', () {
      expect(fnv1a32Hex(<int>[]), '811c9dc5');
      expect(fnv1a32Hex('a'.codeUnits), 'e40c292c');
    });

    test('csvField quotes only when needed', () {
      expect(csvField('plain.jpg'), 'plain.jpg');
      expect(csvField('a,b.jpg'), '"a,b.jpg"');
      expect(csvField('say "hi".jpg'), '"say ""hi"".jpg"');
    });

    test('EvalTally counts the four outcomes', () {
      const hash = '00000000';
      EvalRecord rec(String truth, bool predictedFake) => EvalRecord(
            model: 'int8',
            resize: 'linear',
            batchLabel: truth,
            hash: hash,
            name: 'x.jpg',
            pReal: 0.5,
            predictedFake: predictedFake,
            inferenceMs: 1,
          );

      final tally = EvalTally()
        ..add(rec('REAL', false))
        ..add(rec('REAL', true))
        ..add(rec('FAKE', true))
        ..add(rec('FAKE', true))
        ..add(rec('FAKE', false));

      expect(tally.realAsReal, 1);
      expect(tally.realAsFake, 1);
      expect(tally.fakeAsFake, 2);
      expect(tally.fakeAsReal, 1);
      expect(tally.total, 5);
      expect(tally.accuracy, closeTo(3 / 5, 1e-12));
    });

    test('CSV row has the documented columns', () {
      const r = EvalRecord(
        model: 'int8',
        resize: 'nearest',
        batchLabel: 'FAKE',
        hash: 'e40c292c',
        name: 'a,b.jpg',
        pReal: 0.0123456789,
        predictedFake: true,
        inferenceMs: 281,
      );
      expect(EvalRecord.csvHeader,
          'model,resize,batch_label,fnv1a32,name,p_real,pred,ms');
      expect(r.toCsvRow(), 'int8,nearest,FAKE,e40c292c,"a,b.jpg",0.012346,FAKE,281');
    });
  });
}
