# DeepShield

Lightweight, on-device deepfake detection for Android. An image is
picked from the gallery or camera, resized and quantized to match an
INT8 TensorFlow Lite model, classified **entirely on the phone**, and
the result (REAL/FAKE, confidence, inference time) can be shared
through Android's native share sheet.

No servers. No accounts. No image ever leaves the device.

## Project structure

```
lib/
  main.dart                          # entry point
  app.dart                           # MaterialApp, theme, owns the detection service
  models/
    detection_result.dart            # DetectionResult + DeepfakeDetectionException
  services/
    deepfake_detection_service.dart  # the only file that talks to TFLite
  screens/
    home_screen.dart                 # image picking + permission handling
    result_screen.dart                # REAL/FAKE result + Share Result
  widgets/
    primary_button.dart
    confidence_indicator.dart
    loading_overlay.dart
  theme/
    app_theme.dart                   # light/dark Material 3 theme
assets/
  models/
    deepshield_int8.tflite           # <-- put your real model here (see MODEL_INTEGRATION.md)
android/                             # standard Flutter Android project
MODEL_INTEGRATION.md                 # how to swap in your own INT8 model
```

`DeepfakeDetectionService` is the single seam between the app and
TensorFlow Lite — swapping in a different INT8 model (different input
size, different quantization) generally requires **no other code
change**. See `MODEL_INTEGRATION.md`.

## Getting started

**Prerequisites:** Flutter 3.22+ (stable channel recommended), Android
SDK with API 36, JDK 17.

```bash
flutter --version   # confirm Flutter is installed
flutter pub get
```

> **Before your first build:** this template ships `android/gradle/wrapper/gradle-wrapper.properties`
> (which pins the Gradle version) but not the binary `gradle-wrapper.jar` that
> normally sits next to it — binary files can't be generated here. Fix it with
> **either**:
> - `flutter create --platforms=android .` — regenerates the wrapper jar and
>   `gradlew`/`gradlew.bat` along with the rest of `android/` for your exact
>   Flutter version (recommended — see below), or
> - opening the project in Android Studio, which downloads the wrapper jar
>   automatically on first Gradle sync.
>
> Without one of these, `flutter run` / `flutter build apk` will fail with a
> "could not find or load main class GradleWrapperMain" error.

The `android/` folder included here is a standard Flutter Android
project (Groovy Gradle, AGP 8.12, Kotlin 2.2, Java 17 — matching what
`share_plus`/`image_picker`'s current versions expect). If your local
Flutter/Android tooling is a different version and Gradle sync fails,
the most reliable fix is to regenerate the platform folder for your
exact installed version instead of debugging Gradle by hand:

```bash
flutter create --platforms=android --org com.deepshield .
```

This only touches `android/` (and `ios/` if you add `,ios`) — it
will not overwrite `lib/`, `pubspec.yaml`'s dependencies, or `assets/`.
Re-check `android/app/build.gradle` afterwards for the `aaptOptions {
noCompress "tflite" }` block (see MODEL_INTEGRATION.md §5) since a
fresh `flutter create` won't add it.

### 1. Add your model

Drop your trained `.tflite` file at `assets/models/deepshield_int8.tflite`.
Until you do, the app will show a friendly "model not available"
error instead of crashing — see `MODEL_INTEGRATION.md` for exactly
what the file needs to look like and how to produce one.

### 2. Run

```bash
flutter devices          # confirm a device/emulator is attached
flutter run
```

### 3. Build a release APK

```bash
flutter build apk --release
```

The output APK is signed with the debug keystore so this works out
of the box for testing. Replace `signingConfigs.debug` in
`android/app/build.gradle` with your own signing config before
publishing anywhere.

## Permissions

| Permission | Why |
|---|---|
| `CAMERA` | Optional "Take a Photo" button |
| `READ_EXTERNAL_STORAGE` (API ≤ 32 only) | Legacy gallery access; API 33+ uses the system photo picker, which needs no permission |
| `INTERNET` (debug/profile builds only) | Flutter's hot-reload / DevTools connection — **not present in release builds** |

The release build never requests `INTERNET`, matching the app's
fully-offline design.

## Privacy

- All image analysis runs locally via TensorFlow Lite; no image or
  result is ever uploaded anywhere.
- No account, sign-in, or analytics SDK is included.
- The app icon shown in the launcher is an original geometric shield
  mark generated for this project.

## Known limitations of this template

- `assets/models/deepshield_int8.tflite` is **not included** — you
  must supply a real trained model (binary model weights can't be
  generated as part of this scaffold). Until then, detection will
  show an error instead of a result.
- Launcher icons are legacy (non-adaptive) PNGs at all five mipmap
  densities — simple and works everywhere, but you may want to
  generate a proper adaptive icon (foreground + background layers)
  before publishing.
- The Android project targets a recent toolchain (AGP 8.12, Kotlin
  2.2, compileSdk/targetSdk 35, Java 17) to match current
  `share_plus`/`image_picker` requirements. If you're on an older
  Flutter install, run `flutter create --platforms=android .` as
  described above rather than manually downgrading versions.
