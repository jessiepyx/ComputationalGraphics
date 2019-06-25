# AR

## Environment

- Language: Java
- IDE: Android Studio 3.4.1
- Vuforia SDK 8.1.10
- OS: Android 8.0.0

## Build

Build project *vuforia/samples/VuforiaSamples-8-3-8* with Android Studio

- Target images can be downloaded from [Vuforia Website](https://library.vuforia.com/content/vuforia-library/en/articles/Solution/sample-apps-target-pdfs.html)
- *src/main/assets/ObjectRecognition* contains my object data *joy_OT.xml* and *joy_OT.dat*
- Change the lisence in *src/main/java/com/vuforia/engine/SampleApplicationSession.java* (the last argument of the following function) to your own

```
Vuforia.setInitParameters(session.mActivityRef.get(), session.mVuforiaFlags, "")
```

