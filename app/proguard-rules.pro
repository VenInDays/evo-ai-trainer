# Add project specific ProGuard rules here.
-keepattributes *Annotation*
-keep class com.evoai.trainer.nn.** { *; }
-keep class com.evoai.trainer.data.** { *; }

# MPAndroidChart
-keep class com.github.mikephil.charting.** { *; }

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.** { *; }
