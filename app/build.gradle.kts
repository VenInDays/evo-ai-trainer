plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.evoai.trainer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.evoai.trainer"
        minSdk = 24
        targetSdk = 34
        versionCode = 3
        versionName = "3.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        create("release") {
            // Read signing config from project properties, local.properties, or env vars
            val ksPath = findProperty("KEYSTORE_PATH") as? String
                ?: System.getenv("KEYSTORE_PATH")
            val ksPassword = findProperty("KEYSTORE_PASSWORD") as? String
                ?: System.getenv("KEYSTORE_PASSWORD")
            val ksAlias = findProperty("KEY_ALIAS") as? String
                ?: System.getenv("KEY_ALIAS")
            val ksKeyPassword = findProperty("KEY_PASSWORD") as? String
                ?: System.getenv("KEY_PASSWORD")

            if (ksPath != null && ksPassword != null && ksAlias != null && ksKeyPassword != null) {
                storeFile = rootProject.file(ksPath)
                storePassword = ksPassword
                keyAlias = ksAlias
                keyPassword = ksKeyPassword
                // Enable v2+v3 signing
                enableV1Signing = false
                enableV2Signing = true
                enableV3Signing = true
                enableV4Signing = true
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Use release signing if keystore is available and exists, otherwise debug
            val ksPath = findProperty("KEYSTORE_PATH") as? String
                ?: System.getenv("KEYSTORE_PATH")
            if (ksPath != null && rootProject.file(ksPath).exists()) {
                signingConfig = signingConfigs.getByName("release")
            } else {
                signingConfig = signingConfigs.getByName("debug")
            }
        }
        debug {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // AndroidX Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.cardview:cardview:1.0.0")

    // Lifecycle
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")

    // Room Database
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // WorkManager
    implementation("androidx.work:work-runtime-ktx:2.9.0")

    // Activity Result API
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // MPAndroidChart for fitness graph
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")

    // Gson for serialization
    implementation("com.google.code.gson:gson:2.10.1")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
