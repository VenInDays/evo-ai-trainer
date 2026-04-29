package com.evoai.trainer.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import java.util.zip.ZipInputStream

/**
 * V4 Commercial-Grade Dataset Parser.
 *
 * Upgrades from V3:
 * - Feature Normalization: pixels 0-255 → 0.0-1.0 (already done in V3, now explicit)
 * - Cross-Validation Split: 80% Training / 20% Validation
 * - Normalized features ensure stable sigmoid inputs, preventing "shaky" predictions
 * - Hard Examples list for Manual Override feature
 */
object ZipDatasetParser {

    data class DatasetResult(
        val samples: List<Pair<FloatArray, Int>>,
        val trainSamples: List<Pair<FloatArray, Int>>,
        val valSamples: List<Pair<FloatArray, Int>>,
        val likeCount: Int,
        val nonlikeCount: Int,
        val featureSize: Int
    ) {
        val totalSamples: Int get() = likeCount + nonlikeCount
    }

    private const val FEATURE_WIDTH = 16
    private const val FEATURE_HEIGHT = 16
    private const val TRAIN_RATIO = 0.8f  // 80% training, 20% validation

    fun parseZip(zipInputStream: ZipInputStream): DatasetResult {
        val samples = mutableListOf<Pair<FloatArray, Int>>()
        var likeCount = 0
        var nonlikeCount = 0

        var entry = zipInputStream.nextEntry
        while (entry != null) {
            val name = entry.name.lowercase()

            if (entry.isDirectory || name.startsWith("__macosx") || name.contains(".ds_store")) {
                entry = zipInputStream.nextEntry
                continue
            }

            val label: Int = when {
                name.contains("nonlike/") || name.contains("nonlike\\") ||
                name.contains("non_like/") || name.contains("non-like/") -> 0
                name.contains("like/") || name.contains("like\\") -> {
                    if (name.contains("non")) 0 else 1
                }
                else -> {
                    entry = zipInputStream.nextEntry
                    continue
                }
            }

            try {
                val imageData = readZipEntry(zipInputStream)
                val features = extractFeatures(imageData)

                if (features != null) {
                    samples.add(Pair(features, label))
                    if (label == 1) likeCount++ else nonlikeCount++
                }
            } catch (_: Exception) {
                // Skip corrupted images
            }

            zipInputStream.closeEntry()
            entry = zipInputStream.nextEntry
        }

        zipInputStream.close()

        val featureSize = FEATURE_WIDTH * FEATURE_HEIGHT

        // V4: Cross-Validation Split — shuffle then 80/20 split
        val shuffled = samples.shuffled()
        val splitIndex = (shuffled.size * TRAIN_RATIO).toInt()
        val trainSamples = shuffled.take(splitIndex)
        val valSamples = shuffled.drop(splitIndex)

        return DatasetResult(samples, trainSamples, valSamples, likeCount, nonlikeCount, featureSize)
    }

    /**
     * Extract features from a Bitmap for inference testing.
     * V4: Explicit normalization to 0.0-1.0 range.
     */
    fun extractFeaturesFromBitmap(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, FEATURE_WIDTH, FEATURE_HEIGHT, true)
        val features = FloatArray(FEATURE_WIDTH * FEATURE_HEIGHT)
        val pixels = IntArray(FEATURE_WIDTH * FEATURE_HEIGHT)
        scaled.getPixels(pixels, 0, FEATURE_WIDTH, 0, 0, FEATURE_WIDTH, FEATURE_HEIGHT)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            // V4: Normalize to [0.0, 1.0] — critical for stable sigmoid inputs
            features[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
        }

        if (bitmap !== scaled) scaled.recycle()
        return features
    }

    private fun readZipEntry(zis: ZipInputStream): ByteArray {
        val buffer = ByteArrayOutputStream()
        val buf = ByteArray(4096)
        var len: Int
        while (zis.read(buf).also { len = it } > 0) {
            buffer.write(buf, 0, len)
        }
        return buffer.toByteArray()
    }

    private fun extractFeatures(imageBytes: ByteArray): FloatArray? {
        return try {
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.RGB_565
                inSampleSize = calculateInSampleSize(imageBytes)
            }

            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
                ?: return null

            val scaled = Bitmap.createScaledBitmap(bitmap, FEATURE_WIDTH, FEATURE_HEIGHT, true)

            val features = FloatArray(FEATURE_WIDTH * FEATURE_HEIGHT)
            val pixels = IntArray(FEATURE_WIDTH * FEATURE_HEIGHT)
            scaled.getPixels(pixels, 0, FEATURE_WIDTH, 0, 0, FEATURE_WIDTH, FEATURE_HEIGHT)

            for (i in pixels.indices) {
                val pixel = pixels[i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                // V4: Normalize to [0.0, 1.0]
                features[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            }

            if (bitmap !== scaled) scaled.recycle()
            bitmap.recycle()

            features
        } catch (_: Exception) {
            null
        }
    }

    private fun calculateInSampleSize(imageBytes: ByteArray): Int {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)

        val targetSize = FEATURE_WIDTH * 2
        var inSampleSize = 1
        if (options.outHeight > targetSize || options.outWidth > targetSize) {
            val halfHeight = options.outHeight / 2
            val halfWidth = options.outWidth / 2
            while (halfHeight / inSampleSize >= targetSize &&
                halfWidth / inSampleSize >= targetSize
            ) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }
}
