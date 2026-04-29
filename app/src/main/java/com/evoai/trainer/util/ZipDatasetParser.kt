package com.evoai.trainer.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import java.util.zip.ZipInputStream

/**
 * V5 Universal Domain Dataset Parser.
 *
 * Upgrades from V4:
 * - Uses AdvancedFeatureExtractor for domain-specific feature extraction
 * - Auto-detects folder names from ZIP (no more hardcoded "like"/"nonlike")
 * - Domain-specific resolution and feature sizes
 * - Sobel Edge Detection, Color Histogram, Global Contrast Normalization
 * - Dynamic label names from folder structure
 */
object ZipDatasetParser {

    data class DatasetResult(
        val samples: List<Pair<FloatArray, Int>>,
        val trainSamples: List<Pair<FloatArray, Int>>,
        val valSamples: List<Pair<FloatArray, Int>>,
        val likeCount: Int,
        val nonlikeCount: Int,
        val featureSize: Int,
        val domain: TrainingDomain = TrainingDomain.GENERAL,
        val labelNames: Pair<String, String> = Pair("Like", "Non-like")
    ) {
        val totalSamples: Int get() = likeCount + nonlikeCount
    }

    private const val TRAIN_RATIO = 0.8f  // 80% training, 20% validation

    /**
     * V5: Parse ZIP with domain-specific feature extraction.
     * Auto-detects folder names for dynamic labels.
     */
    fun parseZip(zipInputStream: ZipInputStream, domain: TrainingDomain = TrainingDomain.GENERAL): DatasetResult {
        val config = AdvancedFeatureExtractor.ExtractionConfig(domain = domain)
        val samples = mutableListOf<Pair<FloatArray, Int>>()
        var likeCount = 0
        var nonlikeCount = 0

        // V5: Auto-detect folder names
        val folderNames = mutableListOf<String>()
        val folderSamples = mutableMapOf<String, MutableList<ByteArray>>()

        // First pass: scan entries and collect folder names + image data
        val allEntries = mutableListOf<Pair<String, ByteArray>>()
        var entry = zipInputStream.nextEntry
        while (entry != null) {
            val name = entry.name

            if (entry.isDirectory || name.lowercase().startsWith("__macosx") || name.lowercase().contains(".ds_store")) {
                zipInputStream.closeEntry()
                entry = zipInputStream.nextEntry
                continue
            }

            // Extract folder name (first directory component)
            val folderName = extractFolderName(name)
            if (folderName != null && folderName !in folderNames) {
                folderNames.add(folderName)
            }

            try {
                val imageData = readZipEntry(zipInputStream)
                if (folderName != null) {
                    if (folderSamples[folderName] == null) {
                        folderSamples[folderName] = mutableListOf()
                    }
                    folderSamples[folderName]!!.add(imageData)
                }
                allEntries.add(Pair(name, imageData))
            } catch (_: Exception) {
                // Skip corrupted entries
            }

            zipInputStream.closeEntry()
            entry = zipInputStream.nextEntry
        }

        zipInputStream.close()

        // V5: Determine label names from folder names
        // First folder = label 1 (positive), second folder = label 0 (negative)
        val labelPositive = if (folderNames.size >= 1) folderNames[0] else "Like"
        val labelNegative = if (folderNames.size >= 2) folderNames[1] else "Non-like"
        val labelNames = Pair(labelPositive, labelNegative)

        // Second pass: extract features from collected images
        for ((name, imageData) in allEntries) {
            val folderName = extractFolderName(name)
            val label: Int = when {
                folderName == null -> continue
                folderNames.size >= 2 -> {
                    if (folderName == folderNames[0]) 1 else 0
                }
                else -> {
                    // Fallback to old behavior
                    val nameLower = name.lowercase()
                    when {
                        nameLower.contains("nonlike/") || nameLower.contains("nonlike\\") ||
                        nameLower.contains("non_like/") || nameLower.contains("non-like/") -> 0
                        nameLower.contains("like/") || nameLower.contains("like\\") -> {
                            if (nameLower.contains("non")) 0 else 1
                        }
                        else -> continue
                    }
                }
            }

            try {
                val features = extractFeatures(imageData, config)
                if (features != null) {
                    samples.add(Pair(features, label))
                    if (label == 1) likeCount++ else nonlikeCount++
                }
            } catch (_: Exception) {
                // Skip corrupted images
            }
        }

        val featureSize = config.totalFeatureSize

        // V5: Cross-Validation Split — shuffle then 80/20 split
        val shuffled = samples.shuffled()
        val splitIndex = (shuffled.size * TRAIN_RATIO).toInt()
        val trainSamples = shuffled.take(splitIndex)
        val valSamples = shuffled.drop(splitIndex)

        return DatasetResult(samples, trainSamples, valSamples, likeCount, nonlikeCount, featureSize, domain, labelNames)
    }

    /**
     * V5: Extract folder name from ZIP entry path.
     * E.g. "Male/img001.jpg" → "Male", "dataset/Female/img002.jpg" → "Female"
     */
    private fun extractFolderName(path: String): String? {
        val normalized = path.replace("\\", "/")
        val parts = normalized.split("/")
        // Find the folder that directly contains the file (second-to-last part)
        // But also skip __MACOSX and .DS_Store
        for (i in parts.indices.reversed()) {
            val part = parts[i]
            if (part.isEmpty() || part.startsWith("__macosx") || part.contains(".ds_store") || part.contains(".")) {
                continue
            }
            // This is a folder name if it's not the last part (which is the filename)
            if (i < parts.size - 1) {
                return part
            }
        }
        // If only one level, use the first directory component
        if (parts.size >= 2) {
            return parts[0].takeIf { it.isNotEmpty() && !it.startsWith("__macosx") }
        }
        return null
    }

    /**
     * Extract features from a Bitmap for inference testing.
     * V5: Uses AdvancedFeatureExtractor with default GENERAL config.
     */
    fun extractFeaturesFromBitmap(bitmap: Bitmap): FloatArray {
        val config = AdvancedFeatureExtractor.ExtractionConfig(domain = currentDomain)
        return AdvancedFeatureExtractor.extractFromBitmap(bitmap, config)
    }

    /**
     * V5: Extract features from a Bitmap with a specific config.
     */
    fun extractFeaturesFromBitmap(bitmap: Bitmap, config: AdvancedFeatureExtractor.ExtractionConfig): FloatArray {
        return AdvancedFeatureExtractor.extractFromBitmap(bitmap, config)
    }

    /**
     * V5: Current domain for simple inference calls.
     */
    var currentDomain: TrainingDomain = TrainingDomain.GENERAL

    private fun readZipEntry(zis: ZipInputStream): ByteArray {
        val buffer = ByteArrayOutputStream()
        val buf = ByteArray(4096)
        var len: Int
        while (zis.read(buf).also { len = it } > 0) {
            buffer.write(buf, 0, len)
        }
        return buffer.toByteArray()
    }

    private fun extractFeatures(imageBytes: ByteArray, config: AdvancedFeatureExtractor.ExtractionConfig): FloatArray? {
        return try {
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.RGB_565
                inSampleSize = calculateInSampleSize(imageBytes, config.resolution)
            }

            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
                ?: return null

            val features = AdvancedFeatureExtractor.extractFromBitmap(bitmap, config)
            bitmap.recycle()
            features
        } catch (_: Exception) {
            null
        }
    }

    private fun calculateInSampleSize(imageBytes: ByteArray, targetSize: Int): Int {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)

        var inSampleSize = 1
        if (options.outHeight > targetSize * 2 || options.outWidth > targetSize * 2) {
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
