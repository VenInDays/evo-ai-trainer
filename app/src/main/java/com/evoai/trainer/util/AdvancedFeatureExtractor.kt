package com.evoai.trainer.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import java.util.zip.ZipInputStream
import kotlin.math.sqrt

enum class TrainingDomain(val displayName: String, val resolution: Int, val useEdges: Boolean) {
    ANIME_ART("Anime / Art", 32, false),
    GAMING_TEXTURE("Gaming / Textures", 64, true),
    SAFETY_NSFW("Safety / NSFW", 48, false),
    GENERAL("General", 32, true)
}

object AdvancedFeatureExtractor {
    
    data class ExtractionConfig(
        val domain: TrainingDomain = TrainingDomain.GENERAL,
        val resolution: Int = domain.resolution,
        val useEdges: Boolean = domain.useEdges,
        val useColorHistogram: Boolean = true,
        val histogramBins: Int = 8,
        val applyGCN: Boolean = true
    ) {
        val pixelFeatures: Int get() = resolution * resolution
        val edgeFeatures: Int get() = if (useEdges) resolution * resolution else 0
        val histogramFeatures: Int get() = if (useColorHistogram) histogramBins * 3 else 0
        val totalFeatureSize: Int get() = pixelFeatures + edgeFeatures + histogramFeatures
    }
    
    fun extractFromBitmap(bitmap: Bitmap, config: ExtractionConfig): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, config.resolution, config.resolution, true)
        val pixels = IntArray(config.resolution * config.resolution)
        scaled.getPixels(pixels, 0, config.resolution, 0, 0, config.resolution, config.resolution)
        
        // Convert to grayscale normalized [0,1]
        val grayscale = FloatArray(config.pixelFeatures)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = Color.red(pixel) / 255f
            val g = Color.green(pixel) / 255f
            val b = Color.blue(pixel) / 255f
            grayscale[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }
        
        // Apply Global Contrast Normalization
        val normalizedGray = if (config.applyGCN) applyGCN(grayscale) else grayscale
        
        val features = mutableListOf<Float>()
        features.addAll(normalizedGray.toList())
        
        // Sobel Edge Detection
        if (config.useEdges) {
            val edges = sobelEdgeDetection(normalizedGray, config.resolution, config.resolution)
            features.addAll(edges.toList())
        }
        
        // Color Histogram
        if (config.useColorHistogram) {
            val histogram = computeColorHistogram(pixels, config.histogramBins)
            features.addAll(histogram.toList())
        }
        
        if (bitmap !== scaled) scaled.recycle()
        return features.toFloatArray()
    }
    
    fun applyGCN(data: FloatArray): FloatArray {
        val mean = data.average().toFloat()
        val std = sqrt(data.map { (it - mean) * (it - mean) }.average()).toFloat().coerceAtLeast(1e-6f)
        return FloatArray(data.size) { i -> ((data[i] - mean) / std).coerceIn(-3f, 3f) / 3f * 0.5f + 0.5f }
    }
    
    fun sobelEdgeDetection(grayscale: FloatArray, width: Int, height: Int): FloatArray {
        val edges = FloatArray(width * height)
        // Sobel kernels
        // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
        // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val tl = grayscale[(y-1)*width+(x-1)]; val tc = grayscale[(y-1)*width+x]; val tr = grayscale[(y-1)*width+(x+1)]
                val ml = grayscale[y*width+(x-1)]; val mr = grayscale[y*width+(x+1)]
                val bl = grayscale[(y+1)*width+(x-1)]; val bc = grayscale[(y+1)*width+x]; val br = grayscale[(y+1)*width+(x+1)]
                
                val gx = -tl + tr - 2f*ml + 2f*mr - bl + br
                val gy = -tl - 2f*tc - tr + bl + 2f*bc + br
                edges[y * width + x] = sqrt(gx*gx + gy*gy).coerceIn(0f, 1f)
            }
        }
        return edges
    }
    
    fun computeColorHistogram(pixels: IntArray, bins: Int): FloatArray {
        val histogram = FloatArray(bins * 3) // R + G + B
        val binSize = 256f / bins
        for (pixel in pixels) {
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            histogram[(r / binSize).toInt().coerceIn(0, bins - 1)] += 1f
            histogram[bins + (g / binSize).toInt().coerceIn(0, bins - 1)] += 1f
            histogram[2 * bins + (b / binSize).toInt().coerceIn(0, bins - 1)] += 1f
        }
        // Normalize
        val maxVal = histogram.maxOrNull()?.coerceAtLeast(1f) ?: 1f
        for (i in histogram.indices) histogram[i] /= maxVal
        return histogram
    }
}
