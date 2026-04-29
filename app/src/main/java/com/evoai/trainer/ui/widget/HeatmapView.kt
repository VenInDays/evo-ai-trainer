package com.evoai.trainer.ui.widget

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class HeatmapView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var heatmapData: FloatArray = floatArrayOf()
    private var gridWidth: Int = 0
    private var gridHeight: Int = 0

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun setData(data: FloatArray, width: Int, height: Int) {
        heatmapData = data
        gridWidth = width
        gridHeight = height
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (heatmapData.isEmpty() || gridWidth == 0 || gridHeight == 0) return

        val cellW = width.toFloat() / gridWidth
        val cellH = height.toFloat() / gridHeight

        for (y in 0 until gridHeight) {
            for (x in 0 until gridWidth) {
                val idx = y * gridWidth + x
                if (idx >= heatmapData.size) break
                val value = heatmapData[idx].coerceIn(0f, 1f)
                paint.color = heatmapColor(value)
                canvas.drawRect(
                    x * cellW, y * cellH,
                    (x + 1) * cellW, (y + 1) * cellH,
                    paint
                )
            }
        }
    }

    private fun heatmapColor(value: Float): Int {
        // Blue(0) → Cyan(0.25) → Green(0.5) → Yellow(0.75) → Red(1.0)
        val r: Int
        val g: Int
        val b: Int
        when {
            value < 0.25f -> {
                val t = value / 0.25f
                r = 0; g = (t * 255).toInt(); b = 255
            }
            value < 0.5f -> {
                val t = (value - 0.25f) / 0.25f
                r = 0; g = 255; b = ((1 - t) * 255).toInt()
            }
            value < 0.75f -> {
                val t = (value - 0.5f) / 0.25f
                r = (t * 255).toInt(); g = 255; b = 0
            }
            else -> {
                val t = (value - 0.75f) / 0.25f
                r = 255; g = ((1 - t) * 255).toInt(); b = 0
            }
        }
        return Color.argb(180, r.coerceIn(0, 255), g.coerceIn(0, 255), b.coerceIn(0, 255))
    }
}
