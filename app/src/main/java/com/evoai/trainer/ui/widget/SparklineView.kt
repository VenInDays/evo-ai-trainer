package com.evoai.trainer.ui.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.evoai.trainer.R

/**
 * Custom View that draws a tiny sparkline chart showing the last N fitness scores.
 * Used in the 2x5 Bot Grid to visualize each model's fitness trend.
 */
class SparklineView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var data: FloatArray = floatArrayOf()
    private var lineColor: Int = ContextCompat.getColor(context, R.color.slate_blue_training)
    private var fillColor: Int = ContextCompat.getColor(context, R.color.slate_blue_training)

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = lineColor
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = fillColor
        alpha = 40 // 15% opacity
    }

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = lineColor
    }

    fun setData(values: FloatArray) {
        data = values
        invalidate()
    }

    fun setLineColor(color: Int) {
        lineColor = color
        linePaint.color = color
        fillColor = color
        fillPaint.color = color
        fillPaint.alpha = 40
        dotPaint.color = color
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (data.size < 2) {
            // Not enough data points — draw a flat line
            val y = height / 2f
            linePaint.alpha = 60
            canvas.drawLine(0f, y, width.toFloat(), y, linePaint)
            linePaint.alpha = 255
            return
        }

        val w = width.toFloat()
        val h = height.toFloat()
        val padding = 4f
        val drawW = w - padding * 2
        val drawH = h - padding * 2

        // Find min/max for scaling
        val minVal = data.minOrNull() ?: 0f
        val maxVal = data.maxOrNull() ?: 1f
        val range = (maxVal - minVal).coerceAtLeast(0.01f)

        // Build points
        val points = mutableListOf<Pair<Float, Float>>()
        for (i in data.indices) {
            val x = padding + (i.toFloat() / (data.size - 1).coerceAtLeast(1)) * drawW
            val y = padding + drawH - ((data[i] - minVal) / range) * drawH
            points.add(Pair(x, y))
        }

        // Draw fill area
        val fillPath = Path()
        fillPath.moveTo(points[0].first, points[0].second)
        for (i in 1 until points.size) {
            fillPath.lineTo(points[i].first, points[i].second)
        }
        fillPath.lineTo(points.last().first, h)
        fillPath.lineTo(points[0].first, h)
        fillPath.close()
        canvas.drawPath(fillPath, fillPaint)

        // Draw line
        val linePath = Path()
        linePath.moveTo(points[0].first, points[0].second)
        for (i in 1 until points.size) {
            linePath.lineTo(points[i].first, points[i].second)
        }
        canvas.drawPath(linePath, linePaint)

        // Draw dot on the last point
        val lastPoint = points.last()
        canvas.drawCircle(lastPoint.first, lastPoint.second, 3f, dotPaint)
    }
}
