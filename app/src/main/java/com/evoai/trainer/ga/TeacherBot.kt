package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlin.math.min

/**
 * V4 Commercial-Grade Teacher Bot.
 *
 * Upgrades:
 * - Computes Binary Cross-Entropy (BCE) loss alongside accuracy
 * - Supports cross-validation (evaluate on validation set for generalization)
 * - Confusion Matrix computation for real-time display
 * - Lower-loss tiebreaker: when accuracy is equal, model with lower loss is more confident
 */
object TeacherBot {

    data class EvaluationResult(
        val accuracy: Float,
        val precision: Float,
        val recall: Float,
        val f1Score: Float,
        val correctCount: Int,
        val totalCount: Int,
        val truePositives: Int = 0,
        val falsePositives: Int = 0,
        val trueNegatives: Int = 0,
        val falseNegatives: Int = 0,
        val avgLoss: Float = 0f
    )

    /**
     * V4: Evaluate a single network on a batch.
     * Returns (fitness, accuracy, correctCount, avgLoss).
     * Loss is BCE (Binary Cross-Entropy) — lower means more confident correct predictions.
     */
    fun evaluateBatch(
        network: NeuralNetwork,
        batch: List<Pair<FloatArray, Int>>
    ): Triple<Float, Float, Int> {
        if (batch.isEmpty()) return Triple(0f, 0f, 0)

        var correct = 0
        var fitness = 0f
        var totalLoss = 0f

        for ((input, label) in batch) {
            val output = network.forward(input)[0]
            val prediction = if (output > 0.5f) 1 else 0

            if (prediction == label) {
                correct++
                fitness += 1f
            } else {
                // Partial credit based on confidence
                val confidence = if (label == 1) output else 1f - output
                fitness += confidence * 0.5f
            }

            // V4: Compute BCE loss
            val p = output.coerceIn(1e-7f, 1f - 1e-7f)
            val target = label.toFloat()
            totalLoss += -(target * kotlin.math.ln(p.toDouble()).toFloat() +
                    (1f - target) * kotlin.math.ln((1f - p).toDouble()).toFloat())
        }

        val accuracy = correct.toFloat() / batch.size * 100f
        val normalizedFitness = fitness / batch.size
        return Triple(normalizedFitness, accuracy, correct)
    }

    /**
     * V5: Evaluate with loss and dropout — returns fitness, accuracy, correctCount, avgLoss.
     * Uses forwardWithDropout during evaluation for regularization.
     */
    fun evaluateBatchWithLossAndDropout(
        network: NeuralNetwork,
        batch: List<Pair<FloatArray, Int>>,
        dropRate: Float = 0.1f
    ): Quadruple<Float, Float, Int, Float> {
        if (batch.isEmpty()) return Quadruple(0f, 0f, 0, 0f)

        var correct = 0
        var fitness = 0f
        var totalLoss = 0f

        for ((input, label) in batch) {
            val output = network.forwardWithDropout(input, dropRate)[0]
            val prediction = if (output > 0.5f) 1 else 0

            if (prediction == label) {
                correct++
                fitness += 1f
            } else {
                val confidence = if (label == 1) output else 1f - output
                fitness += confidence * 0.5f
            }

            // BCE loss
            val p = output.coerceIn(1e-7f, 1f - 1e-7f)
            val target = label.toFloat()
            totalLoss += -(target * kotlin.math.ln(p.toDouble()).toFloat() +
                    (1f - target) * kotlin.math.ln((1f - p).toDouble()).toFloat())
        }

        val accuracy = correct.toFloat() / batch.size * 100f
        val normalizedFitness = fitness / batch.size
        val avgLoss = totalLoss / batch.size
        return Quadruple(normalizedFitness, accuracy, correct, avgLoss)
    }

    /**
     * V4: Evaluate with loss — returns fitness, accuracy, correctCount, avgLoss.
     */
    fun evaluateBatchWithLoss(
        network: NeuralNetwork,
        batch: List<Pair<FloatArray, Int>>
    ): Quadruple<Float, Float, Int, Float> {
        if (batch.isEmpty()) return Quadruple(0f, 0f, 0, 0f)

        var correct = 0
        var fitness = 0f
        var totalLoss = 0f

        for ((input, label) in batch) {
            val output = network.forward(input)[0]
            val prediction = if (output > 0.5f) 1 else 0

            if (prediction == label) {
                correct++
                fitness += 1f
            } else {
                val confidence = if (label == 1) output else 1f - output
                fitness += confidence * 0.5f
            }

            // BCE loss
            val p = output.coerceIn(1e-7f, 1f - 1e-7f)
            val target = label.toFloat()
            totalLoss += -(target * kotlin.math.ln(p.toDouble()).toFloat() +
                    (1f - target) * kotlin.math.ln((1f - p).toDouble()).toFloat())
        }

        val accuracy = correct.toFloat() / batch.size * 100f
        val normalizedFitness = fitness / batch.size
        val avgLoss = totalLoss / batch.size
        return Quadruple(normalizedFitness, accuracy, correct, avgLoss)
    }

    /**
     * Full evaluation with precision/recall/F1 + Confusion Matrix + Loss.
     */
    fun evaluateFull(
        network: NeuralNetwork,
        dataset: List<Pair<FloatArray, Int>>
    ): EvaluationResult {
        if (dataset.isEmpty()) {
            return EvaluationResult(0f, 0f, 0f, 0f, 0, 0)
        }

        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0
        var totalLoss = 0f

        for ((input, label) in dataset) {
            val output = network.forward(input)[0]
            val prediction = if (output > 0.5f) 1 else 0

            // BCE loss
            val p = output.coerceIn(1e-7f, 1f - 1e-7f)
            val target = label.toFloat()
            totalLoss += -(target * kotlin.math.ln(p.toDouble()).toFloat() +
                    (1f - target) * kotlin.math.ln((1f - p).toDouble()).toFloat())

            when {
                prediction == 1 && label == 1 -> truePositives++
                prediction == 1 && label == 0 -> falsePositives++
                prediction == 0 && label == 0 -> trueNegatives++
                prediction == 0 && label == 1 -> falseNegatives++
            }
        }

        val totalCount = dataset.size
        val correctCount = truePositives + trueNegatives
        val accuracy = correctCount.toFloat() / totalCount * 100f

        val precision = if (truePositives + falsePositives > 0) {
            truePositives.toFloat() / (truePositives + falsePositives)
        } else 0f

        val recall = if (truePositives + falseNegatives > 0) {
            truePositives.toFloat() / (truePositives + falseNegatives)
        } else 0f

        val f1Score = if (precision + recall > 0) {
            2 * precision * recall / (precision + recall)
        } else 0f

        val avgLoss = totalLoss / totalCount

        return EvaluationResult(
            accuracy, precision, recall, f1Score, correctCount, totalCount,
            truePositives, falsePositives, trueNegatives, falseNegatives,
            avgLoss
        )
    }

    /**
     * Select a random batch from the dataset.
     */
    fun selectBatch(
        dataset: List<Pair<FloatArray, Int>>,
        batchSize: Int = 32
    ): List<Pair<FloatArray, Int>> {
        if (dataset.size <= batchSize) return dataset
        return dataset.shuffled().take(batchSize)
    }
}

/**
 * V4: Quadruple data class for evaluateBatchWithLoss return type.
 */
data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D
)
