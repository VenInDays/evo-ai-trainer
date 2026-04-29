package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.math.min

/**
 * Teacher Bot: Holds ground truth labels and evaluates models.
 * Picks random batches for training and challenges all 10 models.
 */
object TeacherBot {

    data class EvaluationResult(
        val accuracy: Float,
        val precision: Float,
        val recall: Float,
        val f1Score: Float,
        val correctCount: Int,
        val totalCount: Int
    )

    /**
     * Evaluate a single network on a batch.
     * Returns (fitness, accuracy, correctCount).
     */
    fun evaluateBatch(
        network: NeuralNetwork,
        batch: List<Pair<FloatArray, Int>>
    ): Triple<Float, Float, Int> {
        if (batch.isEmpty()) return Triple(0f, 0f, 0)

        var correct = 0
        var fitness = 0f

        for ((input, label) in batch) {
            val prediction = network.predict(input)
            if (prediction == label) {
                correct++
                fitness += 1f
            } else {
                // Partial credit based on confidence
                val output = network.forward(input)[0]
                val confidence = if (label == 1) output else 1f - output
                fitness += confidence * 0.5f
            }
        }

        val accuracy = correct.toFloat() / batch.size * 100f
        val normalizedFitness = fitness / batch.size
        return Triple(normalizedFitness, accuracy, correct)
    }

    /**
     * Full evaluation with precision/recall/F1.
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

        for ((input, label) in dataset) {
            val prediction = network.predict(input)
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

        return EvaluationResult(accuracy, precision, recall, f1Score, correctCount, totalCount)
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
