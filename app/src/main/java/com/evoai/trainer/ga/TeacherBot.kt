package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork

/**
 * Teacher Bot: Evaluates BOT outputs against dataset labels.
 * Computes accuracy, precision, recall, F1-score.
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
     * Evaluate a single neural network against the dataset.
     */
    fun evaluate(
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
     * Quick accuracy check for a network.
     */
    fun quickAccuracy(network: NeuralNetwork, dataset: List<Pair<FloatArray, Int>>): Float {
        if (dataset.isEmpty()) return 0f
        var correct = 0
        for ((input, label) in dataset) {
            if (network.predict(input) == label) correct++
        }
        return correct.toFloat() / dataset.size * 100f
    }
}
