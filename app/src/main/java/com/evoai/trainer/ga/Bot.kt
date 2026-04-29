package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork

/**
 * Represents a single BOT in the evolutionary population.
 * Each BOT wraps a Neural Network and tracks its fitness score.
 */
data class Bot(
    val id: Int,
    val network: NeuralNetwork,
    var fitness: Float = 0f,
    var accuracy: Float = 0f,
    var status: BotStatus = BotStatus.READY
)

enum class BotStatus {
    READY,
    TRAINING,
    EVALUATED,
    BEST,
    ELIMINATED
}
