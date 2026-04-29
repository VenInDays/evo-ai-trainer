package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlin.math.max

/**
 * Genetic Algorithm engine that manages the evolutionary training loop.
 *
 * 1. Evaluate all BOTs on the dataset
 * 2. Kill the 9 weakest BOTs
 * 3. Clone the strongest BOT with mutation into 9 new variants
 * 4. Repeat until accuracy > 90%
 */
class GeneticTrainer(
    private val populationSize: Int = 10,
    private val inputSize: Int,
    private val mutationRate: Float = 0.05f
) {
    private var bots: MutableList<Bot> = mutableListOf()
    private var generation: Int = 0
    private var bestAccuracy: Float = 0f

    // Dataset: list of (input features, label 0 or 1)
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()

    // Callback for progress updates
    var onGenerationComplete: ((gen: Int, bestAcc: Float, avgFitness: Float, bots: List<Bot>) -> Unit)? = null

    fun setDataset(data: List<Pair<FloatArray, Int>>) {
        dataset = data
    }

    fun setMutationRate(rate: Float) {
        // Mutation rate is applied when creating new bots
    }

    fun getBots(): List<Bot> = bots.toList()

    fun getGeneration(): Int = generation

    fun getBestAccuracy(): Float = bestAccuracy

    /**
     * Initialize or reset the population with fresh neural networks.
     */
    fun initializePopulation() {
        generation = 0
        bestAccuracy = 0f
        bots = mutableListOf()
        for (i in 0 until populationSize) {
            val network = NeuralNetwork(intArrayOf(inputSize, 32, 16, 1))
            bots.add(Bot(id = i, network = network))
        }
    }

    /**
     * Restore population from saved brain data.
     */
    fun restorePopulation(savedBots: List<Bot>) {
        bots = savedBots.toMutableList()
        if (bots.isNotEmpty()) {
            generation = bots.first().let {
                // Generation is stored externally, use current best accuracy
                0
            }
        }
    }

    /**
     * Run one generation of the evolutionary loop.
     * Returns true if target accuracy (>90%) has been reached.
     */
    suspend fun runGeneration(currentMutationRate: Float): Boolean {
        if (dataset.isEmpty()) return false

        // Step 1: Evaluate all BOTs
        for (bot in bots) {
            bot.status = BotStatus.TRAINING
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, 0f, bots.toList())

        var totalFitness = 0f
        var correctPredictions = 0
        val totalSamples = dataset.size

        for (bot in bots) {
            var botCorrect = 0
            var botFitness = 0f

            for ((input, label) in dataset) {
                val prediction = bot.network.predict(input)
                if (prediction == label) {
                    botCorrect++
                    botFitness += 1f
                } else {
                    // Partial credit based on confidence
                    val output = bot.network.forward(input)[0]
                    val confidence = if (label == 1) output else 1f - output
                    botFitness += confidence * 0.5f
                }
            }

            bot.fitness = botFitness / totalSamples
            bot.accuracy = botCorrect.toFloat() / totalSamples * 100f
            bot.status = BotStatus.EVALUATED
            totalFitness += bot.fitness

            if (bot.accuracy > bestAccuracy) {
                bestAccuracy = bot.accuracy
            }
        }

        val avgFitness = totalFitness / bots.size

        // Step 2: Sort by fitness and kill 9 weakest
        bots.sortByDescending { it.fitness }
        val bestBot = bots[0]
        bestBot.status = BotStatus.BEST

        // Mark eliminated bots
        for (i in 1 until bots.size) {
            bots[i].status = BotStatus.ELIMINATED
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, avgFitness, bots.toList())

        // Step 3: Clone strongest BOT into 9 new variants with mutation
        val newBots = mutableListOf<Bot>()
        newBots.add(bestBot.copy(id = 0, status = BotStatus.READY))

        for (i in 1 until populationSize) {
            val mutatedNetwork = bestBot.network.mutate(currentMutationRate)
            newBots.add(Bot(id = i, network = mutatedNetwork, status = BotStatus.READY))
        }

        bots.clear()
        bots.addAll(newBots)
        generation++

        // Step 4: Check if target reached
        return bestAccuracy >= 90f
    }

    /**
     * Get the best performing neural network.
     */
    fun getBestNetwork(): NeuralNetwork? {
        return bots.maxByOrNull { it.fitness }?.network
    }

    /**
     * Get the best bot.
     */
    fun getBestBot(): Bot? {
        return bots.maxByOrNull { it.fitness }
    }
}
