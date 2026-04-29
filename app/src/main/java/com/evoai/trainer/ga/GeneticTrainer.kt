package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.random.Random

/**
 * Multi-Agent Evolutionary Training System.
 *
 * Implements "Survival of the Fittest" with 10 concurrent Neural Network models.
 *
 * The Loop:
 * 1. Execute: All 10 Models process the same training batch.
 * 2. Evaluate: Teacher Bot compares outputs vs. ground truth for Fitness Score.
 * 3. Natural Selection: Identify #1 Model (highest accuracy), terminate the other 9.
 * 4. Replication: Deep-clone #1 Model into 10 new instances.
 * 5. Mutation: Apply random variance to 9 clones to ensure evolution.
 * 6. Termination: Stop when global accuracy reaches user-defined target.
 */
class GeneticTrainer(
    private val populationSize: Int = 10,
    private val inputSize: Int
) {
    private var bots: MutableList<Bot> = mutableListOf()
    private var generation: Int = 0
    private var bestAccuracy: Float = 0f
    private var targetAccuracy: Float = 90f
    private var batchSize: Int = 32

    // Full dataset
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()

    // Callbacks
    var onGenerationComplete: ((gen: Int, bestAcc: Float, avgFitness: Float, bots: List<Bot>) -> Unit)? = null
    var onAutoSave: ((bestNetwork: NeuralNetwork, gen: Int, acc: Float) -> Unit)? = null

    fun setDataset(data: List<Pair<FloatArray, Int>>) {
        dataset = data
        batchSize = min(32, data.size / 2).coerceAtLeast(1)
    }

    fun setTargetAccuracy(target: Float) {
        targetAccuracy = target
    }

    fun getBots(): List<Bot> = bots.toList()
    fun getGeneration(): Int = generation
    fun getBestAccuracy(): Float = bestAccuracy

    /**
     * Initialize population with 10 fresh neural networks.
     */
    fun initializePopulation() {
        generation = 0
        bestAccuracy = 0f
        bots = mutableListOf()
        for (i in 0 until populationSize) {
            val network = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            bots.add(Bot(id = i, network = network))
        }
    }

    /**
     * Run one generation of the evolutionary loop.
     * Uses parallel coroutines for concurrent evaluation.
     * Returns true if target accuracy has been reached.
     */
    suspend fun runGeneration(currentMutationRate: Float): Boolean {
        if (dataset.isEmpty()) return false

        // Step 1: Teacher Bot selects a random training batch
        val batch = TeacherBot.selectBatch(dataset, batchSize)

        // Step 2: Execute - All 10 Models process the batch in parallel
        for (bot in bots) {
            bot.status = BotStatus.TRAINING
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, 0f, bots.toList())

        // Parallel evaluation using coroutines
        val results = coroutineScope {
            bots.map { bot ->
                async(Dispatchers.Default) {
                    val (fitness, accuracy, correct) = TeacherBot.evaluateBatch(bot.network, batch)
                    Triple(bot.id, fitness, accuracy)
                }
            }.awaitAll()
        }

        // Apply results
        for ((botId, fitness, accuracy) in results) {
            val bot = bots.find { it.id == botId } ?: continue
            bot.fitness = fitness
            bot.accuracy = accuracy
            bot.status = BotStatus.EVALUATED

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy
            }
        }

        val avgFitness = bots.map { it.fitness }.average().toFloat()

        // Step 3: Natural Selection - Sort by fitness, identify #1 Model
        bots.sortByDescending { it.fitness }
        val bestBot = bots[0]
        bestBot.status = BotStatus.BEST

        // Mark the other 9 as eliminated
        for (i in 1 until bots.size) {
            bots[i].status = BotStatus.ELIMINATED
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, avgFitness, bots.toList())

        // Step 4: Replication - Deep-clone #1 Model into 10 instances
        val bestNetwork = bestBot.network.deepClone()
        val newBots = mutableListOf<Bot>()

        // Keep the champion unmodified
        newBots.add(Bot(id = 0, network = bestNetwork.deepClone(), status = BotStatus.READY))

        // Step 5: Mutation - Apply random variance to the 9 clones
        for (i in 1 until populationSize) {
            val mutatedNetwork = bestNetwork.mutate(currentMutationRate)
            newBots.add(Bot(id = i, network = mutatedNetwork, status = BotStatus.READY))
        }

        bots.clear()
        bots.addAll(newBots)
        generation++

        // Auto-save trigger every 5 generations
        if (generation % 5 == 0) {
            onAutoSave?.invoke(bestNetwork, generation, bestAccuracy)
        }

        // Step 6: Termination check
        return bestAccuracy >= targetAccuracy
    }

    /**
     * Get the best performing neural network.
     */
    fun getBestNetwork(): NeuralNetwork? {
        return bots.maxByOrNull { it.fitness }?.network
    }

    fun getBestBot(): Bot? {
        return bots.maxByOrNull { it.fitness }
    }

    /**
     * Run inference on a single input using the best model.
     */
    fun predict(input: FloatArray): Pair<Int, Float> {
        val best = getBestNetwork() ?: return Pair(0, 0f)
        return best.predictWithConfidence(input)
    }
}
