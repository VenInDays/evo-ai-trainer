package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.math.min
import kotlin.random.Random

/**
 * Multi-Agent Evolutionary Training System v2.0.
 *
 * Implements "Survival of the Fittest" with 10 concurrent Neural Network models.
 *
 * Enhanced Evolution Loop:
 * 1. Execute: All 10 Models process the same training batch in parallel (Dispatchers.Default).
 * 2. Evaluate: Teacher Bot compares outputs vs. ground truth for Fitness Score.
 * 3. Natural Selection: Identify Top 2 Elite models, terminate the other 8.
 * 4. Replication: Deep-clone Top 2 Elites; distribute 4 clones from Elite #1, 4 from Elite #2.
 * 5. Mutation: Apply random variance to 8 clones to ensure diversity and evolution.
 * 6. Reset Injection: If any model hits 0.00 fitness, inject a fresh random network.
 * 7. Stagnation Tracking: Count generations without improvement in best fitness.
 * 8. Auto-Save: Persist best model every 5 generations.
 * 9. Termination: Stop when global accuracy reaches user-defined target.
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
    private var stagnantGenerations: Int = 0
    private var allTimeBestAccuracy: Float = 0f

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
    fun getStagnantGenerations(): Int = stagnantGenerations
    fun getAllTimeBestAccuracy(): Float = allTimeBestAccuracy

    /**
     * Initialize population with 10 fresh neural networks.
     * All start as MUTATED_CLONE lineage (first generation has no elites).
     */
    fun initializePopulation() {
        generation = 0
        bestAccuracy = 0f
        allTimeBestAccuracy = 0f
        stagnantGenerations = 0
        bots = mutableListOf()
        for (i in 0 until populationSize) {
            val network = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            bots.add(Bot(
                id = i,
                network = network,
                lineage = BotLineage.MUTATED_CLONE,
                mutationVariance = 0.05f
            ))
        }
    }

    /**
     * Run one generation of the evolutionary loop.
     * Uses parallel coroutines (Dispatchers.Default) for concurrent evaluation.
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

        // Parallel evaluation using coroutines on Dispatchers.Default
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
        }

        val avgFitness = bots.map { it.fitness }.average().toFloat()

        // Step 3: Natural Selection - Sort by fitness
        bots.sortByDescending { it.fitness }
        val elite1 = bots[0]
        val elite2 = bots[1]

        elite1.status = BotStatus.BEST
        elite2.status = BotStatus.BEST

        // Mark the other 8 as eliminated
        for (i in 2 until bots.size) {
            bots[i].status = BotStatus.ELIMINATED
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, avgFitness, bots.toList())

        // Track stagnant generations (no improvement in all-time best)
        val currentBest = elite1.accuracy
        if (currentBest > allTimeBestAccuracy) {
            allTimeBestAccuracy = currentBest
            stagnantGenerations = 0
        } else {
            stagnantGenerations++
        }

        // Update running best accuracy
        if (currentBest > bestAccuracy) {
            bestAccuracy = currentBest
        }

        // Step 4 & 5: Replication + Mutation with Top-2 Elite Strategy
        val elite1Network = elite1.network.deepClone()
        val elite2Network = elite2.network.deepClone()
        val newBots = mutableListOf<Bot>()

        // Retain Top 2 Elites unmodified (Elite Parents)
        newBots.add(Bot(
            id = 0,
            network = elite1Network.deepClone(),
            status = BotStatus.READY,
            lineage = BotLineage.ELITE_PARENT,
            parentRank = 1
        ))
        newBots.add(Bot(
            id = 1,
            network = elite2Network.deepClone(),
            status = BotStatus.READY,
            lineage = BotLineage.ELITE_PARENT,
            parentRank = 2
        ))

        // Generate 4 mutated clones from Elite #1 and 4 from Elite #2
        val clonesPerElite = (populationSize - 2) / 2  // = 4 each
        var botIdCounter = 2

        // Clones from Elite #1
        for (i in 0 until clonesPerElite) {
            val mutationVariance = currentMutationRate * (1f + Random.nextFloat() * 0.5f)
            val mutatedNetwork = elite1Network.mutate(mutationVariance)

            // Reset injection: if the clone gets 0.00 fitness potential, randomize
            val lineage = if (elite1.fitness == 0f && i == 0) {
                BotLineage.RESET_RANDOM
            } else {
                BotLineage.MUTATED_CLONE
            }

            val network = if (lineage == BotLineage.RESET_RANDOM) {
                NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            } else {
                mutatedNetwork
            }

            newBots.add(Bot(
                id = botIdCounter++,
                network = network,
                status = BotStatus.READY,
                lineage = lineage,
                mutationVariance = mutationVariance,
                parentRank = 1
            ))
        }

        // Clones from Elite #2
        for (i in 0 until clonesPerElite) {
            val mutationVariance = currentMutationRate * (1f + Random.nextFloat() * 0.5f)
            val mutatedNetwork = elite2Network.mutate(mutationVariance)

            val lineage = if (elite2.fitness == 0f && i == 0) {
                BotLineage.RESET_RANDOM
            } else {
                BotLineage.MUTATED_CLONE
            }

            val network = if (lineage == BotLineage.RESET_RANDOM) {
                NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            } else {
                mutatedNetwork
            }

            newBots.add(Bot(
                id = botIdCounter++,
                network = network,
                status = BotStatus.READY,
                lineage = lineage,
                mutationVariance = mutationVariance,
                parentRank = 2
            ))
        }

        // If population is not full (edge case), add random models
        while (newBots.size < populationSize) {
            val randomNetwork = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            newBots.add(Bot(
                id = botIdCounter++,
                network = randomNetwork,
                status = BotStatus.READY,
                lineage = BotLineage.RESET_RANDOM
            ))
        }

        // Additional reset injection: if ALL models have 0.00 fitness, inject 2 fresh random networks
        val allZero = newBots.all { it.lineage != BotLineage.RESET_RANDOM }
        val anyZeroFitness = elite1.fitness == 0f && elite2.fitness == 0f
        if (anyZeroFitness && allZero) {
            // Replace the last 2 clones with fresh random networks
            for (i in (newBots.size - 2) until newBots.size) {
                val freshNetwork = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
                newBots[i] = Bot(
                    id = newBots[i].id,
                    network = freshNetwork,
                    status = BotStatus.READY,
                    lineage = BotLineage.RESET_RANDOM,
                    mutationVariance = 0f
                )
            }
        }

        bots.clear()
        bots.addAll(newBots)
        generation++

        // Auto-save trigger every 5 generations
        if (generation % 5 == 0) {
            onAutoSave?.invoke(elite1Network, generation, bestAccuracy)
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
