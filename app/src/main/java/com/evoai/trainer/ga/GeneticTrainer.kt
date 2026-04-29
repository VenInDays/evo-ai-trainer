package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.math.min
import kotlin.random.Random

/**
 * V3 Dynamic Lineage Evolution Engine.
 *
 * Hardcore "Survival of the Fittest" with 10 concurrent Neural Network models.
 * No model is safe — even previous champions can be replaced.
 *
 * The V3 Loop:
 * 1. SELECTION: Teacher Bot evaluates all 10 models in parallel (Dispatchers.Default).
 * 2. THE EXECUTION: Sort by Accuracy. Keep Top 2. Purge the other 8 from memory.
 * 3. RE-POPULATION: Clone Top 2 → 8 offspring (4 from Model A, 4 from Model B).
 * 4. DYNAMIC EVOLUTION: If a child outperforms its parent, the parent is discarded.
 * 5. GAUSSIAN MUTATION: Apply random Gaussian shift to clone weights.
 * 6. STAGNATION TRIGGER: 15 generations without improvement → Hyper-Mutation (2x rate for 1 round).
 * 7. PERSISTENCE: Save Top 2 models to Room DB every generation.
 * 8. TERMINATION: Stop when accuracy reaches user-defined target.
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

    // V3: Hyper-mutation state
    private var hyperMutationActive: Boolean = false
    private var activeMutationRate: Float = 0f

    // V3: Family ID counter for naming
    private var nextFamilyId: Int = 1

    // Full dataset
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()

    // Callbacks
    var onGenerationComplete: ((gen: Int, bestAcc: Float, avgFitness: Float, bots: List<Bot>) -> Unit)? = null
    var onAutoSave: ((networks: List<NeuralNetwork>, gen: Int, acc: Float) -> Unit)? = null
    var onHyperMutation: ((gen: Int, boostedRate: Float) -> Unit)? = null

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
    fun isHyperMutationActive(): Boolean = hyperMutationActive
    fun getActiveMutationRate(): Float = activeMutationRate

    /**
     * Initialize population with 10 fresh neural networks.
     * Each gets a unique familyId.
     */
    fun initializePopulation() {
        generation = 0
        bestAccuracy = 0f
        allTimeBestAccuracy = 0f
        stagnantGenerations = 0
        hyperMutationActive = false
        activeMutationRate = 0f
        nextFamilyId = 1
        bots = mutableListOf()
        for (i in 0 until populationSize) {
            val network = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            bots.add(Bot(
                id = i,
                network = network,
                lineage = BotLineage.CLONE,
                mutationVariance = 0.05f,
                familyId = nextFamilyId++,
                generationBorn = 0
            ))
        }
    }

    /**
     * Run one generation of the V3 Evolution Loop.
     * Uses parallel coroutines (Dispatchers.Default) for concurrent evaluation.
     * Returns true if target accuracy has been reached.
     */
    suspend fun runGeneration(currentMutationRate: Float): Boolean {
        if (dataset.isEmpty()) return false

        // V3: Determine active mutation rate (may be boosted by hyper-mutation)
        val effectiveMutationRate = if (hyperMutationActive) {
            val boostedRate = currentMutationRate * 2f
            activeMutationRate = boostedRate
            hyperMutationActive = false // Reset after 1 round
            boostedRate
        } else {
            activeMutationRate = currentMutationRate
            currentMutationRate
        }

        // Step 1: SELECTION — Teacher Bot evaluates all 10 models
        val batch = TeacherBot.selectBatch(dataset, batchSize)

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

        // Apply results and record sparkline history
        for ((botId, fitness, accuracy) in results) {
            val bot = bots.find { it.id == botId } ?: continue
            bot.fitness = fitness
            bot.accuracy = accuracy
            bot.status = BotStatus.EVALUATED
            bot.recordFitness(fitness) // V3: sparkline history
        }

        val avgFitness = bots.map { it.fitness }.average().toFloat()

        // Step 2: THE EXECUTION — Sort by fitness, identify Top 2
        bots.sortByDescending { it.fitness }
        val elite1 = bots[0]
        val elite2 = bots[1]

        elite1.status = BotStatus.BEST
        elite2.status = BotStatus.BEST

        // The Purge: Mark the other 8 as eliminated
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

        // V3: Stagnation Trigger — 15 generations → Hyper-Mutation
        if (stagnantGenerations >= 15 && !hyperMutationActive) {
            hyperMutationActive = true
            onHyperMutation?.invoke(generation, effectiveMutationRate * 2f)
        }

        // Update running best accuracy
        if (currentBest > bestAccuracy) {
            bestAccuracy = currentBest
        }

        // Step 3: RE-POPULATION — Clone Top 2 to create 8 offspring
        val elite1Network = elite1.network.deepClone()
        val elite2Network = elite2.network.deepClone()
        val newBots = mutableListOf<Bot>()

        // V3: Retain Top 2 as LEGACY survivors
        newBots.add(Bot(
            id = 0,
            network = elite1Network.deepClone(),
            status = BotStatus.READY,
            lineage = BotLineage.LEGACY,
            parentRank = 1,
            familyId = elite1.familyId,     // Inherit family ID
            generationBorn = elite1.generationBorn, // Keep original birth generation
            fitnessHistory = elite1.fitnessHistory.toMutableList() // Carry sparkline
        ))
        newBots.add(Bot(
            id = 1,
            network = elite2Network.deepClone(),
            status = BotStatus.READY,
            lineage = BotLineage.LEGACY,
            parentRank = 2,
            familyId = elite2.familyId,
            generationBorn = elite2.generationBorn,
            fitnessHistory = elite2.fitnessHistory.toMutableList()
        ))

        // Generate 4 mutated clones from Model A + 4 from Model B
        val clonesPerElite = (populationSize - 2) / 2  // = 4 each
        var botIdCounter = 2

        // Clones from Legacy #1 (Model A)
        for (i in 0 until clonesPerElite) {
            val mutationVariance = effectiveMutationRate * (0.8f + Random.nextFloat() * 0.4f)
            val mutatedNetwork = elite1Network.mutateGaussian(mutationVariance)

            // Reset injection: if parent had 0.00 fitness, randomize first clone
            val lineage = if (elite1.fitness == 0f && i == 0) {
                BotLineage.RESET_RANDOM
            } else {
                BotLineage.CLONE
            }

            val network = if (lineage == BotLineage.RESET_RANDOM) {
                NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            } else {
                mutatedNetwork
            }

            val familyId = if (lineage == BotLineage.RESET_RANDOM) nextFamilyId++ else elite1.familyId

            newBots.add(Bot(
                id = botIdCounter++,
                network = network,
                status = BotStatus.READY,
                lineage = lineage,
                mutationVariance = mutationVariance,
                parentRank = 1,
                familyId = familyId,
                generationBorn = generation + 1  // V3: Born in next generation
            ))
        }

        // Clones from Legacy #2 (Model B)
        for (i in 0 until clonesPerElite) {
            val mutationVariance = effectiveMutationRate * (0.8f + Random.nextFloat() * 0.4f)
            val mutatedNetwork = elite2Network.mutateGaussian(mutationVariance)

            val lineage = if (elite2.fitness == 0f && i == 0) {
                BotLineage.RESET_RANDOM
            } else {
                BotLineage.CLONE
            }

            val network = if (lineage == BotLineage.RESET_RANDOM) {
                NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            } else {
                mutatedNetwork
            }

            val familyId = if (lineage == BotLineage.RESET_RANDOM) nextFamilyId++ else elite2.familyId

            newBots.add(Bot(
                id = botIdCounter++,
                network = network,
                status = BotStatus.READY,
                lineage = lineage,
                mutationVariance = mutationVariance,
                parentRank = 2,
                familyId = familyId,
                generationBorn = generation + 1
            ))
        }

        // If population is not full, add random models
        while (newBots.size < populationSize) {
            val randomNetwork = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
            newBots.add(Bot(
                id = botIdCounter++,
                network = randomNetwork,
                status = BotStatus.READY,
                lineage = BotLineage.RESET_RANDOM,
                familyId = nextFamilyId++,
                generationBorn = generation + 1
            ))
        }

        // Additional reset injection: if both elites have 0.00 fitness
        if (elite1.fitness == 0f && elite2.fitness == 0f) {
            for (i in (newBots.size - 2) until newBots.size) {
                val freshNetwork = NeuralNetwork(intArrayOf(inputSize, 64, 32, 16, 1))
                newBots[i] = Bot(
                    id = newBots[i].id,
                    network = freshNetwork,
                    status = BotStatus.READY,
                    lineage = BotLineage.RESET_RANDOM,
                    mutationVariance = 0f,
                    familyId = nextFamilyId++,
                    generationBorn = generation + 1
                )
            }
        }

        bots.clear()
        bots.addAll(newBots)
        generation++

        // V3: Save Top 2 models EVERY generation (persistence guarantee)
        onAutoSave?.invoke(
            listOf(elite1Network, elite2Network),
            generation,
            bestAccuracy
        )

        // Step 4: Termination check
        return bestAccuracy >= targetAccuracy
    }

    /**
     * Get the best performing neural network (Current Champion).
     */
    fun getBestNetwork(): NeuralNetwork? {
        return bots.maxByOrNull { it.fitness }?.network
    }

    fun getBestBot(): Bot? {
        return bots.maxByOrNull { it.fitness }
    }

    /**
     * Run inference on a single input using the Current Champion.
     */
    fun predict(input: FloatArray): Pair<Int, Float> {
        val best = getBestNetwork() ?: return Pair(0, 0f)
        return best.predictWithConfidence(input)
    }
}
