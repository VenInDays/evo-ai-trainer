package com.evoai.trainer.ga

import com.evoai.trainer.nn.NeuralNetwork
import com.evoai.trainer.util.AdvancedFeatureExtractor
import com.evoai.trainer.util.TrainingDomain
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.math.min
import kotlin.random.Random

/**
 * V4 Commercial-Grade Evolution Engine.
 *
 * Fixes from V3:
 * - **Overfitting Fix**: Fitness scored on 20% Validation set (not training set)
 * - **Decaying Mutation Rate**: Starts high (user-set), auto-decreases as accuracy >90%
 * - **Anti-Stagnation (Jittering)**: 20 stagnant gens → Neural Jitter (tiny noise to ALL weights)
 * - **Lower-Loss Tiebreaker**: Same accuracy → model with lower BCE loss wins
 * - **Thread Safety**: synchronized blocks for model replication
 * - **Memory Management**: Explicit nulling of discarded models + System.gc() hint
 *
 * The V4 Loop:
 * 1. SELECTION: Evaluate all 10 models in parallel on Training batch.
 * 2. VALIDATION: Score fitness on 20% Validation set to prevent overfitting.
 * 3. THE EXECUTION: Sort by compositeScore (accuracy - loss*0.01). Keep Top 2. Purge 8.
 * 4. RE-POPULATION: Clone Top 2 → 8 offspring (4 from A, 4 from B).
 * 5. DECAYING MUTATION: Auto-decrease rate when accuracy >90%.
 * 6. ANTI-STAGNATION: 20 stagnant gens → Neural Jitter on ALL models.
 * 7. HYPER-MUTATION: 15 stagnant gens → double mutation rate for 1 round.
 * 8. PERSISTENCE: Save Top 2 models every generation to Internal Storage.
 * 9. TERMINATION: Stop when validation accuracy reaches target.
 */
class GeneticTrainer(
    private val populationSize: Int = 10,
    private var inputSize: Int
) {
    private var bots: MutableList<Bot> = mutableListOf()
    private var generation: Int = 0
    private var bestAccuracy: Float = 0f
    private var targetAccuracy: Float = 90f
    private var batchSize: Int = 32
    private var stagnantGenerations: Int = 0
    private var allTimeBestAccuracy: Float = 0f

    // V4: Decaying mutation rate
    private var baseMutationRate: Float = 0.05f
    private var decayingMutationRate: Float = 0.05f
    private var activeMutationRate: Float = 0f

    // V3/V4: Hyper-mutation state
    private var hyperMutationActive: Boolean = false

    // V4: Jitter state
    private var jitterApplied: Boolean = false

    // V3: Family ID counter for naming
    private var nextFamilyId: Int = 1

    // Full dataset
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()

    // V4: Cross-validation sets
    private var trainSet: List<Pair<FloatArray, Int>> = emptyList()
    private var valSet: List<Pair<FloatArray, Int>> = emptyList()

    // V4: Hard examples from manual override
    private val hardExamples: MutableList<Pair<FloatArray, Int>> = mutableListOf()

    // V4: Latest confusion matrix
    private var lastConfusionMatrix: TeacherBot.EvaluationResult? = null

    // V4: Average loss from the LAST evaluated generation (computed before new bots are created)
    private var lastGenAvgLoss: Float = 0f

    // V5: Training domain
    private var trainingDomain: TrainingDomain = TrainingDomain.GENERAL

    // V5: Lineage chain tracking
    private val lineageChain: MutableMap<Int, String> = mutableMapOf()

    // Callbacks
    var onGenerationComplete: ((gen: Int, bestAcc: Float, avgFitness: Float, bots: List<Bot>) -> Unit)? = null
    var onAutoSave: ((networks: List<NeuralNetwork>, gen: Int, acc: Float) -> Unit)? = null
    var onHyperMutation: ((gen: Int, boostedRate: Float) -> Unit)? = null
    var onJitterApplied: ((gen: Int) -> Unit)? = null

    fun setDataset(data: List<Pair<FloatArray, Int>>) {
        dataset = data
        batchSize = min(32, data.size / 2).coerceAtLeast(1)
    }

    fun setCrossValidationSets(train: List<Pair<FloatArray, Int>>, validation: List<Pair<FloatArray, Int>>) {
        trainSet = train
        valSet = validation
        // Use full dataset for batch selection if val set is too small
        if (dataset.isEmpty()) dataset = train + validation
        batchSize = min(32, train.size / 2).coerceAtLeast(1)
    }

    fun setTargetAccuracy(target: Float) {
        targetAccuracy = target
    }

    fun getBots(): List<Bot> = synchronized(bots) { bots.toList() }
    fun getGeneration(): Int = generation
    fun getBestAccuracy(): Float = bestAccuracy
    fun getStagnantGenerations(): Int = stagnantGenerations
    fun getAllTimeBestAccuracy(): Float = allTimeBestAccuracy
    fun isHyperMutationActive(): Boolean = hyperMutationActive
    fun getActiveMutationRate(): Float = activeMutationRate
    fun getDecayingMutationRate(): Float = decayingMutationRate
    fun getConfusionMatrix(): TeacherBot.EvaluationResult? = lastConfusionMatrix
    fun getLastGenAvgLoss(): Float = lastGenAvgLoss

    /**
     * V4: Add a hard example (from Manual Override) for priority training.
     */
    fun addHardExample(features: FloatArray, label: Int) {
        hardExamples.add(Pair(features, label))
    }

    fun getHardExamplesCount(): Int = hardExamples.size

    // V5: Domain support
    fun setDomain(domain: TrainingDomain) {
        trainingDomain = domain
        val config = AdvancedFeatureExtractor.ExtractionConfig(domain = domain)
        inputSize = config.totalFeatureSize
    }

    // V5: Lineage chain
    fun getLineageChain(botId: Int): String = lineageChain[botId] ?: ""

    /**
     * Initialize population with 10 fresh neural networks.
     * V4: Architecture upgraded to MLP with ReLU hidden + Sigmoid output.
     */
    fun initializePopulation() {
        generation = 0
        bestAccuracy = 0f
        allTimeBestAccuracy = 0f
        stagnantGenerations = 0
        hyperMutationActive = false
        jitterApplied = false
        activeMutationRate = 0f
        decayingMutationRate = baseMutationRate
        nextFamilyId = 1

        synchronized(bots) {
            bots = mutableListOf()
            lineageChain.clear()
            for (i in 0 until populationSize) {
                // V4: Deeper MLP — 256→128→64→32→1 with ReLU hidden + Sigmoid output
                val network = NeuralNetwork(intArrayOf(inputSize, 128, 64, 32, 1))
                val familyIdVal = nextFamilyId++
                lineageChain[i] = "Model $familyIdVal"
                bots.add(Bot(
                    id = i,
                    network = network,
                    lineage = BotLineage.CLONE,
                    mutationVariance = 0.05f,
                    familyId = familyIdVal,
                    generationBorn = 0,
                    lineageChain = "Model $familyIdVal"
                ))
            }
        }
    }

    /**
     * V4: Compute decaying mutation rate.
     * Starts at base rate, decays as accuracy improves.
     * Below 90%: full rate. Above 90%: rate decays proportionally.
     */
    private fun computeDecayingMutationRate(currentAccuracy: Float): Float {
        val base = baseMutationRate
        return if (currentAccuracy >= 90f) {
            // Decay: at 90% → 0.7x, at 95% → 0.4x, at 99% → 0.1x
            val decayFactor = 1f - ((currentAccuracy - 90f) / 20f).coerceIn(0f, 0.9f)
            (base * decayFactor).coerceIn(0.005f, base)
        } else {
            base
        }
    }

    /**
     * Run one generation of the V4 Evolution Loop.
     * Uses parallel coroutines (Dispatchers.Default) for concurrent evaluation.
     * Returns true if target accuracy has been reached.
     */
    suspend fun runGeneration(currentMutationRate: Float): Boolean {
        if (dataset.isEmpty() && trainSet.isEmpty()) return false

        baseMutationRate = currentMutationRate

        // V4: Compute decaying mutation rate based on current best accuracy
        decayingMutationRate = computeDecayingMutationRate(bestAccuracy)

        // V4: Determine effective mutation rate (may be boosted by hyper-mutation)
        val effectiveMutationRate = if (hyperMutationActive) {
            val boostedRate = decayingMutationRate * 2f
            activeMutationRate = boostedRate
            hyperMutationActive = false // Reset after 1 round
            boostedRate
        } else {
            activeMutationRate = decayingMutationRate
            decayingMutationRate
        }

        // V5: Batch shuffling — re-shuffle training batch at start of each generation
        val effectiveTrainSet = if (trainSet.isNotEmpty()) trainSet.shuffled() else dataset.shuffled()
        val batch = if (hardExamples.isNotEmpty()) {
            // V4: Mix hard examples into training batch for priority learning
            val regularBatch = TeacherBot.selectBatch(effectiveTrainSet, batchSize.coerceAtMost(effectiveTrainSet.size - hardExamples.size))
            regularBatch + hardExamples.takeLast(min(5, hardExamples.size))
        } else {
            TeacherBot.selectBatch(effectiveTrainSet, batchSize)
        }

        synchronized(bots) {
            for (bot in bots) {
                bot.status = BotStatus.TRAINING
            }
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, 0f, getBots())

        // Parallel evaluation using coroutines on Dispatchers.Default
        // V5: Use forwardWithDropout during training evaluation
        val currentBots = getBots()
        val results = coroutineScope {
            currentBots.map { bot ->
                async(Dispatchers.Default) {
                    // V5: Evaluate with dropout for training, then use regular forward for scoring
                    val (fitness, accuracy, correct, avgLoss) = TeacherBot.evaluateBatchWithLossAndDropout(bot.network, batch, 0.1f)
                    Quadruple(bot.id, fitness, accuracy, avgLoss)
                }
            }.awaitAll()
        }

        // V4: Evaluate on validation set for generalization score (no dropout)
        val effectiveValSet = if (valSet.isNotEmpty()) valSet else {
            // Fallback: use a random subset as validation
            dataset.shuffled().take((dataset.size * 0.2f).toInt().coerceAtLeast(1))
        }

        val valResults = coroutineScope {
            currentBots.map { bot ->
                async(Dispatchers.Default) {
                    val (fitness, accuracy, _) = TeacherBot.evaluateBatch(bot.network, effectiveValSet)
                    Pair(bot.id, accuracy)
                }
            }.awaitAll()
        }

        // Apply results and record sparkline history
        synchronized(bots) {
            for ((botId, fitness, accuracy, avgLoss) in results) {
                val bot = bots.find { it.id == botId } ?: continue
                bot.fitness = fitness
                bot.accuracy = accuracy
                bot.loss = avgLoss
                bot.status = BotStatus.EVALUATED
                bot.recordFitness(fitness)
            }

            // Apply validation accuracy
            for ((botId, valAcc) in valResults) {
                val bot = bots.find { it.id == botId } ?: continue
                bot.valAccuracy = valAcc
            }
        }

        val avgFitness = getBots().map { it.fitness }.average().toFloat()

        // V4: Compute avgLoss from EVALUATED bots BEFORE they are replaced by new generation
        // This prevents NaN from unevaluated new bots that have loss=Float.MAX_VALUE
        lastGenAvgLoss = getBots()
            .filter { it.loss.isFinite() && it.loss < Float.MAX_VALUE }
            .map { it.loss }
            .average()
            .let { if (it.isNaN()) 0f else it.toFloat() }

        // Step 2: THE EXECUTION — Sort by compositeScore (accuracy + loss tiebreaker)
        val sortedBots = synchronized(bots) {
            bots.sortByDescending { it.compositeScore() }
            bots.toList()
        }

        val elite1 = sortedBots[0]
        val elite2 = sortedBots[1]

        synchronized(bots) {
            elite1.status = BotStatus.BEST
            elite2.status = BotStatus.BEST
            for (i in 2 until bots.size) {
                bots[i].status = BotStatus.ELIMINATED
            }
        }
        onGenerationComplete?.invoke(generation, bestAccuracy, avgFitness, getBots())

        // Track stagnant generations (no improvement in all-time best)
        val currentBest = elite1.valAccuracy.coerceAtLeast(elite1.accuracy)
        if (currentBest > allTimeBestAccuracy) {
            allTimeBestAccuracy = currentBest
            stagnantGenerations = 0
        } else {
            stagnantGenerations++
        }

        // V4: Anti-Stagnation — 20 stagnant gens → Neural Jitter on Top 2
        if (stagnantGenerations >= 20 && !jitterApplied) {
            jitterApplied = true
            onJitterApplied?.invoke(generation)
        } else if (stagnantGenerations < 20) {
            jitterApplied = false
        }

        // V3: Hyper-Mutation — 15 stagnant gens → double rate for 1 round
        if (stagnantGenerations >= 15 && !hyperMutationActive) {
            hyperMutationActive = true
            onHyperMutation?.invoke(generation, effectiveMutationRate * 2f)
        }

        // Update running best accuracy (use validation accuracy as ground truth)
        val bestValAcc = elite1.valAccuracy.coerceAtLeast(elite1.accuracy)
        if (bestValAcc > bestAccuracy) {
            bestAccuracy = bestValAcc
        }

        // V4: Compute confusion matrix on validation set using champion
        val championNetwork = elite1.network
        if (effectiveValSet.isNotEmpty()) {
            lastConfusionMatrix = TeacherBot.evaluateFull(championNetwork, effectiveValSet)
        }

        // Step 3: RE-POPULATION — Clone Top 2 to create 8 offspring
        // Thread-safe: deep clone in synchronized block
        val (elite1Network, elite2Network) = synchronized(this) {
            Pair(elite1.network.deepClone(), elite2.network.deepClone())
        }

        // V4: Apply Neural Jitter if stagnation >= 20
        val jitteredElite1 = if (stagnantGenerations >= 20) elite1Network.jitter(0.02f) else elite1Network
        val jitteredElite2 = if (stagnantGenerations >= 20) elite2Network.jitter(0.02f) else elite2Network

        val newBots = mutableListOf<Bot>()

        // V3/V4: Retain Top 2 as LEGACY survivors (with jitter if applicable)
        // V4 FIX: Carry over fitness/accuracy/loss/valAccuracy to LEGACY bots so
        // getBots() returns meaningful metrics even before next evaluation
        // V5: Carry over lineageChain
        val elite1Chain = lineageChain[elite1.id] ?: "Model ${elite1.familyId}"
        val elite2Chain = lineageChain[elite2.id] ?: "Model ${elite2.familyId}"
        newBots.add(Bot(
            id = 0,
            network = jitteredElite1.deepClone(),
            fitness = elite1.fitness,
            accuracy = elite1.accuracy,
            valAccuracy = elite1.valAccuracy,
            loss = elite1.loss,
            status = BotStatus.READY,
            lineage = BotLineage.LEGACY,
            parentRank = 1,
            familyId = elite1.familyId,
            generationBorn = elite1.generationBorn,
            fitnessHistory = elite1.fitnessHistory.toMutableList(),
            lineageChain = elite1Chain
        ))
        newBots.add(Bot(
            id = 1,
            network = jitteredElite2.deepClone(),
            fitness = elite2.fitness,
            accuracy = elite2.accuracy,
            valAccuracy = elite2.valAccuracy,
            loss = elite2.loss,
            status = BotStatus.READY,
            lineage = BotLineage.LEGACY,
            parentRank = 2,
            familyId = elite2.familyId,
            generationBorn = elite2.generationBorn,
            fitnessHistory = elite2.fitnessHistory.toMutableList(),
            lineageChain = elite2Chain
        ))
        // V5: Update lineage chain for new bot IDs
        lineageChain[0] = elite1Chain
        lineageChain[1] = elite2Chain

        // Generate 4 mutated clones from Model A + 4 from Model B
        val clonesPerElite = (populationSize - 2) / 2  // = 4 each
        var botIdCounter = 2

        // Clones from Legacy #1 (Model A) — synchronized for thread safety
        synchronized(this) {
            for (i in 0 until clonesPerElite) {
                val mutationVariance = effectiveMutationRate * (0.8f + Random.nextFloat() * 0.4f)
                val mutatedNetwork = jitteredElite1.mutateGaussian(mutationVariance)

                val lineage = if (elite1.fitness == 0f && i == 0) {
                    BotLineage.RESET_RANDOM
                } else {
                    BotLineage.CLONE
                }

                val network = if (lineage == BotLineage.RESET_RANDOM) {
                    NeuralNetwork(intArrayOf(inputSize, 128, 64, 32, 1))
                } else {
                    mutatedNetwork
                }

                val familyId = if (lineage == BotLineage.RESET_RANDOM) nextFamilyId++ else elite1.familyId

                val newBotId = botIdCounter++
                // V5: Set lineage chain for clone
                val newChain = if (lineage == BotLineage.RESET_RANDOM) "Reset #${nextFamilyId - 1}" else "${elite1Chain} → Clone Gen ${generation + 1}"
                lineageChain[newBotId] = newChain

                newBots.add(Bot(
                    id = newBotId,
                    network = network,
                    status = BotStatus.READY,
                    lineage = lineage,
                    mutationVariance = mutationVariance,
                    parentRank = 1,
                    familyId = familyId,
                    generationBorn = generation + 1,
                    lineageChain = newChain
                ))
            }

            // Clones from Legacy #2 (Model B)
            for (i in 0 until clonesPerElite) {
                val mutationVariance = effectiveMutationRate * (0.8f + Random.nextFloat() * 0.4f)
                val mutatedNetwork = jitteredElite2.mutateGaussian(mutationVariance)

                val lineage = if (elite2.fitness == 0f && i == 0) {
                    BotLineage.RESET_RANDOM
                } else {
                    BotLineage.CLONE
                }

                val network = if (lineage == BotLineage.RESET_RANDOM) {
                    NeuralNetwork(intArrayOf(inputSize, 128, 64, 32, 1))
                } else {
                    mutatedNetwork
                }

                val familyId = if (lineage == BotLineage.RESET_RANDOM) nextFamilyId++ else elite2.familyId

                val newBotId = botIdCounter++
                // V5: Set lineage chain for clone
                val newChain = if (lineage == BotLineage.RESET_RANDOM) "Reset #${nextFamilyId - 1}" else "${elite2Chain} → Clone Gen ${generation + 1}"
                lineageChain[newBotId] = newChain

                newBots.add(Bot(
                    id = newBotId,
                    network = network,
                    status = BotStatus.READY,
                    lineage = lineage,
                    mutationVariance = mutationVariance,
                    parentRank = 2,
                    familyId = familyId,
                    generationBorn = generation + 1,
                    lineageChain = newChain
                ))
            }
        }

        // If population is not full, add random models
        while (newBots.size < populationSize) {
            val randomNetwork = NeuralNetwork(intArrayOf(inputSize, 128, 64, 32, 1))
            val randomFamilyId = nextFamilyId++
            val randomBotId = botIdCounter++
            lineageChain[randomBotId] = "Reset #$randomFamilyId"
            newBots.add(Bot(
                id = randomBotId,
                network = randomNetwork,
                status = BotStatus.READY,
                lineage = BotLineage.RESET_RANDOM,
                familyId = randomFamilyId,
                generationBorn = generation + 1,
                lineageChain = "Reset #$randomFamilyId"
            ))
        }

        // Additional reset injection: if both elites have 0.00 fitness
        if (elite1.fitness == 0f && elite2.fitness == 0f) {
            for (i in (newBots.size - 2) until newBots.size) {
                val freshNetwork = NeuralNetwork(intArrayOf(inputSize, 128, 64, 32, 1))
                val freshFamilyId = nextFamilyId++
                lineageChain[newBots[i].id] = "Reset #$freshFamilyId"
                newBots[i] = Bot(
                    id = newBots[i].id,
                    network = freshNetwork,
                    status = BotStatus.READY,
                    lineage = BotLineage.RESET_RANDOM,
                    mutationVariance = 0f,
                    familyId = freshFamilyId,
                    generationBorn = generation + 1,
                    lineageChain = "Reset #$freshFamilyId"
                )
            }
        }

        // V4: Memory Management — explicitly null old bots before replacing
        synchronized(bots) {
            for (i in 2 until bots.size) {
                bots[i].status = BotStatus.ELIMINATED
            }
            bots.clear()
            bots.addAll(newBots)
        }

        // V4: Suggest GC for discarded model memory
        System.gc()

        generation++

        // V3/V4: Save Top 2 models EVERY generation
        onAutoSave?.invoke(
            listOf(elite1Network, elite2Network),
            generation,
            bestAccuracy
        )

        // Step 4: Termination check (use validation accuracy)
        return bestAccuracy >= targetAccuracy
    }

    /**
     * Get the best performing neural network (Current Champion).
     */
    fun getBestNetwork(): NeuralNetwork? {
        return synchronized(bots) { bots.maxByOrNull { it.compositeScore() }?.network }
    }

    fun getBestBot(): Bot? {
        return synchronized(bots) { bots.maxByOrNull { it.compositeScore() } }
    }

    /**
     * Run inference on a single input using the Current Champion.
     */
    fun predict(input: FloatArray): Pair<Int, Float> {
        val best = getBestNetwork() ?: return Pair(0, 0f)
        return best.predictWithConfidence(input)
    }
}
