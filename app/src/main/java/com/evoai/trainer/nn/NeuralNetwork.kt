package com.evoai.trainer.nn

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * V4 Commercial-Grade Multi-Layer Perceptron (MLP).
 *
 * Architecture upgrades from V3:
 * - ReLU activation for hidden layers (captures non-linear patterns)
 * - Sigmoid activation only for the output layer (probability 0.0–1.0)
 * - He initialization for ReLU layers, Xavier for sigmoid output layer
 * - Deep copy with thread safety
 * - Gaussian jitter for anti-stagnation
 * - Binary Cross-Entropy loss computation
 */
class NeuralNetwork(
    val layerSizes: IntArray,
    var weights: MutableList<Array<FloatArray>> = mutableListOf(),
    var biases: MutableList<FloatArray> = mutableListOf(),
    val activations: MutableList<Activation> = mutableListOf()  // per-layer activation
) {
    private val rng = Random.Default

    enum class Activation { RELU, SIGMOID }

    init {
        if (weights.isEmpty() || biases.isEmpty()) {
            initializeWeights()
        }
        if (activations.isEmpty()) {
            inferActivations()
        }
    }

    /**
     * Infer activation functions: ReLU for all hidden layers, Sigmoid for output.
     */
    private fun inferActivations() {
        activations.clear()
        for (i in weights.indices) {
            if (i == weights.size - 1) {
                activations.add(Activation.SIGMOID)
            } else {
                activations.add(Activation.RELU)
            }
        }
    }

    /**
     * He initialization for ReLU layers, Xavier for sigmoid output layer.
     */
    private fun initializeWeights() {
        weights.clear()
        biases.clear()
        activations.clear()

        for (i in 0 until layerSizes.size - 1) {
            val inputSize = layerSizes[i]
            val outputSize = layerSizes[i + 1]
            val w = Array(outputSize) { FloatArray(inputSize) }
            val b = FloatArray(outputSize)

            val isOutputLayer = (i == layerSizes.size - 2)

            if (isOutputLayer) {
                // Xavier initialization for sigmoid output
                val scale = sqrt(2.0 / (inputSize + outputSize)).toFloat()
                for (j in 0 until outputSize) {
                    for (k in 0 until inputSize) {
                        w[j][k] = (rng.nextFloat() * 2 - 1) * scale
                    }
                    b[j] = 0f
                }
                activations.add(Activation.SIGMOID)
            } else {
                // He initialization for ReLU hidden layers
                val scale = sqrt(2.0 / inputSize).toFloat()
                for (j in 0 until outputSize) {
                    for (k in 0 until inputSize) {
                        w[j][k] = (rng.nextFloat() * 2 - 1) * scale
                    }
                    b[j] = 0f
                }
                activations.add(Activation.RELU)
            }

            weights.add(w)
            biases.add(b)
        }
    }

    /**
     * Forward pass through the network.
     * Uses ReLU for hidden layers and Sigmoid for the output layer.
     * Returns the output array (last layer activations — probabilities 0.0 to 1.0).
     */
    fun forward(input: FloatArray): FloatArray {
        var activation = input
        for (layerIdx in weights.indices) {
            val w = weights[layerIdx]
            val b = biases[layerIdx]
            val act = activations[layerIdx]
            val output = FloatArray(w.size)

            for (j in w.indices) {
                var sum = b[j]
                for (k in activation.indices) {
                    sum += w[j][k] * activation[k]
                }
                output[j] = when (act) {
                    Activation.RELU -> relu(sum)
                    Activation.SIGMOID -> sigmoid(sum)
                }
            }
            activation = output
        }
        return activation
    }

    /**
     * Predict: returns 1 if output > 0.5 (like), 0 otherwise (non-like).
     */
    fun predict(input: FloatArray): Int {
        val output = forward(input)
        return if (output[0] > 0.5f) 1 else 0
    }

    /**
     * Predict with confidence score (0.0 to 1.0).
     * The confidence is the probability of the predicted class.
     */
    fun predictWithConfidence(input: FloatArray): Pair<Int, Float> {
        val output = forward(input)
        val confidence = if (output[0] > 0.5f) output[0] else 1f - output[0]
        val label = if (output[0] > 0.5f) 1 else 0
        return Pair(label, confidence)
    }

    /**
     * Compute Binary Cross-Entropy loss for a single sample.
     * Lower loss = more confident correct predictions.
     */
    fun computeLoss(input: FloatArray, label: Int): Float {
        val output = forward(input)[0]
        val p = output.coerceIn(1e-7f, 1f - 1e-7f)
        val target = label.toFloat()
        return -(target * kotlin.math.ln(p.toDouble()).toFloat() +
                (1f - target) * kotlin.math.ln((1f - p).toDouble()).toFloat())
    }

    /**
     * Compute average Binary Cross-Entropy loss over a dataset batch.
     */
    fun computeAverageLoss(batch: List<Pair<FloatArray, Int>>): Float {
        if (batch.isEmpty()) return 0f
        var totalLoss = 0f
        for ((input, label) in batch) {
            totalLoss += computeLoss(input, label)
        }
        return totalLoss / batch.size
    }

    /**
     * Deep-clone this network (exact copy, no mutation).
     * Thread-safe: creates completely independent weight arrays.
     */
    fun deepClone(): NeuralNetwork {
        val newWeights = mutableListOf<Array<FloatArray>>()
        val newBiases = mutableListOf<FloatArray>()

        for (layerIdx in weights.indices) {
            val w = weights[layerIdx]
            val b = biases[layerIdx]
            val newW = Array(w.size) { j -> w[j].copyOf() }
            val newB = b.copyOf()
            newWeights.add(newW)
            newBiases.add(newB)
        }

        return NeuralNetwork(
            layerSizes.copyOf(),
            newWeights,
            newBiases,
            activations.toMutableList()
        )
    }

    /**
     * Create a mutated clone using Gaussian (normal distribution) perturbation.
     * This produces more natural evolutionary drift compared to uniform random mutation.
     */
    fun mutateGaussian(mutationRate: Float): NeuralNetwork {
        val clone = deepClone()

        for (layerIdx in clone.weights.indices) {
            val w = clone.weights[layerIdx]
            val b = clone.biases[layerIdx]

            for (j in w.indices) {
                for (k in w[j].indices) {
                    if (rng.nextFloat() < mutationRate) {
                        w[j][k] += gaussianRandom() * mutationRate
                    }
                }
                if (rng.nextFloat() < mutationRate) {
                    b[j] += gaussianRandom() * mutationRate
                }
            }
        }

        return clone
    }

    /**
     * V4: Neural Jitter — add tiny random noise to ALL weights.
     * Used as anti-stagnation mechanism when stuck for 20+ generations.
     * Unlike mutation, jitter affects every weight with very small magnitude.
     */
    fun jitter(noiseScale: Float = 0.02f): NeuralNetwork {
        val clone = deepClone()

        for (layerIdx in clone.weights.indices) {
            val w = clone.weights[layerIdx]
            val b = clone.biases[layerIdx]

            for (j in w.indices) {
                for (k in w[j].indices) {
                    w[j][k] += gaussianRandom() * noiseScale
                }
                b[j] += gaussianRandom() * noiseScale
            }
        }

        return clone
    }

    /**
     * Box-Muller transform to generate Gaussian (normal) random numbers.
     * Mean = 0, Standard Deviation = 1.
     */
    private fun gaussianRandom(): Float {
        val u1 = rng.nextFloat().coerceIn(1e-10f, 1f)
        val u2 = rng.nextFloat()
        return (sqrt(-2f * kotlin.math.ln(u1.toDouble())).toFloat() *
                kotlin.math.cos(2f * Math.PI * u2).toFloat())
    }

    /**
     * ReLU activation function with leaky variant for negative inputs.
     * Uses a small leak (0.01) to prevent dead neurons.
     */
    private fun relu(x: Float): Float = max(0.01f * x, x)  // Leaky ReLU

    companion object {
        fun sigmoid(x: Float): Float = (1f / (1f + exp(-x.toDouble()))).toFloat()

        /**
         * Deserialize from flat string (Room DB).
         */
        fun deserialize(data: String): NeuralNetwork {
            val parts = data.split("|")
            val layerSizes = parts[0].split(",").map { it.toInt() }.toIntArray()
            val layerData = parts[1].split(";").filter { it.isNotEmpty() }

            val weights = mutableListOf<Array<FloatArray>>()
            val biases = mutableListOf<FloatArray>()
            val activations = mutableListOf<Activation>()

            for ((layerIdx, ld) in layerData.withIndex()) {
                val values = ld.split(",").filter { it.isNotEmpty() }.map { it.toFloat() }
                val inputSize = layerSizes[layerIdx]
                val outputSize = layerSizes[layerIdx + 1]

                val w = Array(outputSize) { FloatArray(inputSize) }
                val b = FloatArray(outputSize)

                var idx = 0
                for (j in 0 until outputSize) {
                    for (k in 0 until inputSize) {
                        w[j][k] = values[idx++]
                    }
                }
                for (j in 0 until outputSize) {
                    b[j] = values[idx++]
                }

                weights.add(w)
                biases.add(b)

                // Infer activation: ReLU for hidden, Sigmoid for output
                if (layerIdx == layerData.size - 1) {
                    activations.add(Activation.SIGMOID)
                } else {
                    activations.add(Activation.RELU)
                }
            }

            return NeuralNetwork(layerSizes, weights, biases, activations)
        }

        /**
         * Import model from JSON (.model file).
         */
        fun fromJson(json: String): NeuralNetwork {
            val gson = Gson()
            val modelData = gson.fromJson(json, ModelData::class.java)

            val layerSizes = modelData.layerSizes.toIntArray()
            val weights = mutableListOf<Array<FloatArray>>()
            val biases = mutableListOf<FloatArray>()
            val activations = mutableListOf<Activation>()

            for ((i, layer) in modelData.layers.withIndex()) {
                val w = Array(layer.weights.size) { j -> layer.weights[j].toFloatArray() }
                val b = layer.biases.toFloatArray()
                weights.add(w)
                biases.add(b)

                // Infer activation from stored data or default
                val actName = layer.activation ?: if (i == modelData.layers.size - 1) "SIGMOID" else "RELU"
                activations.add(Activation.valueOf(actName))
            }

            return NeuralNetwork(layerSizes, weights, biases, activations)
        }
    }

    /**
     * Serialize to flat string for Room DB storage.
     */
    fun serialize(): String {
        val sb = StringBuilder()
        sb.append(layerSizes.joinToString(",")).append("|")
        for (layerIdx in weights.indices) {
            val w = weights[layerIdx]
            val b = biases[layerIdx]
            for (j in w.indices) {
                for (k in w[j].indices) {
                    sb.append(w[j][k].toString()).append(",")
                }
            }
            for (j in b.indices) {
                sb.append(b[j].toString()).append(",")
            }
            sb.append(";")
        }
        return sb.toString()
    }

    /**
     * Export model weights as JSON (for .model file — inference only, production-ready).
     */
    fun toJson(): String {
        val gson = Gson()
        val layers = mutableListOf<LayerData>()
        for (i in weights.indices) {
            val weightRows = weights[i].map { row -> row.toList() }
            layers.add(LayerData(
                weights = weightRows,
                biases = biases[i].toList(),
                activation = activations[i].name
            ))
        }
        val modelData = ModelData(
            layerSizes = layerSizes.toList(),
            layers = layers,
            version = "4.0.0"
        )
        return gson.toJson(modelData)
    }
}

// JSON serialization helpers
data class ModelData(
    @SerializedName("layerSizes") val layerSizes: List<Int>,
    @SerializedName("layers") val layers: List<LayerData>,
    @SerializedName("version") val version: String = "4.0.0"
)

data class LayerData(
    @SerializedName("weights") val weights: List<List<Float>>,
    @SerializedName("biases") val biases: List<Float>,
    @SerializedName("activation") val activation: String? = null
)

/**
 * Full checkpoint data for .ckpt export (includes training state).
 */
data class CheckpointData(
    @SerializedName("model") val model: ModelData,
    @SerializedName("generation") val generation: Int,
    @SerializedName("bestAccuracy") val bestAccuracy: Float,
    @SerializedName("avgFitness") val avgFitness: Float,
    @SerializedName("avgLoss") val avgLoss: Float = 0f,
    @SerializedName("mutationRate") val mutationRate: Float,
    @SerializedName("decayingMutationRate") val decayingMutationRate: Float = 0f,
    @SerializedName("fitnessHistory") val fitnessHistory: List<HistoryEntry>,
    @SerializedName("version") val version: String = "4.0.0"
)

data class HistoryEntry(
    @SerializedName("generation") val generation: Int,
    @SerializedName("accuracy") val accuracy: Float
)
