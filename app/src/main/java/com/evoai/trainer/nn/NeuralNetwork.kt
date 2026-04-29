package com.evoai.trainer.nn

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlin.math.exp
import kotlin.random.Random

/**
 * Lightweight feedforward Neural Network with configurable hidden layers.
 * Supports forward pass, weight mutation, deep-clone, and JSON serialization.
 * No external ML dependencies required.
 */
class NeuralNetwork(
    val layerSizes: IntArray,
    var weights: MutableList<Array<FloatArray>> = mutableListOf(),
    var biases: MutableList<FloatArray> = mutableListOf()
) {
    private val rng = Random.Default

    init {
        if (weights.isEmpty() || biases.isEmpty()) {
            initializeWeights()
        }
    }

    private fun initializeWeights() {
        weights.clear()
        biases.clear()
        for (i in 0 until layerSizes.size - 1) {
            val inputSize = layerSizes[i]
            val outputSize = layerSizes[i + 1]
            val w = Array(outputSize) { FloatArray(inputSize) }
            val b = FloatArray(outputSize)

            // Xavier initialization
            val scale = Math.sqrt(2.0 / (inputSize + outputSize)).toFloat()
            for (j in 0 until outputSize) {
                for (k in 0 until inputSize) {
                    w[j][k] = (rng.nextFloat() * 2 - 1) * scale
                }
                b[j] = (rng.nextFloat() * 2 - 1) * 0.01f
            }
            weights.add(w)
            biases.add(b)
        }
    }

    /**
     * Forward pass through the network.
     * Returns output array (last layer activations).
     */
    fun forward(input: FloatArray): FloatArray {
        var activation = input
        for (layerIdx in weights.indices) {
            val w = weights[layerIdx]
            val b = biases[layerIdx]
            val output = FloatArray(w.size)
            for (j in w.indices) {
                var sum = b[j]
                for (k in activation.indices) {
                    sum += w[j][k] * activation[k]
                }
                output[j] = sigmoid(sum)
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
     * Get confidence score for the prediction (0.0 to 1.0).
     */
    fun predictWithConfidence(input: FloatArray): Pair<Int, Float> {
        val output = forward(input)
        val confidence = if (output[0] > 0.5f) output[0] else 1f - output[0]
        val label = if (output[0] > 0.5f) 1 else 0
        return Pair(label, confidence)
    }

    /**
     * Deep-clone this network (exact copy, no mutation).
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

        return NeuralNetwork(layerSizes.copyOf(), newWeights, newBiases)
    }

    /**
     * Create a mutated clone of this network.
     * mutationRate controls the probability and magnitude of weight perturbation.
     */
    fun mutate(mutationRate: Float): NeuralNetwork {
        val clone = deepClone()

        for (layerIdx in clone.weights.indices) {
            val w = clone.weights[layerIdx]
            val b = clone.biases[layerIdx]

            for (j in w.indices) {
                for (k in w[j].indices) {
                    if (rng.nextFloat() < mutationRate) {
                        w[j][k] += (rng.nextFloat() * 2 - 1) * mutationRate * 2f
                    }
                }
                if (rng.nextFloat() < mutationRate) {
                    b[j] += (rng.nextFloat() * 2 - 1) * mutationRate * 2f
                }
            }
        }

        return clone
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
     * Export model weights as JSON (for .model file - inference only).
     */
    fun toJson(): String {
        val gson = Gson()
        val layers = mutableListOf<LayerData>()
        for (i in weights.indices) {
            val weightRows = weights[i].map { row -> row.toList() }
            layers.add(LayerData(weightRows, biases[i].toList()))
        }
        val modelData = ModelData(
            layerSizes = layerSizes.toList(),
            layers = layers
        )
        return gson.toJson(modelData)
    }

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
            }

            return NeuralNetwork(layerSizes, weights, biases)
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

            for (layer in modelData.layers) {
                val w = Array(layer.weights.size) { j -> layer.weights[j].toFloatArray() }
                val b = layer.biases.toFloatArray()
                weights.add(w)
                biases.add(b)
            }

            return NeuralNetwork(layerSizes, weights, biases)
        }
    }
}

// JSON serialization helpers
data class ModelData(
    @SerializedName("layerSizes") val layerSizes: List<Int>,
    @SerializedName("layers") val layers: List<LayerData>
)

data class LayerData(
    @SerializedName("weights") val weights: List<List<Float>>,
    @SerializedName("biases") val biases: List<Float>
)

/**
 * Full checkpoint data for .ckpt export (includes training state).
 */
data class CheckpointData(
    @SerializedName("model") val model: ModelData,
    @SerializedName("generation") val generation: Int,
    @SerializedName("bestAccuracy") val bestAccuracy: Float,
    @SerializedName("avgFitness") val avgFitness: Float,
    @SerializedName("mutationRate") val mutationRate: Float,
    @SerializedName("fitnessHistory") val fitnessHistory: List<HistoryEntry>
)

data class HistoryEntry(
    @SerializedName("generation") val generation: Int,
    @SerializedName("accuracy") val accuracy: Float
)
