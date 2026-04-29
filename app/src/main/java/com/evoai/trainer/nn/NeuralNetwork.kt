package com.evoai.trainer.nn

import kotlin.math.exp
import kotlin.random.Random

/**
 * Lightweight feedforward Neural Network with configurable hidden layers.
 * Supports forward pass, weight mutation, and serialization.
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
     * Create a mutated clone of this network.
     * mutationRate controls the probability and magnitude of weight perturbation.
     */
    fun mutate(mutationRate: Float): NeuralNetwork {
        val newWeights = mutableListOf<Array<FloatArray>>()
        val newBiases = mutableListOf<FloatArray>()

        for (layerIdx in weights.indices) {
            val w = weights[layerIdx]
            val b = biases[layerIdx]
            val newW = Array(w.size) { j -> FloatArray(w[j].size) }
            val newB = FloatArray(b.size)

            for (j in w.indices) {
                for (k in w[j].indices) {
                    newW[j][k] = if (rng.nextFloat() < mutationRate) {
                        w[j][k] + (rng.nextFloat() * 2 - 1) * mutationRate * 2f
                    } else {
                        w[j][k]
                    }
                }
                newB[j] = if (rng.nextFloat() < mutationRate) {
                    b[j] + (rng.nextFloat() * 2 - 1) * mutationRate * 2f
                } else {
                    b[j]
                }
            }
            newWeights.add(newW)
            newBiases.add(newB)
        }

        return NeuralNetwork(layerSizes, newWeights, newBiases)
    }

    /**
     * Serialize weights to a flat string for Room DB storage.
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

    companion object {
        fun sigmoid(x: Float): Float = (1f / (1f + exp(-x.toDouble()))).toFloat()

        /**
         * Deserialize from flat string.
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
    }
}
