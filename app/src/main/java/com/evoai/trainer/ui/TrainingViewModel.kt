package com.evoai.trainer.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.evoai.trainer.data.AppDatabase
import com.evoai.trainer.data.BotEntity
import com.evoai.trainer.data.DatasetMetaEntity
import com.evoai.trainer.data.TrainingHistoryEntity
import com.evoai.trainer.ga.Bot
import com.evoai.trainer.ga.GeneticTrainer
import com.evoai.trainer.util.ZipDatasetParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TrainingViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getInstance(application)

    // Training state
    private var trainer: GeneticTrainer? = null
    private var isTraining = false
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()

    // LiveData
    private val _generation = MutableLiveData(0)
    val generation: LiveData<Int> = _generation

    private val _bestAccuracy = MutableLiveData(0f)
    val bestAccuracy: LiveData<Float> = _bestAccuracy

    private val _avgFitness = MutableLiveData(0f)
    val avgFitness: LiveData<Float> = _avgFitness

    private val _bots = MutableLiveData<List<Bot>>(emptyList())
    val bots: LiveData<List<Bot>> = _bots

    private val _datasetInfo = MutableLiveData<ZipDatasetParser.DatasetResult?>(null)
    val datasetInfo: LiveData<ZipDatasetParser.DatasetResult?> = _datasetInfo

    private val _isTraining = MutableLiveData(false)
    val isTrainingLive: LiveData<Boolean> = _isTraining

    private val _trainingComplete = MutableLiveData(false)
    val trainingComplete: LiveData<Boolean> = _trainingComplete

    private val _error = MutableLiveData<String?>(null)
    val error: LiveData<String?> = _error

    // Fitness history for graph
    private val _fitnessHistory = MutableLiveData<List<Pair<Int, Float>>>(emptyList())
    val fitnessHistory: LiveData<List<Pair<Int, Float>>> = _fitnessHistory

    private var mutationRate: Float = 0.05f

    fun setMutationRate(rate: Float) {
        mutationRate = rate
    }

    /**
     * Load and parse ZIP dataset from URI.
     */
    fun loadDataset(uri: Uri) {
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val inputStream = getApplication<Application>().contentResolver.openInputStream(uri)
                        ?: throw Exception("Cannot open file")

                    val zipInputStream = java.util.zip.ZipInputStream(inputStream)
                    ZipDatasetParser.parseZip(zipInputStream)
                }

                dataset = result.samples
                _datasetInfo.value = result

                // Save metadata
                withContext(Dispatchers.IO) {
                    db.botDao().insertDatasetMeta(
                        DatasetMetaEntity(
                            fileName = uri.lastPathSegment ?: "unknown.zip",
                            likeCount = result.likeCount,
                            nonlikeCount = result.nonlikeCount,
                            totalSamples = result.samples.size,
                            featureSize = result.featureSize
                        )
                    )
                }

                // Initialize trainer with dataset
                trainer = GeneticTrainer(
                    populationSize = 10,
                    inputSize = result.featureSize,
                    mutationRate = mutationRate
                ).apply {
                    setDataset(dataset)
                    initializePopulation()
                    _bots.value = getBots()
                }
            } catch (e: Exception) {
                _error.value = "Failed to load dataset: ${e.message}"
            }
        }
    }

    /**
     * Start the evolutionary training loop.
     */
    fun startTraining() {
        if (isTraining) return
        if (dataset.isEmpty()) {
            _error.value = "Please import a ZIP dataset first"
            return
        }

        val currentTrainer = trainer
        if (currentTrainer == null) {
            _error.value = "Please import a ZIP dataset first"
            return
        }

        isTraining = true
        _isTraining.value = true
        _trainingComplete.value = false

        viewModelScope.launch {
            try {
                var targetReached = false
                val history = mutableListOf<Pair<Int, Float>>()

                // Load existing history
                val existingHistory = withContext(Dispatchers.IO) { db.botDao().getTrainingHistory() }
                for (h in existingHistory) {
                    history.add(Pair(h.generation, h.bestAccuracy))
                }

                while (isTraining && !targetReached) {
                    targetReached = withContext(Dispatchers.Default) {
                        currentTrainer.runGeneration(mutationRate)
                    }

                    val gen = currentTrainer.getGeneration()
                    val acc = currentTrainer.getBestAccuracy()
                    val avg = currentTrainer.getBots().map { it.fitness }.average().toFloat()

                    _generation.postValue(gen)
                    _bestAccuracy.postValue(acc)
                    _avgFitness.postValue(avg)
                    _bots.postValue(currentTrainer.getBots())

                    history.add(Pair(gen, acc))
                    _fitnessHistory.postValue(history.toList())

                    // Save to database
                    withContext(Dispatchers.IO) {
                        db.botDao().insertHistory(
                            TrainingHistoryEntity(
                                generation = gen,
                                bestAccuracy = acc,
                                avgFitness = avg,
                                mutationRate = mutationRate
                            )
                        )

                        // Save bot states
                        val botEntities = currentTrainer.getBots().mapIndexed { index, bot ->
                            BotEntity(
                                id = index,
                                name = String.format("BOT-%02d", index + 1),
                                serializedNetwork = bot.network.serialize(),
                                fitness = bot.fitness,
                                accuracy = bot.accuracy,
                                generation = gen,
                                isBest = bot.status == com.evoai.trainer.ga.BotStatus.BEST
                            )
                        }
                        db.botDao().insertBots(botEntities)
                    }

                    if (targetReached) {
                        _trainingComplete.postValue(true)
                    }

                    // Small delay to prevent overwhelming the system
                    withContext(Dispatchers.IO) {
                        Thread.sleep(100)
                    }
                }
            } catch (e: Exception) {
                _error.postValue("Training error: ${e.message}")
            } finally {
                isTraining = false
                _isTraining.postValue(false)
            }
        }
    }

    /**
     * Stop training.
     */
    fun stopTraining() {
        isTraining = false
        _isTraining.value = false
    }

    /**
     * Reset all stored data.
     */
    fun resetStorage() {
        viewModelScope.launch {
            stopTraining()
            withContext(Dispatchers.IO) {
                db.botDao().clearAll()
            }
            trainer = null
            dataset = emptyList()
            _generation.value = 0
            _bestAccuracy.value = 0f
            _avgFitness.value = 0f
            _bots.value = emptyList()
            _datasetInfo.value = null
            _fitnessHistory.value = emptyList()
            _trainingComplete.value = false
        }
    }

    /**
     * Restore saved state from database.
     */
    fun restoreState() {
        viewModelScope.launch {
            try {
                val meta = withContext(Dispatchers.IO) { db.botDao().getDatasetMeta() }
                val history = withContext(Dispatchers.IO) { db.botDao().getTrainingHistory() }

                if (history.isNotEmpty()) {
                    val lastEntry = history.last()
                    _generation.value = lastEntry.generation
                    _bestAccuracy.value = lastEntry.bestAccuracy
                    _avgFitness.value = lastEntry.avgFitness

                    val historyPairs = history.map { Pair(it.generation, it.bestAccuracy) }
                    _fitnessHistory.value = historyPairs
                }

                if (meta != null) {
                    _datasetInfo.value = ZipDatasetParser.DatasetResult(
                        samples = emptyList(),
                        likeCount = meta.likeCount,
                        nonlikeCount = meta.nonlikeCount,
                        featureSize = meta.featureSize
                    )
                }
            } catch (e: Exception) {
                // Ignore restore errors
            }
        }
    }
}
