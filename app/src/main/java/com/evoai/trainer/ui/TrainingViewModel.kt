package com.evoai.trainer.ui

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.evoai.trainer.data.AppDatabase
import com.evoai.trainer.data.BestModelCheckpointEntity
import com.evoai.trainer.data.BotEntity
import com.evoai.trainer.data.DatasetMetaEntity
import com.evoai.trainer.data.TrainingHistoryEntity
import com.evoai.trainer.ga.Bot
import com.evoai.trainer.ga.BotStatus
import com.evoai.trainer.ga.GeneticTrainer
import com.evoai.trainer.nn.CheckpointData
import com.evoai.trainer.nn.HistoryEntry
import com.evoai.trainer.nn.NeuralNetwork
import com.evoai.trainer.util.ZipDatasetParser
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStream

class TrainingViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getInstance(application)
    private val gson = Gson()

    // Training state
    private var trainer: GeneticTrainer? = null
    private var isTraining = false
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()
    private var featureSize: Int = 256 // 16x16

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

    private val _fitnessHistory = MutableLiveData<List<Pair<Int, Float>>>(emptyList())
    val fitnessHistory: LiveData<List<Pair<Int, Float>>> = _fitnessHistory

    // Inference test result
    private val _inferenceResult = MutableLiveData<InferenceResult?>(null)
    val inferenceResult: LiveData<InferenceResult?> = _inferenceResult

    // Export status
    private val _exportStatus = MutableLiveData<String?>(null)
    val exportStatus: LiveData<String?> = _exportStatus

    // Import status
    private val _importStatus = MutableLiveData<String?>(null)
    val importStatus: LiveData<String?> = _importStatus

    private var mutationRate: Float = 0.05f
    private var targetAccuracy: Float = 90f

    fun setMutationRate(rate: Float) {
        mutationRate = rate
    }

    fun setTargetAccuracy(target: Float) {
        targetAccuracy = target
        trainer?.setTargetAccuracy(target)
    }

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
                featureSize = result.featureSize
                _datasetInfo.value = result

                withContext(Dispatchers.IO) {
                    db.botDao().insertDatasetMeta(
                        DatasetMetaEntity(
                            fileName = uri.lastPathSegment ?: "unknown.zip",
                            likeCount = result.likeCount,
                            nonlikeCount = result.nonlikeCount,
                            totalSamples = result.totalSamples,
                            featureSize = result.featureSize
                        )
                    )
                }

                trainer = GeneticTrainer(
                    populationSize = 10,
                    inputSize = result.featureSize
                ).apply {
                    setDataset(dataset)
                    setTargetAccuracy(targetAccuracy)
                    initializePopulation()
                    _bots.value = getBots()
                }
            } catch (e: Exception) {
                _error.value = "Failed to load dataset: ${e.message}"
            }
        }
    }

    fun startTraining() {
        if (isTraining) return
        if (dataset.isEmpty()) {
            _error.value = "Please import a ZIP dataset first"
            return
        }

        val currentTrainer = trainer ?: run {
            _error.value = "Please import a ZIP dataset first"
            return
        }

        isTraining = true
        _isTraining.value = true
        _trainingComplete.value = false

        // Set up auto-save callback
        currentTrainer.onAutoSave = { bestNetwork, gen, acc ->
            viewModelScope.launch {
                withContext(Dispatchers.IO) {
                    db.botDao().insertCheckpoint(
                        BestModelCheckpointEntity(
                            generation = gen,
                            serializedNetwork = bestNetwork.serialize(),
                            accuracy = acc,
                            fitness = 0f,
                            mutationRate = this@TrainingViewModel.mutationRate,
                            targetAccuracy = this@TrainingViewModel.targetAccuracy
                        )
                    )
                }
            }
        }

        viewModelScope.launch {
            try {
                var targetReached = false
                val history = mutableListOf<Pair<Int, Float>>()

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

                    // Save history to DB
                    withContext(Dispatchers.IO) {
                        db.botDao().insertHistory(
                            TrainingHistoryEntity(
                                generation = gen,
                                bestAccuracy = acc,
                                avgFitness = avg,
                                mutationRate = mutationRate
                            )
                        )

                        val botEntities = currentTrainer.getBots().mapIndexed { index, bot ->
                            BotEntity(
                                id = index,
                                name = String.format("BOT-%02d", index + 1),
                                serializedNetwork = bot.network.serialize(),
                                fitness = bot.fitness,
                                accuracy = bot.accuracy,
                                generation = gen,
                                isBest = bot.status == BotStatus.BEST
                            )
                        }
                        db.botDao().insertBots(botEntities)
                    }

                    if (targetReached) {
                        _trainingComplete.postValue(true)
                    }

                    withContext(Dispatchers.IO) {
                        Thread.sleep(50)
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

    fun stopTraining() {
        isTraining = false
        _isTraining.value = false
    }

    fun resetStorage() {
        viewModelScope.launch {
            stopTraining()
            withContext(Dispatchers.IO) {
                db.botDao().deleteAllBots()
                db.botDao().deleteAllHistory()
                db.botDao().deleteDatasetMeta()
                db.botDao().deleteAllCheckpoints()
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
            _inferenceResult.value = null
        }
    }

    // ========== Export / Import ==========

    /**
     * Export best model as .model file (JSON, inference-only weights).
     */
    fun exportModel() {
        val bestNetwork = trainer?.getBestNetwork()
        if (bestNetwork == null) {
            _error.value = "No trained model to export"
            return
        }

        viewModelScope.launch {
            try {
                val json = bestNetwork.toJson()
                val fileName = "evoai_model_gen${_generation.value}.model"

                withContext(Dispatchers.IO) {
                    saveFileToDownloads(fileName, json.toByteArray())
                }

                _exportStatus.value = "Model exported: $fileName"
            } catch (e: Exception) {
                _error.value = "Export failed: ${e.message}"
            }
        }
    }

    /**
     * Export full training checkpoint as .ckpt file (JSON, includes training state).
     */
    fun exportCheckpoint() {
        val bestNetwork = trainer?.getBestNetwork()
        if (bestNetwork == null) {
            _error.value = "No trained model to export"
            return
        }

        viewModelScope.launch {
            try {
                val modelData = com.evoai.trainer.nn.ModelData(
                    layerSizes = bestNetwork.layerSizes.toList(),
                    layers = bestNetwork.weights.indices.map { i ->
                        com.evoai.trainer.nn.LayerData(
                            weights = bestNetwork.weights[i].map { row -> row.toList() },
                            biases = bestNetwork.biases[i].toList()
                        )
                    }
                )

                val historyEntries = _fitnessHistory.value?.map { (gen, acc) ->
                    HistoryEntry(gen, acc)
                } ?: emptyList()

                val checkpoint = CheckpointData(
                    model = modelData,
                    generation = _generation.value ?: 0,
                    bestAccuracy = _bestAccuracy.value ?: 0f,
                    avgFitness = _avgFitness.value ?: 0f,
                    mutationRate = mutationRate,
                    fitnessHistory = historyEntries
                )

                val json = gson.toJson(checkpoint)
                val fileName = "evoai_checkpoint_gen${_generation.value}.ckpt"

                withContext(Dispatchers.IO) {
                    saveFileToDownloads(fileName, json.toByteArray())
                }

                _exportStatus.value = "Checkpoint exported: $fileName"
            } catch (e: Exception) {
                _error.value = "Export failed: ${e.message}"
            }
        }
    }

    /**
     * Import a .model or .ckpt file.
     */
    fun importModel(uri: Uri) {
        viewModelScope.launch {
            try {
                val content = withContext(Dispatchers.IO) {
                    val inputStream = getApplication<Application>().contentResolver.openInputStream(uri)
                        ?: throw Exception("Cannot open file")
                    inputStream.bufferedReader().use { it.readText() }
                }

                val fileName = uri.lastPathSegment ?: "unknown"

                when {
                    fileName.endsWith(".ckpt") -> {
                        val checkpoint = gson.fromJson(content, CheckpointData::class.java)
                        val network = NeuralNetwork.fromJson(gson.toJson(checkpoint.model))

                        _generation.value = checkpoint.generation
                        _bestAccuracy.value = checkpoint.bestAccuracy
                        _avgFitness.value = checkpoint.avgFitness
                        mutationRate = checkpoint.mutationRate

                        val historyPairs = checkpoint.fitnessHistory.map { Pair(it.generation, it.accuracy) }
                        _fitnessHistory.value = historyPairs

                        // Recreate trainer with the loaded network
                        trainer = GeneticTrainer(populationSize = 10, inputSize = network.layerSizes.first()).apply {
                            setDataset(dataset)
                            setTargetAccuracy(targetAccuracy)
                            initializePopulation()
                            // Replace first bot with the loaded network
                            val bots = getBots().toMutableList()
                            bots[0] = Bot(id = 0, network = network, fitness = checkpoint.bestAccuracy / 100f, accuracy = checkpoint.bestAccuracy, status = BotStatus.BEST)
                            for (i in 1 until bots.size) {
                                bots[i] = Bot(id = i, network = network.mutate(mutationRate))
                            }
                        }
                        _bots.value = trainer?.getBots() ?: emptyList()

                        _importStatus.value = "Checkpoint loaded: Gen ${checkpoint.generation}, Acc ${String.format("%.1f%%", checkpoint.bestAccuracy)}"
                    }
                    fileName.endsWith(".model") -> {
                        val network = NeuralNetwork.fromJson(content)

                        trainer = GeneticTrainer(populationSize = 10, inputSize = network.layerSizes.first()).apply {
                            setDataset(dataset)
                            setTargetAccuracy(targetAccuracy)
                            initializePopulation()
                            val bots = getBots().toMutableList()
                            bots[0] = Bot(id = 0, network = network, status = BotStatus.BEST)
                            for (i in 1 until bots.size) {
                                bots[i] = Bot(id = i, network = network.mutate(mutationRate))
                            }
                        }
                        _bots.value = trainer?.getBots() ?: emptyList()

                        _importStatus.value = "Model loaded successfully"
                    }
                    else -> {
                        _error.value = "Unsupported file format. Use .model or .ckpt files"
                    }
                }
            } catch (e: Exception) {
                _error.value = "Import failed: ${e.message}"
            }
        }
    }

    // ========== Live Testing ==========

    /**
     * Run inference on a single image from the gallery.
     */
    fun testImage(uri: Uri) {
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val inputStream = getApplication<Application>().contentResolver.openInputStream(uri)
                        ?: throw Exception("Cannot open image")
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                        ?: throw Exception("Cannot decode image")

                    val features = ZipDatasetParser.extractFeaturesFromBitmap(bitmap)
                    bitmap.recycle()

                    val currentTrainer = trainer ?: throw Exception("No model available - train first")
                    currentTrainer.predict(features)
                }

                _inferenceResult.value = InferenceResult(
                    label = if (result.first == 1) "Like" else "Non-like",
                    confidence = result.second * 100f
                )
            } catch (e: Exception) {
                _error.value = "Inference failed: ${e.message}"
            }
        }
    }

    // ========== Helpers ==========

    private fun saveFileToDownloads(fileName: String, data: ByteArray) {
        val context = getApplication<Application>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                put(MediaStore.Downloads.MIME_TYPE, "application/json")
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                context.contentResolver.openOutputStream(it)?.use { os: OutputStream ->
                    os.write(data)
                }
                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                context.contentResolver.update(uri, values, null, null)
            }
        } else {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val file = java.io.File(dir, fileName)
            file.writeBytes(data)
        }
    }

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
            } catch (_: Exception) {
                // Ignore restore errors
            }
        }
    }
}

data class InferenceResult(
    val label: String,
    val confidence: Float
)
