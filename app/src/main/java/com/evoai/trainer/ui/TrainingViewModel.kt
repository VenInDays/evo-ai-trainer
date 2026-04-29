package com.evoai.trainer.ui

import android.app.Application
import android.content.ContentValues
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
import com.evoai.trainer.data.HardExampleEntity
import com.evoai.trainer.data.TrainingHistoryEntity
import com.evoai.trainer.ga.Bot
import com.evoai.trainer.ga.BotLineage
import com.evoai.trainer.ga.BotStatus
import com.evoai.trainer.ga.GeneticTrainer
import com.evoai.trainer.ga.TeacherBot
import com.evoai.trainer.nn.CheckpointData
import com.evoai.trainer.nn.HistoryEntry
import com.evoai.trainer.nn.NeuralNetwork
import com.evoai.trainer.util.ZipDatasetParser
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.OutputStream

class TrainingViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getInstance(application)
    private val gson = Gson()
    private val appContext = application.applicationContext

    // Training state
    private var trainer: GeneticTrainer? = null
    private var isTraining = false
    private var dataset: List<Pair<FloatArray, Int>> = emptyList()
    private var trainSet: List<Pair<FloatArray, Int>> = emptyList()
    private var valSet: List<Pair<FloatArray, Int>> = emptyList()
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

    // V3: Stagnant generations counter
    private val _stagnantGenerations = MutableLiveData(0)
    val stagnantGenerations: LiveData<Int> = _stagnantGenerations

    // V3: History log text (last 10 generations)
    private val _historyLog = MutableLiveData("")
    val historyLog: LiveData<String> = _historyLog

    // V3: Global Best (all-time highest accuracy ever recorded)
    private val _globalBest = MutableLiveData(0f)
    val globalBest: LiveData<Float> = _globalBest

    // V3: Active Mutation Rate (including auto-boosts from hyper-mutation)
    private val _activeMutRate = MutableLiveData(0.05f)
    val activeMutRate: LiveData<Float> = _activeMutRate

    // V4: Decaying Mutation Rate
    private val _decayingMutRate = MutableLiveData(0.05f)
    val decayingMutRate: LiveData<Float> = _decayingMutRate

    // V3: Hyper-mutation notification
    private val _hyperMutationEvent = MutableLiveData<String?>(null)
    val hyperMutationEvent: LiveData<String?> = _hyperMutationEvent

    // V4: Jitter event
    private val _jitterEvent = MutableLiveData<String?>(null)
    val jitterEvent: LiveData<String?> = _jitterEvent

    // V4: Confusion Matrix
    private val _confusionMatrix = MutableLiveData<TeacherBot.EvaluationResult?>(null)
    val confusionMatrix: LiveData<TeacherBot.EvaluationResult?> = _confusionMatrix

    // V4: Hard examples count
    private val _hardExamplesCount = MutableLiveData(0)
    val hardExamplesCount: LiveData<Int> = _hardExamplesCount

    // Inference test result
    private val _inferenceResult = MutableLiveData<InferenceResult?>(null)
    val inferenceResult: LiveData<InferenceResult?> = _inferenceResult

    // Export status
    private val _exportStatus = MutableLiveData<String?>(null)
    val exportStatus: LiveData<String?> = _exportStatus

    // Import status
    private val _importStatus = MutableLiveData<String?>(null)
    val importStatus: LiveData<String?> = _importStatus

    // V4: Auto-recovery status
    private val _recoveryStatus = MutableLiveData<String?>(null)
    val recoveryStatus: LiveData<String?> = _recoveryStatus

    private var mutationRate: Float = 0.05f
    private var targetAccuracy: Float = 90f

    // History log entries buffer
    private val historyLogEntries = mutableListOf<String>()

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
                trainSet = result.trainSamples
                valSet = result.valSamples
                featureSize = result.featureSize
                _datasetInfo.value = result

                withContext(Dispatchers.IO) {
                    db.botDao().insertDatasetMeta(
                        DatasetMetaEntity(
                            fileName = uri.lastPathSegment ?: "unknown.zip",
                            likeCount = result.likeCount,
                            nonlikeCount = result.nonlikeCount,
                            totalSamples = result.totalSamples,
                            trainSamples = result.trainSamples.size,
                            valSamples = result.valSamples.size,
                            featureSize = result.featureSize
                        )
                    )
                }

                trainer = GeneticTrainer(
                    populationSize = 10,
                    inputSize = result.featureSize
                ).apply {
                    // V4: Set cross-validation sets
                    setCrossValidationSets(trainSet, valSet)
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

        // V4: Auto-save callback — save to Internal Storage (.json)
        currentTrainer.onAutoSave = { networks, gen, acc ->
            viewModelScope.launch {
                withContext(Dispatchers.IO) {
                    // V4: Save champion to Internal Storage instead of SharedPreferences
                    if (networks.isNotEmpty()) {
                        val championJson = networks[0].toJson()
                        saveModelToInternalStorage("champion_gen$gen.model", championJson)
                        saveModelToInternalStorage("latest_checkpoint.model", championJson)

                        // Also save checkpoint with full state
                        val checkpoint = CheckpointData(
                            model = com.evoai.trainer.nn.ModelData(
                                layerSizes = networks[0].layerSizes.toList(),
                                layers = networks[0].weights.indices.map { i ->
                                    com.evoai.trainer.nn.LayerData(
                                        weights = networks[0].weights[i].map { row -> row.toList() },
                                        biases = networks[0].biases[i].toList(),
                                        activation = networks[0].activations[i].name
                                    )
                                }
                            ),
                            generation = gen,
                            bestAccuracy = acc,
                            avgFitness = 0f,
                            avgLoss = 0f,
                            mutationRate = mutationRate,
                            decayingMutationRate = currentTrainer.getDecayingMutationRate(),
                            fitnessHistory = emptyList()
                        )
                        val checkpointJson = gson.toJson(checkpoint)
                        saveModelToInternalStorage("latest_checkpoint.ckpt", checkpointJson)

                        // Also save to Room for backward compat
                        db.botDao().insertCheckpoint(
                            BestModelCheckpointEntity(
                                generation = gen,
                                serializedNetwork = networks[0].serialize(),
                                accuracy = acc,
                                fitness = 0f,
                                loss = 0f,
                                mutationRate = mutationRate,
                                decayingMutationRate = currentTrainer.getDecayingMutationRate(),
                                targetAccuracy = targetAccuracy
                            )
                        )
                    }
                }
            }
        }

        // V3: Hyper-mutation callback
        currentTrainer.onHyperMutation = { gen, boostedRate ->
            _hyperMutationEvent.postValue(String.format("HYPER-MUTATION at Gen %d! Rate: %.3f", gen, boostedRate))
        }

        // V4: Jitter callback
        currentTrainer.onJitterApplied = { gen ->
            _jitterEvent.postValue(String.format("NEURAL JITTER at Gen %d! Noise applied to ALL weights", gen))
        }

        viewModelScope.launch {
            try {
                var targetReached = false
                val history = mutableListOf<Pair<Int, Float>>()

                val existingHistory = withContext(Dispatchers.IO) { db.botDao().getTrainingHistory() }
                for (h in existingHistory) {
                    history.add(Pair(h.generation, h.bestAccuracy))
                }

                // V4: Load hard examples from DB and add to trainer
                val hardExamples = withContext(Dispatchers.IO) { db.botDao().getHardExamples() }
                for (ex in hardExamples) {
                    val features = ex.serializedFeatures.split(",").mapNotNull { it.toFloatOrNull() }.toFloatArray()
                    currentTrainer.addHardExample(features, ex.label)
                }
                _hardExamplesCount.postValue(currentTrainer.getHardExamplesCount())

                while (isTraining && !targetReached) {
                    targetReached = withContext(Dispatchers.Default) {
                        currentTrainer.runGeneration(mutationRate)
                    }

                    val gen = currentTrainer.getGeneration()
                    val acc = currentTrainer.getBestAccuracy()
                    val avg = currentTrainer.getBots().map { it.fitness }.average().toFloat()
                    val stagnant = currentTrainer.getStagnantGenerations()
                    val confusion = currentTrainer.getConfusionMatrix()

                    _generation.postValue(gen)
                    _bestAccuracy.postValue(acc)
                    _avgFitness.postValue(avg)
                    _stagnantGenerations.postValue(stagnant)
                    _globalBest.postValue(currentTrainer.getAllTimeBestAccuracy())
                    _activeMutRate.postValue(currentTrainer.getActiveMutationRate())
                    _decayingMutRate.postValue(currentTrainer.getDecayingMutationRate())
                    _bots.postValue(currentTrainer.getBots())
                    _confusionMatrix.postValue(confusion)

                    history.add(Pair(gen, acc))
                    _fitnessHistory.postValue(history.toList())

                    // V4 FIX: Use getLastGenAvgLoss() from trainer — computed from EVALUATED bots
                    // BEFORE new generation replaces them. The old approach was broken because
                    // getBots() returns NEW unevaluated bots with loss=Float.MAX_VALUE,
                    // causing .average() on empty list → NaN → SQLite NOT NULL constraint failure.
                    val avgLoss = currentTrainer.getLastGenAvgLoss()
                    val logEntry = String.format(
                        "Gen %d: Val %.1f%% | Loss %.3f | Decay %.3f | Stag %d",
                        gen, acc, avgLoss, currentTrainer.getDecayingMutationRate(), stagnant
                    )
                    historyLogEntries.add(logEntry)
                    if (historyLogEntries.size > 10) {
                        historyLogEntries.removeAt(0)
                    }
                    _historyLog.postValue(historyLogEntries.joinToString("\n"))

                    // Save history to DB
                    withContext(Dispatchers.IO) {
                        db.botDao().insertHistory(
                            TrainingHistoryEntity(
                                generation = gen,
                                bestAccuracy = acc,
                                bestValAccuracy = acc,
                                avgFitness = avg,
                                avgLoss = avgLoss,
                                mutationRate = mutationRate,
                                decayingMutationRate = currentTrainer.getDecayingMutationRate()
                            )
                        )

                        val botEntities = currentTrainer.getBots().mapIndexed { index, bot ->
                            BotEntity(
                                id = index,
                                name = String.format("BOT-%02d", index + 1),
                                serializedNetwork = bot.network.serialize(),
                                fitness = bot.fitness,
                                accuracy = bot.accuracy,
                                valAccuracy = bot.valAccuracy,
                                loss = bot.loss,
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
                db.botDao().deleteAllHardExamples()

                // V4: Delete internal storage model files
                deleteInternalStorageModel("latest_checkpoint.model")
                deleteInternalStorageModel("latest_checkpoint.ckpt")
            }
            trainer = null
            dataset = emptyList()
            trainSet = emptyList()
            valSet = emptyList()
            historyLogEntries.clear()
            _generation.value = 0
            _bestAccuracy.value = 0f
            _avgFitness.value = 0f
            _stagnantGenerations.value = 0
            _globalBest.value = 0f
            _activeMutRate.value = 0.05f
            _decayingMutRate.value = 0.05f
            _bots.value = emptyList()
            _datasetInfo.value = null
            _fitnessHistory.value = emptyList()
            _historyLog.value = ""
            _trainingComplete.value = false
            _inferenceResult.value = null
            _confusionMatrix.value = null
            _hardExamplesCount.value = 0
        }
    }

    // ========== V4: Internal Storage Model Save/Load ==========

    /**
     * V4: Save model JSON to Internal Storage (context.filesDir).
     * Replaces SharedPreferences for large weight arrays — more reliable.
     */
    private fun saveModelToInternalStorage(fileName: String, json: String) {
        val file = File(appContext.filesDir, fileName)
        file.writeText(json)
    }

    /**
     * V4: Load model JSON from Internal Storage.
     */
    private fun loadModelFromInternalStorage(fileName: String): String? {
        val file = File(appContext.filesDir, fileName)
        return if (file.exists()) file.readText() else null
    }

    /**
     * V4: Delete a model file from Internal Storage.
     */
    private fun deleteInternalStorageModel(fileName: String) {
        val file = File(appContext.filesDir, fileName)
        if (file.exists()) file.delete()
    }

    // ========== Export / Import ==========

    /**
     * Export best model as .model file (JSON, inference-only weights).
     * V4: Also saves to Internal Storage for auto-recovery.
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
                    // V4: Save to Internal Storage first
                    saveModelToInternalStorage("export_$fileName", json)
                    // Also save to Downloads for user access
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
                            biases = bestNetwork.biases[i].toList(),
                            activation = bestNetwork.activations[i].name
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
                    avgLoss = 0f,
                    mutationRate = mutationRate,
                    decayingMutationRate = trainer?.getDecayingMutationRate() ?: mutationRate,
                    fitnessHistory = historyEntries
                )

                val json = gson.toJson(checkpoint)
                val fileName = "evoai_checkpoint_gen${_generation.value}.ckpt"

                withContext(Dispatchers.IO) {
                    saveModelToInternalStorage("export_$fileName", json)
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

                        trainer = GeneticTrainer(populationSize = 10, inputSize = network.layerSizes.first()).apply {
                            setCrossValidationSets(trainSet, valSet)
                            setTargetAccuracy(targetAccuracy)
                            initializePopulation()
                            val botsList = getBots().toMutableList()
                            botsList[0] = Bot(id = 0, network = network, fitness = checkpoint.bestAccuracy / 100f, accuracy = checkpoint.bestAccuracy, status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 1, loss = checkpoint.avgLoss)
                            botsList[1] = Bot(id = 1, network = network.deepClone(), fitness = checkpoint.bestAccuracy / 100f, accuracy = checkpoint.bestAccuracy, status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 2, loss = checkpoint.avgLoss)
                            for (i in 2 until botsList.size) {
                                botsList[i] = Bot(id = i, network = network.mutateGaussian(mutationRate), lineage = BotLineage.CLONE, mutationVariance = mutationRate, parentRank = 1)
                            }
                        }
                        _bots.value = trainer?.getBots() ?: emptyList()
                        _importStatus.value = "Checkpoint loaded: Gen ${checkpoint.generation}, Acc ${String.format("%.1f%%", checkpoint.bestAccuracy)}"
                    }
                    fileName.endsWith(".model") -> {
                        val network = NeuralNetwork.fromJson(content)

                        trainer = GeneticTrainer(populationSize = 10, inputSize = network.layerSizes.first()).apply {
                            setCrossValidationSets(trainSet, valSet)
                            setTargetAccuracy(targetAccuracy)
                            initializePopulation()
                            val botsList = getBots().toMutableList()
                            botsList[0] = Bot(id = 0, network = network, status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 1)
                            botsList[1] = Bot(id = 1, network = network.deepClone(), status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 2)
                            for (i in 2 until botsList.size) {
                                botsList[i] = Bot(id = i, network = network.mutateGaussian(mutationRate), lineage = BotLineage.CLONE, mutationVariance = mutationRate, parentRank = 1)
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
     * V4: Returns label + confidence + isUncertain flag.
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

                val isUncertain = result.second < 0.5f
                _inferenceResult.value = InferenceResult(
                    label = if (result.first == 1) "Like" else "Non-like",
                    confidence = result.second * 100f,
                    isUncertain = isUncertain
                )
            } catch (e: Exception) {
                _error.value = "Inference failed: ${e.message}"
            }
        }
    }

    // ========== V4: Manual Override (Correct the AI) ==========

    /**
     * V4: Add a corrected label for a test image to Hard Examples.
     * This image will be prioritized in the next training loop.
     */
    fun addHardExample(imageUri: Uri, correctLabel: Int) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val inputStream = getApplication<Application>().contentResolver.openInputStream(imageUri)
                        ?: throw Exception("Cannot open image")
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                        ?: throw Exception("Cannot decode image")

                    val features = ZipDatasetParser.extractFeaturesFromBitmap(bitmap)
                    bitmap.recycle()

                    val serializedFeatures = features.joinToString(",")
                    db.botDao().insertHardExample(
                        HardExampleEntity(
                            serializedFeatures = serializedFeatures,
                            label = correctLabel,
                            source = "manual"
                        )
                    )

                    // Also add to trainer for immediate effect
                    trainer?.addHardExample(features, correctLabel)
                }
                _hardExamplesCount.value = (trainer?.getHardExamplesCount() ?: 0)
                _exportStatus.value = "Hard example added! It will be prioritized in next training."
            } catch (e: Exception) {
                _error.value = "Failed to add hard example: ${e.message}"
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

    /**
     * V4: Auto-Recovery — Load Last Session from Internal Storage on startup.
     * Checks for latest_checkpoint.json first, then falls back to Room DB.
     */
    fun restoreState() {
        viewModelScope.launch {
            try {
                // V4: Try auto-recovery from Internal Storage first
                val checkpointJson = withContext(Dispatchers.IO) {
                    loadModelFromInternalStorage("latest_checkpoint.ckpt")
                }

                if (checkpointJson != null) {
                    try {
                        val checkpoint = gson.fromJson(checkpointJson, CheckpointData::class.java)
                        val network = NeuralNetwork.fromJson(gson.toJson(checkpoint.model))

                        _generation.value = checkpoint.generation
                        _bestAccuracy.value = checkpoint.bestAccuracy
                        _avgFitness.value = checkpoint.avgFitness
                        mutationRate = checkpoint.mutationRate
                        _decayingMutRate.value = checkpoint.decayingMutationRate

                        val historyPairs = checkpoint.fitnessHistory.map { Pair(it.generation, it.accuracy) }
                        _fitnessHistory.value = historyPairs

                        // Recreate trainer with the loaded network
                        trainer = GeneticTrainer(populationSize = 10, inputSize = network.layerSizes.first()).apply {
                            setCrossValidationSets(trainSet, valSet)
                            setTargetAccuracy(targetAccuracy)
                            initializePopulation()
                            val botsList = getBots().toMutableList()
                            botsList[0] = Bot(id = 0, network = network, fitness = checkpoint.bestAccuracy / 100f, accuracy = checkpoint.bestAccuracy, status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 1, loss = checkpoint.avgLoss)
                            botsList[1] = Bot(id = 1, network = network.deepClone(), fitness = checkpoint.bestAccuracy / 100f, accuracy = checkpoint.bestAccuracy, status = BotStatus.BEST, lineage = BotLineage.LEGACY, parentRank = 2, loss = checkpoint.avgLoss)
                            for (i in 2 until botsList.size) {
                                botsList[i] = Bot(id = i, network = network.mutateGaussian(mutationRate), lineage = BotLineage.CLONE, mutationVariance = mutationRate, parentRank = 1)
                            }
                        }
                        _bots.value = trainer?.getBots() ?: emptyList()

                        _recoveryStatus.value = String.format(
                            "Session recovered: Gen %d, Acc %.1f%%",
                            checkpoint.generation, checkpoint.bestAccuracy
                        )
                        return@launch
                    } catch (_: Exception) {
                        // Fall through to Room DB recovery
                    }
                }

                // Fallback: Restore from Room DB
                val meta = withContext(Dispatchers.IO) { db.botDao().getDatasetMeta() }
                val history = withContext(Dispatchers.IO) { db.botDao().getTrainingHistory() }

                if (history.isNotEmpty()) {
                    val lastEntry = history.last()
                    _generation.value = lastEntry.generation
                    _bestAccuracy.value = lastEntry.bestAccuracy
                    _avgFitness.value = lastEntry.avgFitness
                    _decayingMutRate.value = lastEntry.decayingMutationRate

                    val historyPairs = history.map { Pair(it.generation, it.bestAccuracy) }
                    _fitnessHistory.value = historyPairs

                    historyLogEntries.clear()
                    val recentHistory = history.takeLast(10)
                    for (h in recentHistory) {
                        historyLogEntries.add(
                            String.format("Gen %d: Best %.1f%% | Avg %.2f", h.generation, h.bestAccuracy, h.avgFitness)
                        )
                    }
                    _historyLog.value = historyLogEntries.joinToString("\n")
                }

                if (meta != null) {
                    _datasetInfo.value = ZipDatasetParser.DatasetResult(
                        samples = emptyList(),
                        trainSamples = emptyList(),
                        valSamples = emptyList(),
                        likeCount = meta.likeCount,
                        nonlikeCount = meta.nonlikeCount,
                        featureSize = meta.featureSize
                    )
                }

                // Load hard examples count
                val hardExamples = withContext(Dispatchers.IO) { db.botDao().getHardExamples() }
                _hardExamplesCount.value = hardExamples.size
            } catch (_: Exception) {
                // Ignore restore errors
            }
        }
    }
}

data class InferenceResult(
    val label: String,
    val confidence: Float,
    val isUncertain: Boolean = false  // V4: <50% confidence = Uncertain
)
