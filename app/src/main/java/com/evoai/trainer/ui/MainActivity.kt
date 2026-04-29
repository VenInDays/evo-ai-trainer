package com.evoai.trainer.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.evoai.trainer.R
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.slider.Slider
import androidx.activity.viewModels

class MainActivity : AppCompatActivity() {

    private val viewModel: TrainingViewModel by viewModels()

    // Dashboard views
    private lateinit var tvGenerationBig: TextView
    private lateinit var tvStagnant: TextView
    private lateinit var tvTargetIndicator: TextView
    private lateinit var tvAccuracy: TextView
    private lateinit var tvFitness: TextView
    private lateinit var tvDatasetStatus: TextView
    private lateinit var tvLikesCount: TextView
    private lateinit var tvNonlikesCount: TextView
    private lateinit var tvMutationRate: TextView
    private lateinit var tvTargetAccuracy: TextView
    private lateinit var tvTrainingStatus: TextView
    private lateinit var sliderMutationRate: Slider
    private lateinit var sliderTargetAccuracy: Slider
    private lateinit var chartFitness: LineChart
    private lateinit var rvBots: RecyclerView
    private lateinit var btnImportZip: MaterialButton
    private lateinit var btnStartTraining: MaterialButton
    private lateinit var btnResetStorage: MaterialButton
    private lateinit var btnSaveModel: MaterialButton
    private lateinit var btnLoadModel: MaterialButton
    private lateinit var btnExportCheckpoint: MaterialButton
    private lateinit var btnTestImage: MaterialButton
    private lateinit var cardTrainingStatus: MaterialCardView
    private lateinit var progressTraining: ProgressBar
    private lateinit var layoutInferenceResult: View
    private lateinit var viewInferenceIndicator: View
    private lateinit var tvInferenceLabel: TextView
    private lateinit var tvInferenceConfidence: TextView

    // v2.0: History Log views
    private lateinit var tvHistoryLog: TextView
    private lateinit var scrollHistoryLog: ScrollView

    private val botAdapter = BotAdapter()

    // ZIP file picker
    private val zipPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.loadDataset(it) } }

    // Image picker for inference test
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.testImage(it) } }

    // Model file picker (.model / .ckpt)
    private val modelPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.importModel(it) } }

    // Permission launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) openZipPicker()
        else Toast.makeText(this, "Storage permission required", Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupChart()
        setupRecyclerView()
        setupListeners()
        observeViewModel()

        viewModel.restoreState()
    }

    private fun initViews() {
        // v2.0: Prominent generation dashboard
        tvGenerationBig = findViewById(R.id.tvGenerationBig)
        tvStagnant = findViewById(R.id.tvStagnant)
        tvTargetIndicator = findViewById(R.id.tvTargetIndicator)

        tvAccuracy = findViewById(R.id.tvAccuracy)
        tvFitness = findViewById(R.id.tvFitness)
        tvDatasetStatus = findViewById(R.id.tvDatasetStatus)
        tvLikesCount = findViewById(R.id.tvLikesCount)
        tvNonlikesCount = findViewById(R.id.tvNonlikesCount)
        tvMutationRate = findViewById(R.id.tvMutationRate)
        tvTargetAccuracy = findViewById(R.id.tvTargetAccuracy)
        tvTrainingStatus = findViewById(R.id.tvTrainingStatus)
        sliderMutationRate = findViewById(R.id.sliderMutationRate)
        sliderTargetAccuracy = findViewById(R.id.sliderTargetAccuracy)
        chartFitness = findViewById(R.id.chartFitness)
        rvBots = findViewById(R.id.rvBots)
        btnImportZip = findViewById(R.id.btnImportZip)
        btnStartTraining = findViewById(R.id.btnStartTraining)
        btnResetStorage = findViewById(R.id.btnResetStorage)
        btnSaveModel = findViewById(R.id.btnSaveModel)
        btnLoadModel = findViewById(R.id.btnLoadModel)
        btnExportCheckpoint = findViewById(R.id.btnExportCheckpoint)
        btnTestImage = findViewById(R.id.btnTestImage)
        cardTrainingStatus = findViewById(R.id.cardTrainingStatus)
        progressTraining = findViewById(R.id.progressTraining)
        layoutInferenceResult = findViewById(R.id.layoutInferenceResult)
        viewInferenceIndicator = findViewById(R.id.viewInferenceIndicator)
        tvInferenceLabel = findViewById(R.id.tvInferenceLabel)
        tvInferenceConfidence = findViewById(R.id.tvInferenceConfidence)

        // v2.0: History log
        tvHistoryLog = findViewById(R.id.tvHistoryLog)
        scrollHistoryLog = findViewById(R.id.scrollHistoryLog)
    }

    private fun setupChart() {
        chartFitness.apply {
            description.isEnabled = false
            setDrawGridBackground(false)
            setDrawBorders(false)
            isAutoScaleMinMaxEnabled = true

            xAxis.apply {
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(false)
                textColor = ContextCompat.getColor(this@MainActivity, R.color.gray_500)
                textSize = 10f
            }

            axisLeft.apply {
                setDrawGridLines(true)
                gridColor = ContextCompat.getColor(this@MainActivity, R.color.gray_200)
                textColor = ContextCompat.getColor(this@MainActivity, R.color.gray_500)
                textSize = 10f
                axisMinimum = 0f
                axisMaximum = 100f
            }

            axisRight.isEnabled = false
            legend.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)
        }
    }

    private fun setupRecyclerView() {
        rvBots.apply {
            layoutManager = GridLayoutManager(this@MainActivity, 2)
            adapter = botAdapter
            isNestedScrollingEnabled = false
        }
    }

    private fun setupListeners() {
        btnImportZip.setOnClickListener { checkPermissionsAndPickZip() }

        btnStartTraining.setOnClickListener {
            if (viewModel.isTrainingLive.value == true) {
                viewModel.stopTraining()
                btnStartTraining.text = getString(R.string.start_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
                cardTrainingStatus.visibility = View.GONE
            } else {
                viewModel.startTraining()
                btnStartTraining.text = getString(R.string.stop_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_pause)
                cardTrainingStatus.visibility = View.VISIBLE
            }
        }

        btnResetStorage.setOnClickListener {
            viewModel.resetStorage()
            btnStartTraining.text = getString(R.string.start_training)
            btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
            cardTrainingStatus.visibility = View.GONE
            layoutInferenceResult.visibility = View.GONE
            Toast.makeText(this, "Storage reset", Toast.LENGTH_SHORT).show()
        }

        btnSaveModel.setOnClickListener { viewModel.exportModel() }
        btnLoadModel.setOnClickListener { openModelPicker() }
        btnExportCheckpoint.setOnClickListener { viewModel.exportCheckpoint() }

        btnTestImage.setOnClickListener { openImagePicker() }

        sliderMutationRate.addOnChangeListener { _, value, _ ->
            tvMutationRate.text = String.format("%.2f", value)
            viewModel.setMutationRate(value)
        }

        sliderTargetAccuracy.addOnChangeListener { _, value, _ ->
            tvTargetAccuracy.text = String.format("%.0f%%", value)
            tvTargetIndicator.text = String.format("%.0f%%", value)
            viewModel.setTargetAccuracy(value)
        }
    }

    private fun observeViewModel() {
        // v2.0: Prominent generation dashboard
        viewModel.generation.observe(this) { gen ->
            tvGenerationBig.text = gen.toString()
        }

        // v2.0: Stagnant generations counter
        viewModel.stagnantGenerations.observe(this) { stagnant ->
            tvStagnant.text = stagnant.toString()
            // Highlight stagnant counter with warning color when high
            if (stagnant > 10) {
                tvStagnant.setTextColor(ContextCompat.getColor(this, R.color.error_red))
            } else if (stagnant > 5) {
                tvStagnant.setTextColor(ContextCompat.getColor(this, R.color.warning_amber))
            } else {
                tvStagnant.setTextColor(ContextCompat.getColor(this, R.color.warning_amber))
            }
        }

        viewModel.bestAccuracy.observe(this) { acc ->
            tvAccuracy.text = String.format("%.1f%%", acc)
            progressTraining.progress = acc.toInt()
        }

        viewModel.avgFitness.observe(this) { fitness -> tvFitness.text = String.format("%.2f", fitness) }

        viewModel.bots.observe(this) { bots -> botAdapter.updateBots(bots) }

        viewModel.datasetInfo.observe(this) { info ->
            if (info != null) {
                tvDatasetStatus.text = String.format("Dataset: %d samples", info.totalSamples)
                tvLikesCount.apply {
                    text = String.format("Like: %d", info.likeCount)
                    visibility = View.VISIBLE
                }
                tvNonlikesCount.apply {
                    text = String.format("Non-like: %d", info.nonlikeCount)
                    visibility = View.VISIBLE
                }
            } else {
                tvDatasetStatus.text = getString(R.string.no_dataset)
                tvLikesCount.visibility = View.GONE
                tvNonlikesCount.visibility = View.GONE
            }
        }

        viewModel.isTrainingLive.observe(this) { training ->
            btnImportZip.isEnabled = !training
            btnResetStorage.isEnabled = !training
            btnSaveModel.isEnabled = !training
            btnLoadModel.isEnabled = !training
            btnExportCheckpoint.isEnabled = !training
            if (training) tvTrainingStatus.text = "Training in progress\u2026"
            else tvTrainingStatus.text = "Training paused"
        }

        viewModel.trainingComplete.observe(this) { complete ->
            if (complete) {
                tvTrainingStatus.text = getString(R.string.target_reached)
                tvTrainingStatus.setTextColor(ContextCompat.getColor(this, R.color.emerald_success))
                cardTrainingStatus.visibility = View.VISIBLE
                btnStartTraining.text = getString(R.string.start_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
                Toast.makeText(this, getString(R.string.target_reached), Toast.LENGTH_LONG).show()
            }
        }

        viewModel.fitnessHistory.observe(this) { history -> updateChart(history) }

        // v2.0: History Log observer
        viewModel.historyLog.observe(this) { logText ->
            tvHistoryLog.text = logText.ifEmpty { getString(R.string.history_log_empty) }
            // Auto-scroll to bottom
            scrollHistoryLog.post {
                scrollHistoryLog.fullScroll(ScrollView.FOCUS_DOWN)
            }
        }

        viewModel.inferenceResult.observe(this) { result ->
            result?.let {
                layoutInferenceResult.visibility = View.VISIBLE
                tvInferenceLabel.text = it.label

                val isLike = it.label == "Like"
                val color = if (isLike) ContextCompat.getColor(this, R.color.emerald_success)
                            else ContextCompat.getColor(this, R.color.error_red)

                tvInferenceLabel.setTextColor(color)
                tvInferenceConfidence.text = String.format("%.1f%% confidence", it.confidence)

                val bg = viewInferenceIndicator.background as? GradientDrawable
                bg?.setColor(color)
            }
        }

        viewModel.exportStatus.observe(this) { msg ->
            msg?.let { Toast.makeText(this, it, Toast.LENGTH_SHORT).show() }
        }

        viewModel.importStatus.observe(this) { msg ->
            msg?.let { Toast.makeText(this, it, Toast.LENGTH_SHORT).show() }
        }

        viewModel.error.observe(this) { err ->
            err?.let { Toast.makeText(this, it, Toast.LENGTH_LONG).show() }
        }
    }

    private fun updateChart(history: List<Pair<Int, Float>>) {
        if (history.isEmpty()) return
        val entries = history.map { Entry(it.first.toFloat(), it.second) }

        val dataSet = LineDataSet(entries, "Best Accuracy").apply {
            color = ContextCompat.getColor(this@MainActivity, R.color.slate_blue_training)
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 2.5f
            mode = LineDataSet.Mode.CUBIC_BEZIER
            setDrawFilled(true)
            fillColor = ContextCompat.getColor(this@MainActivity, R.color.slate_blue_training)
            fillAlpha = 30
        }

        chartFitness.data = LineData(dataSet)
        chartFitness.invalidate()
    }

    private fun checkPermissionsAndPickZip() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            openZipPicker()
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED
            ) {
                openZipPicker()
            } else {
                permissionLauncher.launch(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE))
            }
        }
    }

    private fun openZipPicker() {
        try { zipPickerLauncher.launch("application/zip") }
        catch (e: Exception) {
            try { zipPickerLauncher.launch("*/*") }
            catch (_: Exception) { Toast.makeText(this, "Cannot open file picker", Toast.LENGTH_SHORT).show() }
        }
    }

    private fun openImagePicker() {
        try { imagePickerLauncher.launch("image/*") }
        catch (_: Exception) { Toast.makeText(this, "Cannot open image picker", Toast.LENGTH_SHORT).show() }
    }

    private fun openModelPicker() {
        try { modelPickerLauncher.launch("*/*") }
        catch (_: Exception) { Toast.makeText(this, "Cannot open file picker", Toast.LENGTH_SHORT).show() }
    }
}
