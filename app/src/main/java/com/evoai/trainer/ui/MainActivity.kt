package com.evoai.trainer.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.ProgressBar
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

    // Views
    private lateinit var tvGeneration: TextView
    private lateinit var tvAccuracy: TextView
    private lateinit var tvFitness: TextView
    private lateinit var tvDatasetStatus: TextView
    private lateinit var tvLikesCount: TextView
    private lateinit var tvNonlikesCount: TextView
    private lateinit var tvMutationRate: TextView
    private lateinit var tvTrainingStatus: TextView
    private lateinit var sliderMutationRate: Slider
    private lateinit var chartFitness: LineChart
    private lateinit var rvBots: RecyclerView
    private lateinit var btnImportZip: MaterialButton
    private lateinit var btnStartTraining: MaterialButton
    private lateinit var btnResetStorage: MaterialButton
    private lateinit var cardTrainingStatus: MaterialCardView
    private lateinit var progressTraining: ProgressBar

    private val botAdapter = BotAdapter()

    // ZIP file picker
    private val zipPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.loadDataset(it)
        }
    }

    // Permission launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            openZipPicker()
        } else {
            Toast.makeText(this, "Storage permission required", Toast.LENGTH_SHORT).show()
        }
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
        tvGeneration = findViewById(R.id.tvGeneration)
        tvAccuracy = findViewById(R.id.tvAccuracy)
        tvFitness = findViewById(R.id.tvFitness)
        tvDatasetStatus = findViewById(R.id.tvDatasetStatus)
        tvLikesCount = findViewById(R.id.tvLikesCount)
        tvNonlikesCount = findViewById(R.id.tvNonlikesCount)
        tvMutationRate = findViewById(R.id.tvMutationRate)
        tvTrainingStatus = findViewById(R.id.tvTrainingStatus)
        sliderMutationRate = findViewById(R.id.sliderMutationRate)
        chartFitness = findViewById(R.id.chartFitness)
        rvBots = findViewById(R.id.rvBots)
        btnImportZip = findViewById(R.id.btnImportZip)
        btnStartTraining = findViewById(R.id.btnStartTraining)
        btnResetStorage = findViewById(R.id.btnResetStorage)
        cardTrainingStatus = findViewById(R.id.cardTrainingStatus)
        progressTraining = findViewById(R.id.progressTraining)
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
        btnImportZip.setOnClickListener {
            checkPermissionsAndPickZip()
        }

        btnStartTraining.setOnClickListener {
            if (viewModel.isTrainingLive.value == true) {
                viewModel.stopTraining()
                btnStartTraining.text = getString(R.string.start_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
                cardTrainingStatus.visibility = android.view.View.GONE
            } else {
                viewModel.startTraining()
                btnStartTraining.text = getString(R.string.stop_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_pause)
                cardTrainingStatus.visibility = android.view.View.VISIBLE
            }
        }

        btnResetStorage.setOnClickListener {
            viewModel.resetStorage()
            btnStartTraining.text = getString(R.string.start_training)
            btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
            cardTrainingStatus.visibility = android.view.View.GONE
            Toast.makeText(this, "Storage reset", Toast.LENGTH_SHORT).show()
        }

        sliderMutationRate.addOnChangeListener { _, value, _ ->
            tvMutationRate.text = String.format("%.2f", value)
            viewModel.setMutationRate(value)
        }
    }

    private fun observeViewModel() {
        viewModel.generation.observe(this) { gen ->
            tvGeneration.text = gen.toString()
        }

        viewModel.bestAccuracy.observe(this) { acc ->
            tvAccuracy.text = String.format("%.1f%%", acc)
            progressTraining.progress = acc.toInt()
        }

        viewModel.avgFitness.observe(this) { fitness ->
            tvFitness.text = String.format("%.2f", fitness)
        }

        viewModel.bots.observe(this) { bots ->
            botAdapter.updateBots(bots)
        }

        viewModel.datasetInfo.observe(this) { info ->
            if (info != null) {
                tvDatasetStatus.text = String.format("Dataset loaded: %d samples", info.totalSamples)
                tvLikesCount.apply {
                    text = String.format("Like: %d", info.likeCount)
                    visibility = android.view.View.VISIBLE
                }
                tvNonlikesCount.apply {
                    text = String.format("Non-like: %d", info.nonlikeCount)
                    visibility = android.view.View.VISIBLE
                }
            } else {
                tvDatasetStatus.text = getString(R.string.no_dataset)
                tvLikesCount.visibility = android.view.View.GONE
                tvNonlikesCount.visibility = android.view.View.GONE
            }
        }

        viewModel.isTrainingLive.observe(this) { training ->
            if (training) {
                tvTrainingStatus.text = "Training in progress…"
                btnImportZip.isEnabled = false
                btnResetStorage.isEnabled = false
            } else {
                tvTrainingStatus.text = "Training paused"
                btnImportZip.isEnabled = true
                btnResetStorage.isEnabled = true
            }
        }

        viewModel.trainingComplete.observe(this) { complete ->
            if (complete) {
                tvTrainingStatus.text = getString(R.string.target_reached)
                tvTrainingStatus.setTextColor(
                    ContextCompat.getColor(this, R.color.emerald_success)
                )
                cardTrainingStatus.visibility = android.view.View.VISIBLE
                btnStartTraining.text = getString(R.string.start_training)
                btnStartTraining.setIconResource(android.R.drawable.ic_media_play)
                Toast.makeText(this, getString(R.string.target_reached), Toast.LENGTH_LONG).show()
            }
        }

        viewModel.fitnessHistory.observe(this) { history ->
            updateChart(history)
        }

        viewModel.error.observe(this) { err ->
            err?.let {
                Toast.makeText(this, it, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateChart(history: List<Pair<Int, Float>>) {
        if (history.isEmpty()) return

        val entries = history.mapIndexed { index, pair ->
            Entry(pair.first.toFloat(), pair.second)
        }

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
            // Android 13+ doesn't need READ_EXTERNAL_STORAGE for content picker
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
        try {
            zipPickerLauncher.launch("application/zip")
        } catch (e: Exception) {
            // Fallback to any file type
            try {
                zipPickerLauncher.launch("*/*")
            } catch (e2: Exception) {
                Toast.makeText(this, "Cannot open file picker", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
