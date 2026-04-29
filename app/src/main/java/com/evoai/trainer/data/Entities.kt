package com.evoai.trainer.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for storing BOT brain data (weights/models).
 * V4: Added loss and valAccuracy fields.
 */
@Entity(tableName = "bots")
data class BotEntity(
    @PrimaryKey val id: Int,
    val name: String,
    val serializedNetwork: String,
    val fitness: Float,
    val accuracy: Float,
    val valAccuracy: Float = 0f,     // V4: Validation accuracy
    val loss: Float = 0f,            // V4: BCE loss
    val generation: Int,
    val isBest: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

/**
 * Room entity for tracking training history across generations.
 * V4: Added valAccuracy and avgLoss fields.
 */
@Entity(tableName = "training_history")
data class TrainingHistoryEntity(
    @PrimaryKey(autoGenerate = true) val uid: Int = 0,
    val generation: Int,
    val bestAccuracy: Float,
    val bestValAccuracy: Float = 0f,  // V4: Best validation accuracy
    val avgFitness: Float,
    val avgLoss: Float = 0f,          // V4: Average loss
    val mutationRate: Float,
    val decayingMutationRate: Float = 0f,  // V4: Decayed mutation rate
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Room entity for dataset metadata.
 * V4: Added train/val split info.
 */
@Entity(tableName = "dataset_meta")
data class DatasetMetaEntity(
    @PrimaryKey val uid: Int = 1,
    val fileName: String,
    val likeCount: Int,
    val nonlikeCount: Int,
    val totalSamples: Int,
    val trainSamples: Int = 0,       // V4: Training set size
    val valSamples: Int = 0,         // V4: Validation set size
    val featureSize: Int,
    val loadedAt: Long = System.currentTimeMillis()
)

/**
 * Room entity for auto-saved best model checkpoints.
 * V4: Added loss field.
 */
@Entity(tableName = "best_model_checkpoints")
data class BestModelCheckpointEntity(
    @PrimaryKey(autoGenerate = true) val uid: Int = 0,
    val generation: Int,
    val serializedNetwork: String,
    val accuracy: Float,
    val valAccuracy: Float = 0f,     // V4
    val fitness: Float,
    val loss: Float = 0f,            // V4
    val mutationRate: Float,
    val decayingMutationRate: Float = 0f,  // V4
    val targetAccuracy: Float,
    val savedAt: Long = System.currentTimeMillis()
)

/**
 * V4: Hard examples table for Manual Override feature.
 */
@Entity(tableName = "hard_examples")
data class HardExampleEntity(
    @PrimaryKey(autoGenerate = true) val uid: Int = 0,
    val serializedFeatures: String,  // Comma-separated float values
    val label: Int,                  // 0 = non-like, 1 = like
    val source: String = "manual",   // "manual" override
    val addedAt: Long = System.currentTimeMillis()
)
