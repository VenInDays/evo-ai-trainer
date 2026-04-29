package com.evoai.trainer.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for storing BOT brain data (weights/models).
 */
@Entity(tableName = "bots")
data class BotEntity(
    @PrimaryKey val id: Int,
    val name: String,
    val serializedNetwork: String,
    val fitness: Float,
    val accuracy: Float,
    val generation: Int,
    val isBest: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

/**
 * Room entity for tracking training history across generations.
 */
@Entity(tableName = "training_history")
data class TrainingHistoryEntity(
    @PrimaryKey(autoGenerate = true) val uid: Int = 0,
    val generation: Int,
    val bestAccuracy: Float,
    val avgFitness: Float,
    val mutationRate: Float,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Room entity for dataset metadata.
 */
@Entity(tableName = "dataset_meta")
data class DatasetMetaEntity(
    @PrimaryKey val uid: Int = 1,
    val fileName: String,
    val likeCount: Int,
    val nonlikeCount: Int,
    val totalSamples: Int,
    val featureSize: Int,
    val loadedAt: Long = System.currentTimeMillis()
)
