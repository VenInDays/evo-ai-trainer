package com.evoai.trainer.data

import androidx.room.*

@Dao
interface BotDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertBots(bots: List<BotEntity>)

    @Query("SELECT * FROM bots ORDER BY fitness DESC")
    suspend fun getAllBots(): List<BotEntity>

    @Query("SELECT * FROM bots WHERE isBest = 1 LIMIT 1")
    suspend fun getBestBot(): BotEntity?

    @Query("DELETE FROM bots")
    suspend fun deleteAllBots()

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertHistory(history: TrainingHistoryEntity)

    @Query("SELECT * FROM training_history ORDER BY generation ASC")
    suspend fun getTrainingHistory(): List<TrainingHistoryEntity>

    @Query("DELETE FROM training_history")
    suspend fun deleteAllHistory()

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertDatasetMeta(meta: DatasetMetaEntity)

    @Query("SELECT * FROM dataset_meta LIMIT 1")
    suspend fun getDatasetMeta(): DatasetMetaEntity?

    @Query("DELETE FROM dataset_meta")
    suspend fun deleteDatasetMeta()

    // Best model checkpoint operations
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCheckpoint(checkpoint: BestModelCheckpointEntity)

    @Query("SELECT * FROM best_model_checkpoints ORDER BY generation DESC LIMIT 1")
    suspend fun getLatestCheckpoint(): BestModelCheckpointEntity?

    @Query("DELETE FROM best_model_checkpoints")
    suspend fun deleteAllCheckpoints()

    // V4: Hard examples operations
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertHardExample(example: HardExampleEntity)

    @Query("SELECT * FROM hard_examples ORDER BY addedAt DESC")
    suspend fun getHardExamples(): List<HardExampleEntity>

    @Query("DELETE FROM hard_examples")
    suspend fun deleteAllHardExamples()
}
