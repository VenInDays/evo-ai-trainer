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
}
