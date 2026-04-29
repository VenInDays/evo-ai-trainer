package com.evoai.trainer

import android.app.Application
import com.evoai.trainer.data.AppDatabase

class App : Application() {
    val database: AppDatabase by lazy { AppDatabase.getInstance(this) }
}
