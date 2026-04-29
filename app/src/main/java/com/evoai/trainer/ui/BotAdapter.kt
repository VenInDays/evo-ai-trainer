package com.evoai.trainer.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.evoai.trainer.R
import com.evoai.trainer.ga.Bot
import com.evoai.trainer.ga.BotStatus

class BotAdapter : RecyclerView.Adapter<BotAdapter.BotViewHolder>() {

    private var bots: List<Bot> = emptyList()

    fun updateBots(newBots: List<Bot>) {
        bots = newBots
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BotViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_bot, parent, false)
        return BotViewHolder(view)
    }

    override fun onBindViewHolder(holder: BotViewHolder, position: Int) {
        val bot = bots[position]
        holder.bind(bot)
    }

    override fun getItemCount(): Int = bots.size

    class BotViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvBotName: TextView = itemView.findViewById(R.id.tvBotName)
        private val tvBotFitness: TextView = itemView.findViewById(R.id.tvBotFitness)
        private val tvBotStatus: TextView = itemView.findViewById(R.id.tvBotStatus)
        private val progressBotFitness: ProgressBar = itemView.findViewById(R.id.progressBotFitness)
        private val viewBotIndicator: View = itemView.findViewById(R.id.viewBotIndicator)

        fun bind(bot: Bot) {
            tvBotName.text = String.format("BOT-%02d", bot.id + 1)
            tvBotFitness.text = String.format("%.2f", bot.fitness)

            val progressValue = (bot.fitness * 100).toInt().coerceIn(0, 100)
            progressBotFitness.progress = progressValue

            when (bot.status) {
                BotStatus.READY -> {
                    tvBotStatus.text = "Ready"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_500))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.gray_400))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                }
                BotStatus.TRAINING -> {
                    tvBotStatus.text = "Thinking…"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                }
                BotStatus.EVALUATED -> {
                    tvBotStatus.text = String.format("Acc: %.1f%%", bot.accuracy)
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_600))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                }
                BotStatus.BEST -> {
                    tvBotStatus.text = String.format("★ Best %.1f%%", bot.accuracy)
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.emerald_success))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.emerald_success))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.emerald_success))
                }
                BotStatus.ELIMINATED -> {
                    tvBotStatus.text = "Eliminated"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_400))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.gray_300))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.gray_300))
                }
            }
        }
    }
}
