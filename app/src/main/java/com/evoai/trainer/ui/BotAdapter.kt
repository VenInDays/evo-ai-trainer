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
import com.evoai.trainer.ga.BotLineage
import com.evoai.trainer.ga.BotStatus
import com.google.android.material.card.MaterialCardView

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
        holder.bind(bots[position])
    }

    override fun getItemCount(): Int = bots.size

    class BotViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val cardBot: MaterialCardView = itemView.findViewById(R.id.cardBot)
        private val tvBotName: TextView = itemView.findViewById(R.id.tvBotName)
        private val tvBotFitness: TextView = itemView.findViewById(R.id.tvBotFitness)
        private val tvBotStatus: TextView = itemView.findViewById(R.id.tvBotStatus)
        private val tvLineageBadge: TextView = itemView.findViewById(R.id.tvLineageBadge)
        private val tvMutationVariance: TextView = itemView.findViewById(R.id.tvMutationVariance)
        private val progressBotFitness: ProgressBar = itemView.findViewById(R.id.progressBotFitness)
        private val viewBotIndicator: View = itemView.findViewById(R.id.viewBotIndicator)

        fun bind(bot: Bot) {
            tvBotName.text = String.format("BOT-%02d", bot.id + 1)
            tvBotFitness.text = String.format("%.2f", bot.fitness)

            val progressValue = (bot.fitness * 100).toInt().coerceIn(0, 100)
            progressBotFitness.progress = progressValue

            // ===== Lineage Badge & Card Styling =====
            setupLineageDisplay(bot)

            // ===== Status Styling =====
            when (bot.status) {
                BotStatus.READY -> {
                    tvBotStatus.text = "Ready"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_500))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.gray_400))
                    progressBotFitness.progressDrawable?.setTint(getLineageProgressColor(bot))
                }
                BotStatus.TRAINING -> {
                    tvBotStatus.text = "Thinking\u2026"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    progressBotFitness.progressDrawable?.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                }
                BotStatus.EVALUATED -> {
                    tvBotStatus.text = String.format("Acc: %.1f%%", bot.accuracy)
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_600))
                    viewBotIndicator.background.setTint(getLineageIndicatorColor(bot))
                    progressBotFitness.progressDrawable?.setTint(getLineageProgressColor(bot))
                }
                BotStatus.BEST -> {
                    tvBotStatus.text = String.format("\u2605 Best %.1f%%", bot.accuracy)
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

        private fun setupLineageDisplay(bot: Bot) {
            when (bot.lineage) {
                BotLineage.ELITE_PARENT -> {
                    // Gold/Amber border + Crown badge
                    tvLineageBadge.text = "\uD83D\uDC51 Elite"
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_elite)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Gold card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.elite_gold)
                    cardBot.strokeWidth = 3

                    // Mutation variance hidden for elites
                    tvMutationVariance.visibility = View.GONE
                }
                BotLineage.MUTATED_CLONE -> {
                    // Blue badge + mutation variance display
                    val parentLabel = if (bot.parentRank == 1) "E1" else "E2"
                    tvLineageBadge.text = "\uD83E\uDDEC Clone"
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_clone)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Blue card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.slate_blue_training)
                    cardBot.strokeWidth = 2

                    // Show mutation variance
                    if (bot.mutationVariance > 0f) {
                        tvMutationVariance.text = String.format("Muta: %.2f  |  From: %s", bot.mutationVariance, parentLabel)
                        tvMutationVariance.visibility = View.VISIBLE
                    } else {
                        tvMutationVariance.visibility = View.GONE
                    }
                }
                BotLineage.RESET_RANDOM -> {
                    // Amber/Warning badge
                    tvLineageBadge.text = "\u26A0\uFE0F Reset"
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_reset)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Amber card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.warning_amber)
                    cardBot.strokeWidth = 2

                    // Mutation variance hidden for reset models
                    tvMutationVariance.visibility = View.GONE
                }
            }
        }

        private fun getLineageIndicatorColor(bot: Bot): Int {
            return when (bot.lineage) {
                BotLineage.ELITE_PARENT -> ContextCompat.getColor(itemView.context, R.color.elite_gold)
                BotLineage.MUTATED_CLONE -> ContextCompat.getColor(itemView.context, R.color.slate_blue_training)
                BotLineage.RESET_RANDOM -> ContextCompat.getColor(itemView.context, R.color.warning_amber)
            }
        }

        private fun getLineageProgressColor(bot: Bot): Int {
            return when (bot.lineage) {
                BotLineage.ELITE_PARENT -> ContextCompat.getColor(itemView.context, R.color.elite_gold)
                BotLineage.MUTATED_CLONE -> ContextCompat.getColor(itemView.context, R.color.slate_blue_training)
                BotLineage.RESET_RANDOM -> ContextCompat.getColor(itemView.context, R.color.warning_amber)
            }
        }
    }
}
