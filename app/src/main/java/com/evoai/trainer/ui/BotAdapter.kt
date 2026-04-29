package com.evoai.trainer.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.evoai.trainer.R
import com.evoai.trainer.ga.Bot
import com.evoai.trainer.ga.BotLineage
import com.evoai.trainer.ga.BotStatus
import com.evoai.trainer.ui.widget.SparklineView
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
        private val tvBotAccuracy: TextView = itemView.findViewById(R.id.tvBotAccuracy)
        private val tvBotStatus: TextView = itemView.findViewById(R.id.tvBotStatus)
        private val tvLineageBadge: TextView = itemView.findViewById(R.id.tvLineageBadge)
        private val tvMutationVariance: TextView = itemView.findViewById(R.id.tvMutationVariance)
        private val sparklineFitness: SparklineView = itemView.findViewById(R.id.sparklineFitness)
        private val viewBotIndicator: View = itemView.findViewById(R.id.viewBotIndicator)

        fun bind(bot: Bot) {
            // V3: Display name = "Model [familyId] Gen [generationBorn]"
            tvBotName.text = bot.displayName

            // V3: Show accuracy prominently
            tvBotAccuracy.text = String.format("%.1f%%", bot.accuracy)

            // V3: Sparkline chart with last 5 fitness scores
            val sparklineData = bot.getSparklineData()
            sparklineFitness.setData(sparklineData)
            sparklineFitness.setLineColor(getLineageColor(bot))

            // ===== Lineage Badge & Card Styling =====
            setupLineageDisplay(bot)

            // ===== Status Styling =====
            when (bot.status) {
                BotStatus.READY -> {
                    tvBotStatus.text = "Ready"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_500))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.gray_400))
                }
                BotStatus.TRAINING -> {
                    tvBotStatus.text = "Training\u2026"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.slate_blue_training))
                }
                BotStatus.EVALUATED -> {
                    tvBotStatus.text = String.format("Acc: %.1f%%", bot.accuracy)
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_600))
                    viewBotIndicator.background.setTint(getLineageColor(bot))
                }
                BotStatus.BEST -> {
                    tvBotStatus.text = String.format("Champion %.1f%%", bot.accuracy)
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.emerald_success))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.emerald_success))
                }
                BotStatus.ELIMINATED -> {
                    tvBotStatus.text = "Purged"
                    tvBotStatus.setTextColor(ContextCompat.getColor(itemView.context, R.color.gray_400))
                    viewBotIndicator.background.setTint(ContextCompat.getColor(itemView.context, R.color.gray_300))
                }
            }
        }

        private fun setupLineageDisplay(bot: Bot) {
            when (bot.lineage) {
                BotLineage.LEGACY -> {
                    // V3: Gold badge + "LEGACY" label
                    tvLineageBadge.text = "LEGACY"
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_elite)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Gold card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.elite_gold)
                    cardBot.strokeWidth = 3

                    // Mutation variance hidden for legacies
                    tvMutationVariance.visibility = View.GONE
                }
                BotLineage.CLONE -> {
                    // V3: Blue badge + "GEN [X] CLONE" label
                    tvLineageBadge.text = String.format("GEN %d CLONE", bot.generationBorn)
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_clone)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Blue card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.slate_blue_training)
                    cardBot.strokeWidth = 2

                    // Show mutation variance
                    if (bot.mutationVariance > 0f) {
                        val parentLabel = if (bot.parentRank == 1) "A" else "B"
                        tvMutationVariance.text = String.format("\u03C3%.2f \u2190 M%s", bot.mutationVariance, parentLabel)
                        tvMutationVariance.visibility = View.VISIBLE
                    } else {
                        tvMutationVariance.visibility = View.GONE
                    }
                }
                BotLineage.RESET_RANDOM -> {
                    // Amber badge + "RESET" label
                    tvLineageBadge.text = "RESET"
                    tvLineageBadge.setBackgroundResource(R.drawable.badge_reset)
                    tvLineageBadge.setTextColor(ContextCompat.getColor(itemView.context, R.color.white_pure))
                    tvLineageBadge.visibility = View.VISIBLE

                    // Amber card border
                    cardBot.strokeColor = ContextCompat.getColor(itemView.context, R.color.warning_amber)
                    cardBot.strokeWidth = 2

                    tvMutationVariance.visibility = View.GONE
                }
            }
        }

        private fun getLineageColor(bot: Bot): Int {
            return when (bot.lineage) {
                BotLineage.LEGACY -> ContextCompat.getColor(itemView.context, R.color.elite_gold)
                BotLineage.CLONE -> ContextCompat.getColor(itemView.context, R.color.slate_blue_training)
                BotLineage.RESET_RANDOM -> ContextCompat.getColor(itemView.context, R.color.warning_amber)
            }
        }
    }
}
