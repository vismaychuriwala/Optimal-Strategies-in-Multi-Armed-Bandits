# Optimal Stategies to score runs in cricket using Multi-Armed Bandits

This repository contains several implementations of multi-armed bandit (MAB) agents applied to a simulated cricket batting scenario. The simulation models a cricket innings where an agent (the batsman) selects among different shot strategies (arms) with the goal of maximizing runs while minimizing the risk of getting out.

## Overview

In this project, you will find **four distinct agent types** implemented as part of our exploration of MAB strategies:

1. **KL-UCB Survival Agent**
   - **Description:** Uses a KL-divergence based Upper Confidence Bound (UCB) method. The reward is based on survival (i.e., `1 - wicket`), focusing on minimizing dismissals.
  
2. **Reward-UCB Agents**
   - **Variant 1: Reward-UCB (KL) Agent**
     - **Description:** Computes rewards using an efficiency metric `(1 - p(out)) * avg_runs` and applies a KL-UCB approach.
   - **Variant 2: Reward-UCB (Simple) Agent**
     - **Description:** A simpler variant that computes the reward as `runs / 6`.
   - **Variant 3: UCB1 Agent**
     - **Description:** Implements the classic UCB1 algorithm based on the average reward plus an exploration bonus.

3. **Risk-Adjusted Successive Elimination Agent**
   - **Description:** Uses a more sophisticated approach by computing a risk-adjusted reward (ratio of expected reward to risk) and eliminates arms that perform poorly. This agent progressively removes suboptimal strategies using confidence bounds.
