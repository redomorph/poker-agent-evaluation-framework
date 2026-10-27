# Project Scope, Motivation, and Research Questions

## Abstract

This document describes the conception of a comprehensive poker agent evaluation framework. In contrast to classic bot development, which primarily aims at achieving the highest possible win rates, the project defines a **multidimensional notion of quality** consisting of population performance, robustness, and game-theoretic quality. The goal is to develop a reproducible, variance-reduced benchmarking methodology for agents in imperfect, stochastic games.

The project transfers experience from solving perfect-information games, particularly the development of a Mancala minimax engine, to the domain of imperfect information games. The central challenges – hidden information, stochasticity, continuous bet-sizing spaces, and the practical computability of game-theoretic equilibrium properties – are addressed through an explicitly defined game abstraction, controlled opponent populations, and reproducible experimental designs.

The framework explicitly distinguishes between three evaluation dimensions:

1. **Population Performance:** expected value against a defined population of opponents.
2. **Robustness:** stability of performance across different, plausible opponent parameters.
3. **Game-Theoretic Quality:** exploitability or distance to game value in games where such computation is actually feasible.

For abstracted NLHE, exploitability is not directly computed. Instead, we investigate whether an agent's Leduc exploitability correlates with its performance in abstracted NLHE – an empirical test of whether game-theoretic properties from small games transfer to larger domains.

The CLI application to be developed is primarily understood as a scientific testbed for strategies and, secondarily, as a playing bot. The first milestone is therefore not a powerful poker agent, but a **validated, reproducible, and statistically defensible evaluation infrastructure**.

---

## 1. Introduction & Motivation

The transition from deterministic or perfect-information games to poker motivates a different evaluation methodology. In idealised games such as Mancala, mathematical claims about outcomes under perfect play can sometimes be established.

Such certainty does not exist in Hold'em poker.

Poker is an **imperfect, stochastic zero-sum game** with hidden information, random cards, and high variance. A simple win rate over a limited number of hands can therefore paint a misleading picture of an agent's strategic quality.

The project is therefore understood not primarily as *bot development*, but as **methodology development for reproducible agent evaluation**.

The central research question is:

> **How can the quality of a poker agent be evaluated reproducibly and multidimensionally in an imperfect, stochastic environment, without reducing the evaluation to a single, highly variable win rate?**

Further research questions arise:

- How strongly does the choice of opponent population influence the measured performance?
- How robust is an agent to changes in opponent parameters?
- How well does a game-theoretic metric from a small, solvable game correlate with practical performance in abstracted NLHE?
- What effects does the bet-sizing abstraction have on the validity of the evaluation?
- How large must the sample be for a measured performance difference to be statistically reliable?
- How much can online opponent modeling improve performance against unknown opponents?
- How well do the actually observed behavioural frequencies of parametric opponents match their specified target parameters?

---
