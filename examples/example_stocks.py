#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de la stratégie de trading long-short sur les actions mentionnées
dans le document original.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Ajout du dossier parent au chemin pour pouvoir importer le module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum_strategy import MomentumStrategy


def run_example():
    # Liste des actifs mentionnés dans le document
    assets = ['BA', 'PEP', 'LMT', 'AAPL', 'JNJ', 'JPM', 'BAC', 'LLY', 'CAR']
    
    # Création de l'instance de la stratégie
    strategy = MomentumStrategy(assets)
    
    # Période mentionnée dans le document
    start_date = '2024-12-01'
    end_date = '2025-04-30'
    
    # Exécution du backtest avec les paramètres du document
    results = strategy.backtest(
        start_date=start_date,
        end_date=end_date,
        momentum_period=30,
        forward_return_days=5,
        quantiles=(0.2, 0.8)
    )
    
    # Affichage des métriques de performance
    print("\nMétriques de performance:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Visualisation des corrélations
    correlations = strategy.calculate_correlations()
    print("\nCorrélations entre momentum et rendements futurs:")
    print(correlations)
    
    # Visualisation graphique
    fig1 = strategy.plot_correlations(correlations)
    fig1.savefig('correlations.png')
    
    # Visualisation de la performance
    fig2 = strategy.plot_performance()
    fig2.savefig('performance.png')
    
    # Affichage des positions longues et courtes à la fin de la période
    print("\nPositions longues à la dernière date:")
    long_stocks = results['long_portfolio'].iloc[-1]
    print(long_stocks[long_stocks].index.tolist())
    
    print("\nPositions courtes à la dernière date:")
    short_stocks = results['short_portfolio'].iloc[-1]
    print(short_stocks[short_stocks].index.tolist())
    
    # Calcul et affichage des rendements moyens par position
    long_returns = results['returns'][results['long_portfolio']].mean().mean()
    short_returns = results['returns'][results['short_portfolio']].mean().mean()
    
    print(f"\nRendement moyen des positions longues: {long_returns:.4f}")
    print(f"Rendement moyen des positions courtes: {short_returns:.4f}")
    print(f"Écart de rendement: {long_returns - short_returns:.4f}")
    
    plt.show()


if __name__ == "__main__":
    run_example()
