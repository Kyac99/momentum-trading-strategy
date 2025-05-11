#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ce script reproduit les résultats présentés dans le document original.
Il implémente une stratégie long-short basée sur le momentum avec les actions spécifiées.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# Ajout du dossier parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum_strategy import MomentumStrategy
from data_loader import DataLoader
from backtest import Backtest
from visualization import plot_correlation_matrix, plot_position_evolution, plot_momentum_heat_map


def reproduce_document_results():
    """
    Reproduit les résultats de la stratégie présentée dans le document original.
    
    Le document utilise:
    - Les actions: BA, PEP, LMT, AAPL, JNJ, JPM, BAC, LLY, CAR
    - Période: 2024-12-01 à 2025-04-30
    - Momentum sur 30 jours
    - Rendements futurs sur 5 jours
    - Quantiles à 20% et 80%
    """
    print("Reproduction des résultats du document original...")
    
    # Liste des actifs mentionnés dans le document
    assets = ['BA', 'PEP', 'LMT', 'AAPL', 'JNJ', 'JPM', 'BAC', 'LLY', 'CAR']
    
    # Période mentionnée dans le document
    start_date = '2024-12-01'
    end_date = '2025-04-30'
    
    # Paramètres de la stratégie
    momentum_period = 30
    forward_return_days = 5
    quantiles = (0.2, 0.8)
    
    # Création de l'instance de la stratégie
    strategy = MomentumStrategy(assets)
    
    # Téléchargement des données
    strategy.download_data(start_date, end_date)
    
    # Calcul du momentum
    momentum_scores = strategy.calculate_momentum(momentum_period)
    
    # Calcul des rendements futurs
    returns = strategy.calculate_forward_returns(forward_return_days)
    
    # Classement des actifs
    ranks = strategy.rank_assets()
    
    # Calcul des corrélations
    correlations = strategy.calculate_correlations()
    print("\nCorrélations entre momentum et rendements futurs:")
    print(correlations)
    
    # Création du portefeuille long-short
    long_portfolio, short_portfolio = strategy.create_long_short_portfolio(ranks, quantiles)
    
    # Calcul des rendements de la stratégie
    strategy_returns = strategy.calculate_strategy_returns(long_portfolio, short_portfolio)
    
    # Création de la figure pour les corrélations
    plt.figure(figsize=(15, 7))
    plt.bar(range(len(correlations)), correlations['Correlation'])
    plt.xticks(range(len(correlations)), correlations.index, rotation=90)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Corrélations Momentum-Rendements par Action')
    plt.ylabel('Corrélation de Spearman')
    plt.tight_layout()
    plt.savefig('correlations.png')
    
    # Visualisation de l'évolution des positions
    pos_fig = plot_position_evolution(long_portfolio, short_portfolio)
    pos_fig.savefig('positions_evolution.png')
    
    # Visualisation des scores de momentum
    heat_fig = plot_momentum_heat_map(momentum_scores)
    heat_fig.savefig('momentum_heatmap.png')
    
    # Visualisation de la matrice de corrélation des rendements
    corr_fig = plot_correlation_matrix(returns)
    corr_fig.savefig('return_correlations.png')
    
    # Visualisation des rendements cumulatifs
    plt.figure(figsize=(15, 7))
    cumulative_returns = (1 + strategy_returns).cumprod()
    cumulative_returns.plot()
    plt.title('Rendements Cumulatifs de la Stratégie')
    plt.ylabel('Rendements Cumulatifs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png')
    
    # Exécution d'un backtest complet
    backtest = Backtest(strategy, start_date, end_date)
    results = backtest.run(
        momentum_period=momentum_period,
        forward_return_days=forward_return_days,
        quantiles=quantiles,
        transaction_cost=0.001
    )
    
    # Affichage du résumé des performances
    backtest.print_performance_summary()
    
    # Sauvegarde des résultats
    result_dir = backtest.save_results()
    
    # Affichage des positions à la fin de la période
    print("\nPositions longues à la dernière date:")
    long_stocks = long_portfolio.iloc[-1]
    print(long_stocks[long_stocks].index.tolist())
    
    print("\nPositions courtes à la dernière date:")
    short_stocks = short_portfolio.iloc[-1]
    print(short_stocks[short_stocks].index.tolist())
    
    return results


def additional_analysis():
    """
    Effectue des analyses supplémentaires non présentes dans le document original.
    
    - Optimisation des paramètres
    - Test sur différentes périodes
    - Analyse sectorielle
    """
    # Liste des actifs
    assets = ['BA', 'PEP', 'LMT', 'AAPL', 'JNJ', 'JPM', 'BAC', 'LLY', 'CAR']
    
    # Chargeur de données
    loader = DataLoader(cache_dir='data/cache')
    
    print("\n" + "="*50)
    print("ANALYSES SUPPLÉMENTAIRES")
    print("="*50)
    
    # 1. Analyse de sensibilité aux paramètres
    print("\n1. Analyse de sensibilité aux paramètres")
    print("-"*50)
    
    strategy = MomentumStrategy(assets)
    strategy.download_data('2024-12-01', '2025-04-30')
    
    # Test de différentes périodes de momentum
    momentum_periods = [5, 10, 20, 30, 60, 90]
    momentum_results = {}
    
    for period in momentum_periods:
        strategy.calculate_momentum(period)
        strategy.calculate_forward_returns(5)
        ranks = strategy.rank_assets()
        long_portfolio, short_portfolio = strategy.create_long_short_portfolio(ranks)
        returns = strategy.calculate_strategy_returns(long_portfolio, short_portfolio)
        momentum_results[period] = returns
    
    # Comparaison des rendements cumulatifs
    plt.figure(figsize=(15, 7))
    cum_returns = pd.DataFrame({
        f'Momentum {period} jours': (1 + rets).cumprod() for period, rets in momentum_results.items()
    })
    cum_returns.plot()
    plt.title('Rendements Cumulatifs selon la Période de Momentum')
    plt.ylabel('Rendements Cumulatifs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('momentum_period_comparison.png')
    
    # Métriques de performance pour chaque période
    performance_metrics = pd.DataFrame(index=momentum_periods, columns=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'])
    
    for period in momentum_periods:
        returns = momentum_results[period]
        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        win_rate = (returns > 0).mean()
        
        performance_metrics.loc[period] = [total_return, sharpe_ratio, max_drawdown, win_rate]
    
    print("\nMétriques de performance par période de momentum:")
    print(performance_metrics)
    
    # 2. Analyse par actif individuel
    print("\n2. Analyse par actif individuel")
    print("-"*50)
    
    # Calcul des rendements moyens par actif en position longue et courte
    strategy.calculate_momentum(30)
    strategy.calculate_forward_returns(5)
    ranks = strategy.rank_assets()
    long_portfolio, short_portfolio = strategy.create_long_short_portfolio(ranks)
    
    # Rendements moyens par actif
    long_returns = pd.DataFrame()
    short_returns = pd.DataFrame()
    
    for asset in assets:
        # Rendements quand l'actif est en position longue
        asset_long = strategy.returns[asset][long_portfolio[asset]]
        asset_long_mean = asset_long.mean() if not asset_long.empty else np.nan
        
        # Rendements quand l'actif est en position courte
        asset_short = strategy.returns[asset][short_portfolio[asset]]
        asset_short_mean = asset_short.mean() if not asset_short.empty else np.nan
        
        long_returns[asset] = [asset_long_mean]
        short_returns[asset] = [asset_short_mean]
    
    print("\nRendements moyens en position longue:")
    print(long_returns.T)
    
    print("\nRendements moyens en position courte:")
    print(short_returns.T)
    
    # 3. Test sur différents univers d'actions
    print("\n3. Test sur différents univers d'actions")
    print("-"*50)
    
    # Création de différents portfolios
    portfolios = {
        'Original': ['BA', 'PEP', 'LMT', 'AAPL', 'JNJ', 'JPM', 'BAC', 'LLY', 'CAR'],
        'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'CRM', 'INTC'],
        'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN']
    }
    
    portfolio_results = {}
    
    for name, stocks in portfolios.items():
        print(f"\nTest sur le portfolio {name}...")
        strat = MomentumStrategy(stocks)
        try:
            results = strat.backtest('2024-12-01', '2025-04-30', 30, 5)
            portfolio_results[name] = results['strategy_returns']
            
            total_return = results['cumulative_returns'].iloc[-1] - 1
            print(f"Rendement total: {total_return:.2%}")
        except Exception as e:
            print(f"Erreur: {str(e)}")
    
    # Comparaison des rendements cumulatifs
    plt.figure(figsize=(15, 7))
    cum_returns = pd.DataFrame({
        name: (1 + rets).cumprod() for name, rets in portfolio_results.items() if not rets.empty
    })
    cum_returns.plot()
    plt.title('Rendements Cumulatifs par Type de Portfolio')
    plt.ylabel('Rendements Cumulatifs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_comparison.png')
    
    # 4. Analyse de drawdown
    print("\n4. Analyse de drawdown")
    print("-"*50)
    
    # Utilisation des résultats du portfolio original
    if 'Original' in portfolio_results:
        original_returns = portfolio_results['Original']
        cumulative_returns = (1 + original_returns).cumprod()
        drawdowns = 1 - cumulative_returns / cumulative_returns.cummax()
        
        # Identification des 3 pires drawdowns
        sorted_drawdowns = drawdowns.sort_values(ascending=False)
        worst_dates = sorted_drawdowns.head(3).index
        
        print("\nPires périodes de drawdown:")
        for date in worst_dates:
            print(f"{date.strftime('%Y-%m-%d')}: {drawdowns[date]:.2%}")
        
        # Visualisation des drawdowns
        plt.figure(figsize=(15, 7))
        drawdowns.plot()
        plt.title('Évolution des Drawdowns')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('drawdowns.png')
    
    return performance_metrics


if __name__ == "__main__":
    # Reproduction des résultats du document
    print("\n*** REPRODUCTION DES RÉSULTATS DU DOCUMENT ***\n")
    results = reproduce_document_results()
    
    # Analyses supplémentaires
    print("\n*** ANALYSES SUPPLÉMENTAIRES ***\n")
    additional_metrics = additional_analysis()
    
    # Affichage des figures
    plt.show()
