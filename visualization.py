#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de visualisation avancée pour la stratégie de trading long-short.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec


def format_percentage(x, pos):
    """Formateur pour afficher les nombres en pourcentage."""
    return f'{100 * x:.1f}%'


def plot_strategy_performance(strategy_returns, benchmark_returns=None, figsize=(15, 10)):
    """
    Visualise les performances de la stratégie avec plusieurs graphiques.
    
    Parameters
    ----------
    strategy_returns : pandas.Series
        Rendements de la stratégie
    benchmark_returns : pandas.Series, optional
        Rendements d'un benchmark pour comparaison
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Calcul des rendements cumulatifs
    cum_returns = (1 + strategy_returns).cumprod()
    
    # Création de la figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Rendements cumulatifs
    ax1 = fig.add_subplot(gs[0, :])
    cum_returns.plot(ax=ax1, color='blue', linewidth=2)
    
    if benchmark_returns is not None:
        cum_bench_returns = (1 + benchmark_returns).cumprod()
        cum_bench_returns.plot(ax=ax1, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.legend(['Stratégie', 'Benchmark'])
    
    ax1.set_title('Rendements Cumulatifs', fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax1.set_ylabel('Rendement Cumulatif')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rendements mensuels
    ax2 = fig.add_subplot(gs[1, 0])
    monthly_returns = strategy_returns.resample('M').sum()
    monthly_returns.plot(kind='bar', ax=ax2, color=monthly_returns.map(lambda x: 'green' if x >= 0 else 'red'))
    ax2.set_title('Rendements Mensuels', fontsize=14)
    ax2.set_ylabel('Rendement')
    ax2.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Distribution des rendements
    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(strategy_returns, ax=ax3, kde=True, bins=30)
    ax3.set_title('Distribution des Rendements', fontsize=14)
    ax3.set_xlabel('Rendement')
    ax3.xaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdowns
    ax4 = fig.add_subplot(gs[2, 0])
    drawdowns = (cum_returns / cum_returns.cummax() - 1)
    drawdowns.plot(ax=ax4, color='red', linewidth=1)
    ax4.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax4.set_title('Drawdowns', fontsize=14)
    ax4.set_ylabel('Drawdown')
    ax4.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax4.grid(True, alpha=0.3)
    
    # 5. Rendements cumulatifs annualisés
    ax5 = fig.add_subplot(gs[2, 1])
    yearly_returns = strategy_returns.resample('Y').sum()
    yearly_returns.index = yearly_returns.index.year
    yearly_returns.plot(kind='bar', ax=ax5, color=yearly_returns.map(lambda x: 'green' if x >= 0 else 'red'))
    ax5.set_title('Rendements Annuels', fontsize=14)
    ax5.set_ylabel('Rendement')
    ax5.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def plot_position_evolution(long_portfolio, short_portfolio, tickers=None, figsize=(15, 10)):
    """
    Visualise l'évolution des positions longues et courtes au fil du temps.
    
    Parameters
    ----------
    long_portfolio : pandas.DataFrame
        Portfolio des positions longues
    short_portfolio : pandas.DataFrame
        Portfolio des positions courtes
    tickers : list, optional
        Liste des tickers à inclure (si None, tous sont inclus)
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    if tickers is None:
        tickers = long_portfolio.columns
    
    # Sélection des tickers
    long_subset = long_portfolio[tickers]
    short_subset = short_portfolio[tickers]
    
    # Calcul du nombre de positions à chaque date
    long_count = long_subset.sum(axis=1)
    short_count = short_subset.sum(axis=1)
    
    # Création de la figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 1. Évolution du nombre de positions
    ax1 = axes[0]
    long_count.plot(ax=ax1, color='green', linewidth=2, label='Positions Longues')
    short_count.plot(ax=ax1, color='red', linewidth=2, label='Positions Courtes')
    ax1.set_title('Évolution du Nombre de Positions', fontsize=14)
    ax1.set_ylabel('Nombre de Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Heatmap des positions par actif
    ax2 = axes[1]
    
    # Combinaison des positions (1 pour long, -1 pour short, 0 pour pas de position)
    combined = pd.DataFrame(0, index=long_subset.index, columns=tickers)
    for date in combined.index:
        for ticker in tickers:
            if long_subset.loc[date, ticker]:
                combined.loc[date, ticker] = 1
            elif short_subset.loc[date, ticker]:
                combined.loc[date, ticker] = -1
    
    # Création de la heatmap
    sns.heatmap(combined, cmap='RdYlGn', center=0, ax=ax2)
    ax2.set_title('Positions par Actif au Fil du Temps', fontsize=14)
    ax2.set_ylabel('Date')
    ax2.set_xlabel('Actif')
    
    plt.tight_layout()
    
    return fig


def plot_momentum_heatmap(momentum_scores, tickers=None, figsize=(15, 10)):
    """
    Visualise l'évolution des scores de momentum sous forme de heatmap.
    
    Parameters
    ----------
    momentum_scores : pandas.DataFrame
        Scores de momentum
    tickers : list, optional
        Liste des tickers à inclure (si None, tous sont inclus)
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    if tickers is None:
        tickers = momentum_scores.columns
    
    # Sélection des tickers
    scores_subset = momentum_scores[tickers]
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Création de la heatmap
    sns.heatmap(scores_subset, cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Évolution des Scores de Momentum', fontsize=14)
    ax.set_ylabel('Date')
    ax.set_xlabel('Actif')
    
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(returns, figsize=(12, 10)):
    """
    Visualise la matrice de corrélation des rendements.
    
    Parameters
    ----------
    returns : pandas.DataFrame
        Rendements des actifs
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Calcul de la matrice de corrélation
    corr_matrix = returns.corr()
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Création de la heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title('Matrice de Corrélation des Rendements', fontsize=14)
    
    plt.tight_layout()
    
    return fig


def plot_performance_metrics(metrics, benchmark_metrics=None, figsize=(12, 8)):
    """
    Visualise les métriques de performance.
    
    Parameters
    ----------
    metrics : dict
        Dictionnaire des métriques de performance
    benchmark_metrics : dict, optional
        Dictionnaire des métriques de performance du benchmark
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Préparation des données
    metrics_df = pd.DataFrame({
        'Stratégie': pd.Series(metrics)
    })
    
    if benchmark_metrics is not None:
        metrics_df['Benchmark'] = pd.Series(benchmark_metrics)
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Création du graphique à barres
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Métriques de Performance', fontsize=14)
    ax.set_ylabel('Valeur')
    ax.set_xticklabels(metrics_df.index, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajout des valeurs au-dessus des barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
    
    plt.tight_layout()
    
    return fig


def plot_rolling_performance(strategy_returns, window=252, figsize=(15, 15)):
    """
    Visualise les métriques de performance sur une fenêtre glissante.
    
    Parameters
    ----------
    strategy_returns : pandas.Series
        Rendements de la stratégie
    window : int, optional
        Taille de la fenêtre glissante (en jours)
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Calcul des métriques glissantes
    rolling_return = strategy_returns.rolling(window=window).mean() * 252  # Annualisation
    rolling_vol = strategy_returns.rolling(window=window).std() * np.sqrt(252)  # Annualisation
    rolling_sharpe = rolling_return / rolling_vol
    rolling_drawdown = (1 + strategy_returns).cumprod().rolling(window=window).apply(
        lambda x: (x / x.cummax() - 1).min(), raw=True)
    
    # Création de la figure
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # 1. Rendement annualisé glissant
    ax1 = axes[0]
    rolling_return.plot(ax=ax1, color='blue', linewidth=2)
    ax1.set_title('Rendement Annualisé Glissant', fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax1.set_ylabel('Rendement')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatilité annualisée glissante
    ax2 = axes[1]
    rolling_vol.plot(ax=ax2, color='orange', linewidth=2)
    ax2.set_title('Volatilité Annualisée Glissante', fontsize=14)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax2.set_ylabel('Volatilité')
    ax2.grid(True, alpha=0.3)
    
    # 3. Ratio de Sharpe glissant
    ax3 = axes[2]
    rolling_sharpe.plot(ax=ax3, color='green', linewidth=2)
    ax3.set_title('Ratio de Sharpe Glissant', fontsize=14)
    ax3.set_ylabel('Ratio de Sharpe')
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown maximal glissant
    ax4 = axes[3]
    rolling_drawdown.plot(ax=ax4, color='red', linewidth=2)
    ax4.set_title('Drawdown Maximal Glissant', fontsize=14)
    ax4.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax4.set_ylabel('Drawdown')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Exemple d'utilisation
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Téléchargement de données pour l'exemple
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    
    # Création de rendements fictifs pour la démonstration
    returns = spy_data.pct_change().dropna()
    strategy_returns = returns * 1.2  # Stratégie fictive surperformant de 20%
    
    # Visualisation des performances
    fig = plot_strategy_performance(strategy_returns, returns)
    plt.show()
