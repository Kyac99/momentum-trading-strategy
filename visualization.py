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
    
    # Calcul du nombre de positions par jour
    long_count = long_subset.sum(axis=1)
    short_count = short_subset.sum(axis=1)
    
    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Graphique pour les positions longues
    long_subset.astype(int).plot(ax=ax1, stacked=True, colormap='viridis')
    ax1.set_title('Évolution des Positions Longues', fontsize=14)
    ax1.set_ylabel('Nombre de positions')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    
    # Graphique pour les positions courtes
    short_subset.astype(int).plot(ax=ax2, stacked=True, colormap='plasma')
    ax2.set_title('Évolution des Positions Courtes', fontsize=14)
    ax2.set_ylabel('Nombre de positions')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_momentum_heat_map(momentum_scores, window=30, figsize=(15, 10)):
    """
    Crée une heatmap des scores de momentum au fil du temps.
    
    Parameters
    ----------
    momentum_scores : pandas.DataFrame
        Scores de momentum
    window : int, optional
        Fenêtre pour le calcul de la moyenne mobile
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Création d'une moyenne mobile sur window jours pour lisser les scores
    smoothed_scores = momentum_scores.rolling(window=window, min_periods=1).mean()
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Création de la heatmap
    sns.heatmap(smoothed_scores.T, cmap='RdBu_r', center=0, 
                robust=True, cbar_kws={'label': f'Momentum (moyenne mobile {window} jours)'}, 
                ax=ax)
    
    ax.set_title('Heatmap des Scores de Momentum', fontsize=14)
    ax.set_ylabel('Ticker')
    ax.set_xlabel('Date')
    
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
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={'shrink': .5},
                ax=ax)
    
    ax.set_title('Matrice de Corrélation des Rendements', fontsize=14)
    
    return fig


def plot_momentum_vs_returns(momentum_scores, returns, ticker, figsize=(12, 6)):
    """
    Visualise la relation entre le momentum et les rendements futurs pour un actif spécifique.
    
    Parameters
    ----------
    momentum_scores : pandas.DataFrame
        Scores de momentum
    returns : pandas.DataFrame
        Rendements des actifs
    ticker : str
        Symbole de l'actif à analyser
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Sélection des données pour le ticker spécifié
    common_dates = momentum_scores.index.intersection(returns.index)
    m_scores = momentum_scores.loc[common_dates, ticker]
    rets = returns.loc[common_dates, ticker]
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot avec une ligne de régression
    sns.regplot(x=m_scores, y=rets, ax=ax, scatter_kws={'alpha': 0.5})
    
    # Calcul de la corrélation de Spearman
    from scipy import stats
    corr, pval = stats.spearmanr(m_scores, rets)
    
    ax.set_title(f'Relation Momentum vs Rendements Futurs pour {ticker}\n'
                 f'Corrélation de Spearman: {corr:.3f} (p-value: {pval:.3f})', fontsize=14)
    ax.set_xlabel('Score de Momentum')
    ax.set_ylabel('Rendement Futur')
    ax.xaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_return_contribution(strategy_returns, long_returns, short_returns, figsize=(15, 7)):
    """
    Visualise la contribution des positions longues et courtes aux rendements de la stratégie.
    
    Parameters
    ----------
    strategy_returns : pandas.Series
        Rendements de la stratégie
    long_returns : pandas.Series
        Rendements des positions longues
    short_returns : pandas.Series
        Rendements des positions courtes
    figsize : tuple, optional
        Taille de la figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib
    """
    # Création d'un DataFrame avec toutes les séries
    contribution_df = pd.DataFrame({
        'Stratégie': strategy_returns,
        'Positions Longues': long_returns,
        'Positions Courtes': -short_returns  # Négatif car les positions courtes contribuent négativement
    })
    
    # Calcul des rendements cumulatifs
    cum_contribution = (1 + contribution_df).cumprod()
    
    # Création de la figure
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Rendements cumulatifs
    cum_contribution.plot(ax=axes[0], linewidth=2)
    axes[0].set_title('Contribution aux Rendements Cumulatifs', fontsize=14)
    axes[0].set_ylabel('Rendement Cumulatif')
    axes[0].yaxis.set_major_formatter(FuncFormatter(format_percentage))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Rendements mensuels
    monthly_contribution = contribution_df.resample('M').sum()
    monthly_contribution.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Contribution aux Rendements Mensuels', fontsize=14)
    axes[1].set_ylabel('Rendement')
    axes[1].yaxis.set_major_formatter(FuncFormatter(format_percentage))
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Exemple d'utilisation
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Téléchargement de données pour l'exemple
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Données de prix
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calcul des rendements
    returns = prices.pct_change().dropna()
    
    # Exemple de rendements de stratégie (fictifs pour cet exemple)
    strategy_returns = returns.mean(axis=1)
    
    # Visualisation de la performance
    fig = plot_strategy_performance(strategy_returns)
    plt.show()
