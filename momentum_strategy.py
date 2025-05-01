#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module principal implémentant la stratégie de trading long-short basée sur le momentum.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta


class MomentumStrategy:
    """
    Classe implémentant une stratégie de trading long-short basée sur le momentum.
    
    La stratégie prend des positions longues sur les actions ayant le momentum le plus faible
    et des positions courtes sur les actions ayant le momentum le plus élevé.
    """
    
    def __init__(self, assets, data=None):
        """
        Initialise la stratégie avec une liste d'actifs.
        
        Parameters
        ----------
        assets : list
            Liste des symboles d'actions (tickers)
        data : pandas.DataFrame, optional
            Données de prix historiques, si elles sont déjà disponibles
        """
        self.assets = assets
        self.data = data
        self.returns = None
        self.momentum_scores = None
        self.strategy_returns = None
        self.cumulative_returns = None
    
    def download_data(self, start_date, end_date, price_type='Close'):
        """
        Télécharge les données historiques de prix pour les actifs spécifiés.
        
        Parameters
        ----------
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        price_type : str, optional
            Type de prix à utiliser ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les prix historiques
        """
        print(f"Téléchargement des données pour {len(self.assets)} actions...")
        self.data = yf.download(self.assets, start=start_date, end=end_date)[price_type]
        self.data = self.data.dropna()
        print(f"Données téléchargées. Période: {self.data.index[0].strftime('%Y-%m-%d')} - {self.data.index[-1].strftime('%Y-%m-%d')}")
        return self.data
    
    def calculate_momentum(self, period=30):
        """
        Calcule les scores de momentum pour chaque actif.
        
        Parameters
        ----------
        period : int, optional
            Période sur laquelle calculer le momentum (en jours)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les scores de momentum
        """
        if self.data is None:
            raise ValueError("Les données de prix ne sont pas disponibles. Utilisez download_data() d'abord.")
        
        self.momentum_scores = (self.data - self.data.shift(period)) / self.data.shift(period)
        return self.momentum_scores
    
    def calculate_forward_returns(self, days=5):
        """
        Calcule les rendements futurs pour la période spécifiée.
        
        Parameters
        ----------
        days : int, optional
            Nombre de jours pour calculer les rendements futurs
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les rendements futurs
        """
        if self.data is None:
            raise ValueError("Les données de prix ne sont pas disponibles. Utilisez download_data() d'abord.")
        
        self.returns = self.data.shift(-days) / self.data - 1
        self.returns.dropna(inplace=True)
        return self.returns
    
    def rank_assets(self, scores=None):
        """
        Classe les actifs en fonction de leurs scores de momentum.
        
        Parameters
        ----------
        scores : pandas.DataFrame, optional
            Scores à utiliser pour le classement. Si None, utilise self.momentum_scores
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les rangs des actifs
        """
        if scores is None:
            if self.momentum_scores is None:
                raise ValueError("Les scores de momentum ne sont pas disponibles. Utilisez calculate_momentum() d'abord.")
            scores = self.momentum_scores
            
        return scores.rank(axis=1)
    
    def create_long_short_portfolio(self, ranks, quantiles=(0.2, 0.8)):
        """
        Crée un portefeuille long-short basé sur les rangs des actifs.
        
        Parameters
        ----------
        ranks : pandas.DataFrame
            Rangs des actifs
        quantiles : tuple, optional
            Quantiles à utiliser pour les positions longues et courtes
            
        Returns
        -------
        tuple
            Tuple contenant deux DataFrames (long_portfolio, short_portfolio)
        """
        # Calcul des quantiles
        quintiles = ranks.quantile(quantiles, axis=1).transpose()
        quintiles.columns = ['q_low', 'q_high']
        
        # Alignement des rangs et des quantiles
        aligned_rank, aligned_quintiles = ranks.align(quintiles, axis=0)
        
        # Création des portefeuilles
        long_portfolio = aligned_rank.le(aligned_quintiles['q_low'], axis=0)
        short_portfolio = aligned_rank.ge(aligned_quintiles['q_high'], axis=0)
        
        return long_portfolio, short_portfolio
    
    def calculate_strategy_returns(self, long_portfolio, short_portfolio, returns=None):
        """
        Calcule les rendements de la stratégie.
        
        Parameters
        ----------
        long_portfolio : pandas.DataFrame
            Portfolio des positions longues
        short_portfolio : pandas.DataFrame
            Portfolio des positions courtes
        returns : pandas.DataFrame, optional
            Rendements à utiliser. Si None, utilise self.returns
            
        Returns
        -------
        pandas.Series
            Série contenant les rendements de la stratégie
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("Les rendements ne sont pas disponibles. Utilisez calculate_forward_returns() d'abord.")
            returns = self.returns
            
        # On ne prend que les dates communes entre les portfolios et les rendements
        common_dates = returns.index.intersection(long_portfolio.index)
        
        # Calcul des rendements moyens pour chaque portfolio
        long_returns = returns.loc[common_dates][long_portfolio.loc[common_dates]].mean(axis=1)
        short_returns = returns.loc[common_dates][short_portfolio.loc[common_dates]].mean(axis=1)
        
        # Rendement de la stratégie = rendement long - rendement short
        self.strategy_returns = long_returns - short_returns
        
        # Calcul des rendements cumulatifs
        self.cumulative_returns = (1 + self.strategy_returns).cumprod()
        
        return self.strategy_returns
    
    def calculate_correlations(self, momentum_scores=None, returns=None):
        """
        Calcule les corrélations entre les scores de momentum et les rendements futurs.
        
        Parameters
        ----------
        momentum_scores : pandas.DataFrame, optional
            Scores de momentum à utiliser. Si None, utilise self.momentum_scores
        returns : pandas.DataFrame, optional
            Rendements à utiliser. Si None, utilise self.returns
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les corrélations et p-values pour chaque actif
        """
        if momentum_scores is None:
            if self.momentum_scores is None:
                raise ValueError("Les scores de momentum ne sont pas disponibles. Utilisez calculate_momentum() d'abord.")
            momentum_scores = self.momentum_scores
            
        if returns is None:
            if self.returns is None:
                raise ValueError("Les rendements ne sont pas disponibles. Utilisez calculate_forward_returns() d'abord.")
            returns = self.returns
            
        # Alignement des périodes pour les scores et les rendements
        common_dates = momentum_scores.index.intersection(returns.index)
        m_scores = momentum_scores.loc[common_dates]
        rets = returns.loc[common_dates]
        
        # Calcul des corrélations
        correlations = pd.DataFrame(index=returns.columns, columns=['Correlation', 'p-value'])
        
        for asset in correlations.index:
            corr, pval = stats.spearmanr(m_scores[asset], rets[asset])
            correlations.loc[asset] = [corr, pval]
            
        return correlations.sort_values('Correlation')
    
    def plot_correlations(self, correlations=None):
        """
        Visualise les corrélations entre le momentum et les rendements futurs.
        
        Parameters
        ----------
        correlations : pandas.DataFrame, optional
            Corrélations à visualiser. Si None, les calcule d'abord.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if correlations is None:
            correlations = self.calculate_correlations()
            
        plt.figure(figsize=(15, 7))
        plt.bar(range(len(correlations)), correlations['Correlation'])
        plt.xticks(range(len(correlations)), correlations.index, rotation=90)
        plt.axhline(0, color='black', linestyle='--')
        plt.title('Corrélations Momentum-Rendements par Action')
        plt.ylabel('Corrélation de Spearman')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_performance(self, strategy_returns=None):
        """
        Visualise la performance de la stratégie.
        
        Parameters
        ----------
        strategy_returns : pandas.Series, optional
            Rendements de la stratégie à visualiser. Si None, utilise self.strategy_returns
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if strategy_returns is None:
            if self.strategy_returns is None:
                raise ValueError("Les rendements de la stratégie ne sont pas disponibles.")
            strategy_returns = self.strategy_returns
            
        if self.cumulative_returns is None:
            self.cumulative_returns = (1 + strategy_returns).cumprod()
            
        plt.figure(figsize=(15, 7))
        self.cumulative_returns.plot()
        plt.title('Rendements Cumulatifs de la Stratégie')
        plt.ylabel('Rendements Cumulatifs')
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_performance_metrics(self, strategy_returns=None):
        """
        Calcule les métriques de performance de la stratégie.
        
        Parameters
        ----------
        strategy_returns : pandas.Series, optional
            Rendements de la stratégie à analyser. Si None, utilise self.strategy_returns
            
        Returns
        -------
        dict
            Dictionnaire contenant les métriques de performance
        """
        if strategy_returns is None:
            if self.strategy_returns is None:
                raise ValueError("Les rendements de la stratégie ne sont pas disponibles.")
            strategy_returns = self.strategy_returns
            
        # Calcul des métriques
        total_return = self.cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = (self.cumulative_returns / self.cumulative_returns.cummax() - 1).min()
        win_rate = (strategy_returns > 0).mean()
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate
        }
        
        return metrics
    
    def backtest(self, start_date, end_date, momentum_period=30, forward_return_days=5, 
                 quantiles=(0.2, 0.8), price_type='Close'):
        """
        Exécute un backtest complet de la stratégie.
        
        Parameters
        ----------
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        momentum_period : int, optional
            Période sur laquelle calculer le momentum (en jours)
        forward_return_days : int, optional
            Nombre de jours pour calculer les rendements futurs
        quantiles : tuple, optional
            Quantiles à utiliser pour les positions longues et courtes
        price_type : str, optional
            Type de prix à utiliser ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns
        -------
        dict
            Dictionnaire contenant les résultats du backtest
        """
        # Téléchargement des données
        self.download_data(start_date, end_date, price_type)
        
        # Calcul du momentum
        self.calculate_momentum(momentum_period)
        
        # Calcul des rendements futurs
        self.calculate_forward_returns(forward_return_days)
        
        # Classement des actifs
        ranks = self.rank_assets()
        
        # Création du portefeuille long-short
        long_portfolio, short_portfolio = self.create_long_short_portfolio(ranks, quantiles)
        
        # Calcul des rendements de la stratégie
        self.calculate_strategy_returns(long_portfolio, short_portfolio)
        
        # Calcul des métriques de performance
        metrics = self.get_performance_metrics()
        
        # Résultats du backtest
        results = {
            'data': self.data,
            'momentum_scores': self.momentum_scores,
            'returns': self.returns,
            'ranks': ranks,
            'long_portfolio': long_portfolio,
            'short_portfolio': short_portfolio,
            'strategy_returns': self.strategy_returns,
            'cumulative_returns': self.cumulative_returns,
            'metrics': metrics
        }
        
        return results


if __name__ == "__main__":
    # Exemple d'utilisation
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'PG', 'V']
    strategy = MomentumStrategy(assets)
    
    # Période de backtest : 2 ans
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    # Exécution du backtest
    results = strategy.backtest(start_date, end_date)
    
    # Affichage des métriques
    print("\nMétriques de performance:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Visualisation de la performance
    strategy.plot_performance()
    plt.show()
