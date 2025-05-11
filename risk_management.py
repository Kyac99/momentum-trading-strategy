#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de gestion des risques pour compléter la stratégie de trading long-short.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


class RiskManager:
    """
    Classe pour gérer les risques et optimiser l'allocation des actifs dans le portefeuille.
    """
    
    def __init__(self, strategy=None, max_position_size=0.1, max_sector_exposure=0.3, 
                stop_loss=0.05, take_profit=0.1, max_drawdown=0.15):
        """
        Initialise le gestionnaire de risques.
        
        Parameters
        ----------
        strategy : MomentumStrategy, optional
            Instance de la stratégie de trading
        max_position_size : float, optional
            Taille maximale d'une position en pourcentage du portefeuille
        max_sector_exposure : float, optional
            Exposition maximale à un secteur en pourcentage du portefeuille
        stop_loss : float, optional
            Niveau de stop loss en pourcentage
        take_profit : float, optional
            Niveau de take profit en pourcentage
        max_drawdown : float, optional
            Drawdown maximum autorisé en pourcentage
        """
        self.strategy = strategy
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.sectors = {}
        
    def set_strategy(self, strategy):
        """
        Définit la stratégie à utiliser.
        
        Parameters
        ----------
        strategy : MomentumStrategy
            Instance de la stratégie de trading
        """
        self.strategy = strategy
    
    def set_sectors(self, sectors_dict):
        """
        Définit les secteurs pour chaque actif.
        
        Parameters
        ----------
        sectors_dict : dict
            Dictionnaire associant chaque actif à son secteur
        """
        self.sectors = sectors_dict
    
    def auto_detect_sectors(self, assets=None):
        """
        Détecte automatiquement les secteurs pour chaque actif.
        
        Parameters
        ----------
        assets : list, optional
            Liste des actifs. Si None, utilise strategy.assets
            
        Returns
        -------
        dict
            Dictionnaire associant chaque actif à son secteur
        """
        if assets is None:
            if self.strategy is None:
                raise ValueError("Aucune stratégie définie et aucun actif fourni.")
            assets = self.strategy.assets
        
        # Utilisation de yfinance pour obtenir les informations sectorielles
        import yfinance as yf
        
        sectors = {}
        for asset in assets:
            try:
                ticker = yf.Ticker(asset)
                info = ticker.info
                if 'sector' in info:
                    sectors[asset] = info['sector']
                else:
                    sectors[asset] = 'Unknown'
            except:
                sectors[asset] = 'Unknown'
        
        self.sectors = sectors
        return sectors
    
    def optimize_position_sizes(self, long_portfolio, short_portfolio, returns=None, risk_aversion=1.0):
        """
        Optimise la taille des positions en fonction des rendements attendus et des risques.
        
        Parameters
        ----------
        long_portfolio : pandas.DataFrame
            Portfolio des positions longues
        short_portfolio : pandas.DataFrame
            Portfolio des positions courtes
        returns : pandas.DataFrame, optional
            Rendements historiques. Si None, utilise strategy.returns
        risk_aversion : float, optional
            Coefficient d'aversion au risque
            
        Returns
        -------
        tuple
            Tuple contenant deux DataFrames (long_weights, short_weights)
        """
        if returns is None:
            if self.strategy is None or self.strategy.returns is None:
                raise ValueError("Aucun rendement disponible.")
            returns = self.strategy.returns
        
        # Extraction des positions actives
        active_long = long_portfolio.iloc[-1]
        active_short = short_portfolio.iloc[-1]
        
        long_assets = active_long[active_long].index.tolist()
        short_assets = active_short[active_short].index.tolist()
        
        if not long_assets and not short_assets:
            return pd.Series(), pd.Series()
        
        # Calcul des rendements moyens et de la matrice de covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Optimisation des poids avec la méthode de Markowitz
        long_weights = self._markowitz_optimization(long_assets, mean_returns, cov_matrix, risk_aversion)
        short_weights = self._markowitz_optimization(short_assets, -mean_returns, cov_matrix, risk_aversion)
        
        # Application des contraintes de taille maximale
        long_weights = self._apply_max_size_constraint(long_weights)
        short_weights = self._apply_max_size_constraint(short_weights)
        
        # Application des contraintes sectorielles si les secteurs sont définis
        if self.sectors:
            long_weights = self._apply_sector_constraints(long_weights)
            short_weights = self._apply_sector_constraints(short_weights)
        
        return long_weights, short_weights
    
    def _markowitz_optimization(self, assets, mean_returns, cov_matrix, risk_aversion):
        """
        Optimisation des poids selon la théorie de Markowitz.
        
        Parameters
        ----------
        assets : list
            Liste des actifs à optimiser
        mean_returns : pandas.Series
            Rendements moyens
        cov_matrix : pandas.DataFrame
            Matrice de covariance
        risk_aversion : float
            Coefficient d'aversion au risque
            
        Returns
        -------
        pandas.Series
            Poids optimisés
        """
        if not assets:
            return pd.Series()
        
        # Extraction des données pertinentes
        relevant_returns = mean_returns.loc[assets]
        relevant_cov = cov_matrix.loc[assets, assets]
        
        # Calcul des poids optimaux (formule de Markowitz)
        try:
            inv_cov = np.linalg.inv(relevant_cov.values)
            weights = np.dot(inv_cov, relevant_returns.values) / risk_aversion
            
            # Normalisation des poids
            if abs(weights.sum()) > 0:
                weights = weights / abs(weights.sum())
            
            # Conversion en Series
            return pd.Series(weights, index=assets)
        except:
            # En cas d'erreur (matrice singulière), utiliser des poids égaux
            return pd.Series(1.0 / len(assets), index=assets)
    
    def _apply_max_size_constraint(self, weights):
        """
        Applique la contrainte de taille maximale des positions.
        
        Parameters
        ----------
        weights : pandas.Series
            Poids des positions
            
        Returns
        -------
        pandas.Series
            Poids ajustés
        """
        if weights.empty:
            return weights
        
        # Écrêtage des poids dépassant la taille maximale
        excess = weights[weights > self.max_position_size] - self.max_position_size
        if not excess.empty:
            weights[weights > self.max_position_size] = self.max_position_size
            # Redistribution de l'excès aux autres positions
            remaining = weights[weights < self.max_position_size].index
            if len(remaining) > 0:
                total_excess = excess.sum()
                weights.loc[remaining] += total_excess / len(remaining)
        
        # Normalisation finale
        if abs(weights.sum()) > 0:
            weights = weights / abs(weights.sum())
        
        return weights
    
    def _apply_sector_constraints(self, weights):
        """
        Applique les contraintes d'exposition sectorielle.
        
        Parameters
        ----------
        weights : pandas.Series
            Poids des positions
            
        Returns
        -------
        pandas.Series
            Poids ajustés
        """
        if weights.empty or not self.sectors:
            return weights
        
        # Calcul de l'exposition par secteur
        sector_exposure = {}
        for asset, weight in weights.items():
            sector = self.sectors.get(asset, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # Vérification des secteurs dépassant l'exposition maximale
        excess_sectors = {s: e for s, e in sector_exposure.items() if e > self.max_sector_exposure}
        
        if excess_sectors:
            for sector, exposure in excess_sectors.items():
                # Actifs du secteur
                sector_assets = [a for a in weights.index if self.sectors.get(a, 'Unknown') == sector]
                
                # Calcul du facteur de réduction
                reduction_factor = self.max_sector_exposure / exposure
                
                # Réduction des poids
                for asset in sector_assets:
                    weights[asset] *= reduction_factor
            
            # Normalisation après ajustement
            if abs(weights.sum()) > 0:
                weights = weights / abs(weights.sum())
        
        return weights
    
    def calculate_var(self, portfolio_returns, confidence_level=0.95, method='historical'):
        """
        Calcule la Value at Risk (VaR) du portefeuille.
        
        Parameters
        ----------
        portfolio_returns : pandas.Series
            Rendements du portefeuille
        confidence_level : float, optional
            Niveau de confiance (défaut: 0.95)
        method : str, optional
            Méthode de calcul ('historical', 'parametric', 'monte_carlo')
            
        Returns
        -------
        float
            Value at Risk
        """
        if portfolio_returns.empty:
            return 0.0
        
        if method == 'historical':
            # VaR historique
            return -portfolio_returns.quantile(1 - confidence_level)
        
        elif method == 'parametric':
            # VaR paramétrique (distribution normale)
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(mean + z_score * std)
        
        elif method == 'monte_carlo':
            # VaR Monte Carlo
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            # Génération de scénarios
            np.random.seed(42)
            simulations = 10000
            simulated_returns = np.random.normal(mean, std, simulations)
            
            # Calcul du quantile
            return -np.percentile(simulated_returns, 100 * (1 - confidence_level))
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    
    def calculate_cvar(self, portfolio_returns, confidence_level=0.95):
        """
        Calcule la Conditional Value at Risk (CVaR) du portefeuille.
        
        Parameters
        ----------
        portfolio_returns : pandas.Series
            Rendements du portefeuille
        confidence_level : float, optional
            Niveau de confiance (défaut: 0.95)
            
        Returns
        -------
        float
            Conditional Value at Risk
        """
        if portfolio_returns.empty:
            return 0.0
        
        # Calcul de la VaR
        var = self.calculate_var(portfolio_returns, confidence_level, 'historical')
        
        # Calcul de la CVaR (Expected Shortfall)
        cvar = -portfolio_returns[portfolio_returns <= -var].mean()
        
        return cvar if not np.isnan(cvar) else var
    
    def apply_stop_loss(self, portfolio_values, current_positions):
        """
        Applique le stop loss aux positions actuelles.
        
        Parameters
        ----------
        portfolio_values : pandas.DataFrame
            Valeurs des positions
        current_positions : pandas.DataFrame
            Positions actuelles
            
        Returns
        -------
        pandas.DataFrame
            Positions ajustées
        """
        if portfolio_values.empty or current_positions.empty:
            return current_positions
        
        # Calcul des rendements depuis l'entrée en position
        # (simplifié, en pratique il faudrait suivre chaque position individuellement)
        position_returns = portfolio_values.pct_change().cumsum()
        
        # Identification des positions à fermer (stop loss)
        to_close = position_returns < -self.stop_loss
        
        # Mise à jour des positions
        adjusted_positions = current_positions.copy()
        adjusted_positions[to_close] = 0
        
        return adjusted_positions
    
    def apply_take_profit(self, portfolio_values, current_positions):
        """
        Applique le take profit aux positions actuelles.
        
        Parameters
        ----------
        portfolio_values : pandas.DataFrame
            Valeurs des positions
        current_positions : pandas.DataFrame
            Positions actuelles
            
        Returns
        -------
        pandas.DataFrame
            Positions ajustées
        """
        if portfolio_values.empty or current_positions.empty:
            return current_positions
        
        # Calcul des rendements depuis l'entrée en position
        position_returns = portfolio_values.pct_change().cumsum()
        
        # Identification des positions à fermer (take profit)
        to_close = position_returns > self.take_profit
        
        # Mise à jour des positions
        adjusted_positions = current_positions.copy()
        adjusted_positions[to_close] = 0
        
        return adjusted_positions
    
    def apply_max_drawdown_limit(self, cumulative_returns, current_positions):
        """
        Applique la limite de drawdown maximum.
        
        Parameters
        ----------
        cumulative_returns : pandas.Series
            Rendements cumulatifs du portefeuille
        current_positions : pandas.DataFrame
            Positions actuelles
            
        Returns
        -------
        pandas.DataFrame
            Positions ajustées
        """
        if cumulative_returns.empty or current_positions.empty:
            return current_positions
        
        # Calcul du drawdown actuel
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        current_drawdown = drawdown.iloc[-1]
        
        # Si le drawdown dépasse le maximum autorisé, fermer toutes les positions
        if current_drawdown > self.max_drawdown:
            return pd.DataFrame(0, index=current_positions.index, columns=current_positions.columns)
        
        return current_positions
    
    def calculate_risk_metrics(self, portfolio_returns):
        """
        Calcule diverses métriques de risque pour le portefeuille.
        
        Parameters
        ----------
        portfolio_returns : pandas.Series
            Rendements du portefeuille
            
        Returns
        -------
        dict
            Dictionnaire contenant les métriques de risque
        """
        if portfolio_returns.empty:
            return {}
        
        # Calcul des rendements cumulatifs
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calcul du drawdown
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        # Métriques de risque
        metrics = {
            'Volatility': portfolio_returns.std() * np.sqrt(252),
            'Max Drawdown': drawdown.max(),
            'VaR (95%)': self.calculate_var(portfolio_returns, 0.95, 'historical'),
            'CVaR (95%)': self.calculate_cvar(portfolio_returns, 0.95),
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis(),
            'Downside Deviation': portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252),
            'Sortino Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)) if len(portfolio_returns[portfolio_returns < 0]) > 0 else np.nan,
            'Calmar Ratio': (portfolio_returns.mean() * 252) / drawdown.max() if drawdown.max() > 0 else np.nan
        }
        
        return metrics
    
    def plot_risk_profile(self, portfolio_returns, figsize=(15, 12)):
        """
        Visualise le profil de risque du portefeuille.
        
        Parameters
        ----------
        portfolio_returns : pandas.Series
            Rendements du portefeuille
        figsize : tuple, optional
            Taille de la figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if portfolio_returns.empty:
            return plt.figure()
        
        # Calcul des rendements cumulatifs
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calcul du drawdown
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        # Calcul de la VaR à différents niveaux de confiance
        confidence_levels = [0.99, 0.95, 0.90]
        var_values = [self.calculate_var(portfolio_returns, cl, 'historical') for cl in confidence_levels]
        
        # Création de la figure
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Rendements cumulatifs
        ax1 = fig.add_subplot(gs[0, :])
        cumulative_returns.plot(ax=ax1)
        ax1.set_title('Rendements Cumulatifs', fontsize=14)
        ax1.set_ylabel('Rendement Cumulatif')
        ax1.grid(True)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        drawdown.plot(ax=ax2, color='red')
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        # Distribution des rendements
        ax3 = fig.add_subplot(gs[1, 1])
        sns.histplot(portfolio_returns, kde=True, bins=30, ax=ax3)
        
        # Ajout des lignes verticales pour la VaR
        for var, cl in zip(var_values, confidence_levels):
            ax3.axvline(-var, color='red', linestyle='--', alpha=0.7)
            ax3.text(-var, ax3.get_ylim()[1]*0.9, f'VaR {cl*100:.0f}%', 
                     rotation=90, verticalalignment='top')
        
        ax3.set_title('Distribution des Rendements', fontsize=14)
        ax3.set_xlabel('Rendement')
        ax3.grid(True)
        
        # Rendements roulants
        ax4 = fig.add_subplot(gs[2, 0])
        rolling_returns = portfolio_returns.rolling(window=21).mean() * 252  # Annualisé
        rolling_returns.plot(ax=ax4)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax4.set_title('Rendements Roulants (21 jours)', fontsize=14)
        ax4.set_ylabel('Rendement Annualisé')
        ax4.grid(True)
        
        # Volatilité roulante
        ax5 = fig.add_subplot(gs[2, 1])
        rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax5, color='purple')
        ax5.set_title('Volatilité Roulante (21 jours)', fontsize=14)
        ax5.set_ylabel('Volatilité Annualisée')
        ax5.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def plot_sector_exposure(self, weights, figsize=(12, 8)):
        """
        Visualise l'exposition sectorielle du portefeuille.
        
        Parameters
        ----------
        weights : pandas.Series
            Poids des positions
        figsize : tuple, optional
            Taille de la figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if weights.empty or not self.sectors:
            return plt.figure()
        
        # Calcul de l'exposition par secteur
        sector_exposure = {}
        for asset, weight in weights.items():
            sector = self.sectors.get(asset, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # Conversion en Series
        sector_exposure = pd.Series(sector_exposure)
        
        # Création de la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Diagramme circulaire
        sector_exposure.plot.pie(ax=ax1, autopct='%1.1f%%', shadow=True)
        ax1.set_title('Exposition Sectorielle', fontsize=14)
        ax1.set_ylabel('')
        
        # Diagramme à barres
        sector_exposure.sort_values(ascending=False).plot.bar(ax=ax2, color='skyblue')
        ax2.axhline(self.max_sector_exposure, color='red', linestyle='--', 
                   label=f'Exposition Max ({self.max_sector_exposure*100:.0f}%)')
        ax2.set_title('Exposition Sectorielle par Secteur', fontsize=14)
        ax2.set_ylabel('Exposition')
        ax2.set_xlabel('Secteur')
        ax2.legend()
        ax2.grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig


if __name__ == "__main__":
    # Exemple d'utilisation
    import yfinance as yf
    from momentum_strategy import MomentumStrategy
    
    # Données de test
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'PG', 'V']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Téléchargement des données
    data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
    
    # Initialisation de la stratégie
    strategy = MomentumStrategy(assets)
    strategy.data = data
    strategy.calculate_momentum(30)
    strategy.calculate_forward_returns(5)
    ranks = strategy.rank_assets()
    long_portfolio, short_portfolio = strategy.create_long_short_portfolio(ranks)
    strategy_returns = strategy.calculate_strategy_returns(long_portfolio, short_portfolio)
    
    # Initialisation du gestionnaire de risques
    risk_manager = RiskManager(strategy)
    
    # Détection automatique des secteurs
    sectors = risk_manager.auto_detect_sectors()
    print("Secteurs détectés:", sectors)
    
    # Optimisation des positions
    long_weights, short_weights = risk_manager.optimize_position_sizes(long_portfolio, short_portfolio)
    print("\nPoids optimisés (long):", long_weights)
    print("\nPoids optimisés (short):", short_weights)
    
    # Calcul des métriques de risque
    risk_metrics = risk_manager.calculate_risk_metrics(strategy_returns)
    print("\nMétriques de risque:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualisation du profil de risque
    risk_fig = risk_manager.plot_risk_profile(strategy_returns)
    plt.show()
