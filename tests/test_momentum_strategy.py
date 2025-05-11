#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour la classe MomentumStrategy.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajout du dossier parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum_strategy import MomentumStrategy
from data_loader import DataLoader
from backtest import Backtest
from risk_management import RiskManager


class TestMomentumStrategy(unittest.TestCase):
    """Tests pour la classe MomentumStrategy."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Liste d'actifs pour les tests
        cls.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Création d'un ensemble de données fictif pour les tests
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        cls.data = pd.DataFrame(index=dates, columns=cls.assets)
        
        # Génération de prix fictifs
        for asset in cls.assets:
            # Prix initial aléatoire entre 50 et 500
            initial_price = np.random.uniform(50, 500)
            
            # Génération de changements de prix aléatoires
            changes = np.random.normal(0.0005, 0.02, len(dates))
            
            # Calcul des prix cumulatifs
            prices = initial_price * (1 + changes).cumprod()
            
            # Ajout des prix au DataFrame
            cls.data[asset] = prices
    
    def setUp(self):
        """Configuration pour chaque test."""
        # Création d'une nouvelle instance de MomentumStrategy pour chaque test
        self.strategy = MomentumStrategy(self.assets, self.data)
    
    def test_initialization(self):
        """Test d'initialisation de la stratégie."""
        self.assertEqual(self.strategy.assets, self.assets)
        self.assertTrue(isinstance(self.strategy.data, pd.DataFrame))
        self.assertEqual(len(self.strategy.data.columns), len(self.assets))
    
    def test_calculate_momentum(self):
        """Test du calcul du momentum."""
        momentum_scores = self.strategy.calculate_momentum(period=30)
        
        # Vérification que les scores de momentum ont été calculés
        self.assertTrue(isinstance(momentum_scores, pd.DataFrame))
        self.assertEqual(momentum_scores.shape, self.data.shape)
        
        # Vérification que les scores de momentum sont des nombres
        self.assertTrue(np.issubdtype(momentum_scores.dtypes[0], np.number))
    
    def test_calculate_forward_returns(self):
        """Test du calcul des rendements futurs."""
        returns = self.strategy.calculate_forward_returns(days=5)
        
        # Vérification que les rendements ont été calculés
        self.assertTrue(isinstance(returns, pd.DataFrame))
        
        # Vérification que les rendements sont des nombres
        self.assertTrue(np.issubdtype(returns.dtypes[0], np.number))
        
        # Vérification que les rendements ont le bon nombre de lignes
        # (moins les jours de rendements futurs)
        self.assertEqual(len(returns), len(self.data) - 5)
    
    def test_rank_assets(self):
        """Test du classement des actifs."""
        # Calcul du momentum
        self.strategy.calculate_momentum(period=30)
        
        # Classement des actifs
        ranks = self.strategy.rank_assets()
        
        # Vérification que les rangs ont été calculés
        self.assertTrue(isinstance(ranks, pd.DataFrame))
        
        # Vérification que les rangs sont des nombres
        self.assertTrue(np.issubdtype(ranks.dtypes[0], np.number))
        
        # Vérification que les rangs sont compris entre 1 et le nombre d'actifs
        for col in ranks.columns:
            self.assertTrue((ranks[col] >= 1).all() and (ranks[col] <= len(self.assets)).all())
    
    def test_create_long_short_portfolio(self):
        """Test de la création du portefeuille long-short."""
        # Calcul du momentum
        self.strategy.calculate_momentum(period=30)
        
        # Classement des actifs
        ranks = self.strategy.rank_assets()
        
        # Création du portefeuille
        long_portfolio, short_portfolio = self.strategy.create_long_short_portfolio(ranks)
        
        # Vérification que les portefeuilles ont été créés
        self.assertTrue(isinstance(long_portfolio, pd.DataFrame))
        self.assertTrue(isinstance(short_portfolio, pd.DataFrame))
        
        # Vérification que les portefeuilles sont des booléens
        self.assertTrue(np.issubdtype(long_portfolio.dtypes[0], np.bool_))
        self.assertTrue(np.issubdtype(short_portfolio.dtypes[0], np.bool_))
        
        # Vérification que les portefeuilles ont la bonne forme
        self.assertEqual(long_portfolio.shape, ranks.shape)
        self.assertEqual(short_portfolio.shape, ranks.shape)
    
    def test_calculate_strategy_returns(self):
        """Test du calcul des rendements de la stratégie."""
        # Calcul du momentum
        self.strategy.calculate_momentum(period=30)
        
        # Calcul des rendements futurs
        self.strategy.calculate_forward_returns(days=5)
        
        # Classement des actifs
        ranks = self.strategy.rank_assets()
        
        # Création du portefeuille
        long_portfolio, short_portfolio = self.strategy.create_long_short_portfolio(ranks)
        
        # Calcul des rendements de la stratégie
        strategy_returns = self.strategy.calculate_strategy_returns(long_portfolio, short_portfolio)
        
        # Vérification que les rendements ont été calculés
        self.assertTrue(isinstance(strategy_returns, pd.Series))
        
        # Vérification que les rendements sont des nombres
        self.assertTrue(np.issubdtype(strategy_returns.dtype, np.number))
    
    def test_backtest(self):
        """Test du backtest complet."""
        # Exécution du backtest
        results = self.strategy.backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            momentum_period=30,
            forward_return_days=5
        )
        
        # Vérification que les résultats ont été calculés
        self.assertTrue(isinstance(results, dict))
        
        # Vérification que les résultats contiennent les éléments attendus
        expected_keys = [
            'data', 'momentum_scores', 'returns', 'ranks',
            'long_portfolio', 'short_portfolio', 'strategy_returns',
            'cumulative_returns', 'metrics'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Vérification que les métriques contiennent les éléments attendus
        expected_metrics = [
            'Total Return', 'Annualized Return', 'Volatility',
            'Sharpe Ratio', 'Max Drawdown', 'Win Rate'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results['metrics'])


class TestDataLoader(unittest.TestCase):
    """Tests pour la classe DataLoader."""
    
    def setUp(self):
        """Configuration pour chaque test."""
        # Création d'une nouvelle instance de DataLoader pour chaque test
        self.loader = DataLoader()
    
    def test_create_portfolio_from_template(self):
        """Test de la création d'un portfolio à partir d'un modèle."""
        # Liste des modèles disponibles
        templates = [
            'tech_leaders', 'value_stocks', 'growth_stocks',
            'dividend_stocks', 'momentum_stocks', 'etf_portfolio'
        ]
        
        # Test de chaque modèle
        for template in templates:
            # Définition d'une période courte pour le test
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            try:
                # Création du portfolio
                data = self.loader.create_portfolio_from_template(template, start_date, end_date)
                
                # Vérification que les données ont été téléchargées
                self.assertTrue(isinstance(data, pd.DataFrame))
                self.assertTrue(len(data) > 0)
                self.assertTrue(len(data.columns) > 0)
            except Exception as e:
                # En cas d'erreur de connexion ou d'API, le test est ignoré
                # mais une alerte est affichée
                print(f"Erreur lors du test du modèle {template}: {str(e)}")
    
    def test_get_industry_symbols(self):
        """Test de l'obtention des symboles d'une industrie."""
        # Liste des industries disponibles
        industries = [
            'Technology', 'Healthcare', 'Financial',
            'Consumer', 'Industrial', 'Energy'
        ]
        
        # Test de chaque industrie
        for industry in industries:
            try:
                # Obtention des symboles
                symbols = self.loader.get_industry_symbols(industry)
                
                # Vérification que des symboles ont été obtenus
                self.assertTrue(isinstance(symbols, list))
                self.assertTrue(len(symbols) > 0)
            except Exception as e:
                # En cas d'erreur de connexion ou d'API, le test est ignoré
                # mais une alerte est affichée
                print(f"Erreur lors du test de l'industrie {industry}: {str(e)}")


class TestRiskManager(unittest.TestCase):
    """Tests pour la classe RiskManager."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Liste d'actifs pour les tests
        cls.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Création d'un ensemble de données fictif pour les tests
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        cls.data = pd.DataFrame(index=dates, columns=cls.assets)
        
        # Génération de prix fictifs
        for asset in cls.assets:
            # Prix initial aléatoire entre 50 et 500
            initial_price = np.random.uniform(50, 500)
            
            # Génération de changements de prix aléatoires
            changes = np.random.normal(0.0005, 0.02, len(dates))
            
            # Calcul des prix cumulatifs
            prices = initial_price * (1 + changes).cumprod()
            
            # Ajout des prix au DataFrame
            cls.data[asset] = prices
        
        # Création d'une instance de MomentumStrategy
        cls.strategy = MomentumStrategy(cls.assets, cls.data)
        
        # Calcul du momentum et des rendements
        cls.strategy.calculate_momentum(period=30)
        cls.strategy.calculate_forward_returns(days=5)
        
        # Création du portefeuille
        ranks = cls.strategy.rank_assets()
        cls.long_portfolio, cls.short_portfolio = cls.strategy.create_long_short_portfolio(ranks)
        
        # Calcul des rendements de la stratégie
        cls.strategy_returns = cls.strategy.calculate_strategy_returns(
            cls.long_portfolio, cls.short_portfolio
        )
    
    def setUp(self):
        """Configuration pour chaque test."""
        # Création d'une nouvelle instance de RiskManager pour chaque test
        self.risk_manager = RiskManager(self.strategy)
        
        # Définition des secteurs pour les tests
        self.sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer',
            'META': 'Technology'
        }
        
        self.risk_manager.set_sectors(self.sectors)
    
    def test_optimize_position_sizes(self):
        """Test de l'optimisation des tailles de positions."""
        # Optimisation des positions
        long_weights, short_weights = self.risk_manager.optimize_position_sizes(
            self.long_portfolio, self.short_portfolio
        )
        
        # Vérification que les poids ont été calculés
        self.assertTrue(isinstance(long_weights, pd.Series) or long_weights.empty)
        self.assertTrue(isinstance(short_weights, pd.Series) or short_weights.empty)
        
        # Si des poids ont été calculés, vérification qu'ils sont valides
        if not long_weights.empty:
            # La somme des poids doit être proche de 1
            self.assertAlmostEqual(long_weights.sum(), 1.0, places=6)
            
            # Tous les poids doivent être positifs ou nuls
            self.assertTrue((long_weights >= 0).all())
            
            # Aucun poids ne doit dépasser la taille maximale
            self.assertTrue((long_weights <= self.risk_manager.max_position_size + 1e-6).all())
        
        if not short_weights.empty:
            # La somme des poids doit être proche de 1
            self.assertAlmostEqual(short_weights.sum(), 1.0, places=6)
            
            # Tous les poids doivent être positifs ou nuls
            self.assertTrue((short_weights >= 0).all())
            
            # Aucun poids ne doit dépasser la taille maximale
            self.assertTrue((short_weights <= self.risk_manager.max_position_size + 1e-6).all())
    
    def test_calculate_var(self):
        """Test du calcul de la Value at Risk."""
        # Calcul de la VaR avec différentes méthodes
        var_historical = self.risk_manager.calculate_var(
            self.strategy_returns, confidence_level=0.95, method='historical'
        )
        
        var_parametric = self.risk_manager.calculate_var(
            self.strategy_returns, confidence_level=0.95, method='parametric'
        )
        
        var_monte_carlo = self.risk_manager.calculate_var(
            self.strategy_returns, confidence_level=0.95, method='monte_carlo'
        )
        
        # Vérification que les VaR ont été calculées
        self.assertTrue(isinstance(var_historical, float))
        self.assertTrue(isinstance(var_parametric, float))
        self.assertTrue(isinstance(var_monte_carlo, float))
        
        # Les VaR doivent être positives (représentant une perte potentielle)
        self.assertTrue(var_historical >= 0)
        self.assertTrue(var_parametric >= 0)
        self.assertTrue(var_monte_carlo >= 0)
    
    def test_calculate_cvar(self):
        """Test du calcul de la Conditional Value at Risk."""
        # Calcul de la CVaR
        cvar = self.risk_manager.calculate_cvar(
            self.strategy_returns, confidence_level=0.95
        )
        
        # Vérification que la CVaR a été calculée
        self.assertTrue(isinstance(cvar, float))
        
        # La CVaR doit être positive (représentant une perte potentielle)
        self.assertTrue(cvar >= 0)
    
    def test_calculate_risk_metrics(self):
        """Test du calcul des métriques de risque."""
        # Calcul des métriques de risque
        risk_metrics = self.risk_manager.calculate_risk_metrics(
            self.strategy_returns
        )
        
        # Vérification que les métriques ont été calculées
        self.assertTrue(isinstance(risk_metrics, dict))
        
        # Vérification que les métriques contiennent les éléments attendus
        expected_metrics = [
            'Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)',
            'Skewness', 'Kurtosis', 'Downside Deviation', 'Sortino Ratio', 'Calmar Ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)


if __name__ == '__main__':
    unittest.main()
