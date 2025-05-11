#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de backtest pour évaluer la stratégie de trading long-short.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import os

# Imports internes
from data_loader import DataLoader
from momentum_strategy import MomentumStrategy
from visualization import plot_strategy_performance, plot_momentum_heat_map


class Backtest:
    """
    Classe pour effectuer des backtests complets de la stratégie de trading.
    """
    
    def __init__(self, strategy, start_date, end_date, initial_capital=10000):
        """
        Initialise le backtest.
        
        Parameters
        ----------
        strategy : MomentumStrategy
            Instance de la stratégie de trading
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        initial_capital : float, optional
            Capital initial pour le backtest
        """
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.results = None
        self.performance_metrics = None
        self.portfolio_values = None
    
    def run(self, momentum_period=30, forward_return_days=5, quantiles=(0.2, 0.8), 
           price_type='Close', benchmark_symbol='SPY', transaction_cost=0.001):
        """
        Exécute le backtest complet.
        
        Parameters
        ----------
        momentum_period : int, optional
            Période sur laquelle calculer le momentum (en jours)
        forward_return_days : int, optional
            Nombre de jours pour calculer les rendements futurs
        quantiles : tuple, optional
            Quantiles à utiliser pour les positions longues et courtes
        price_type : str, optional
            Type de prix à utiliser ('Open', 'High', 'Low', 'Close', 'Adj Close')
        benchmark_symbol : str, optional
            Symbole du benchmark (e.g., 'SPY', '^GSPC')
        transaction_cost : float, optional
            Coût de transaction en pourcentage
            
        Returns
        -------
        dict
            Dictionnaire contenant les résultats du backtest
        """
        print(f"Exécution du backtest de {self.start_date} à {self.end_date}...")
        
        # Exécution de la stratégie
        self.results = self.strategy.backtest(
            self.start_date, 
            self.end_date, 
            momentum_period, 
            forward_return_days, 
            quantiles, 
            price_type
        )
        
        # Téléchargement des données du benchmark
        loader = DataLoader()
        benchmark_returns = loader.get_benchmark_returns(
            benchmark_symbol, 
            self.start_date, 
            self.end_date
        )
        
        # Alignement des dates
        common_dates = self.results['strategy_returns'].index.intersection(benchmark_returns.index)
        strategy_returns = self.results['strategy_returns'].loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Prise en compte des coûts de transaction
        if transaction_cost > 0:
            # Calcul du nombre de positions qui changent à chaque période
            long_changes = self.results['long_portfolio'].diff().abs().sum(axis=1)
            short_changes = self.results['short_portfolio'].diff().abs().sum(axis=1)
            total_changes = long_changes + short_changes
            
            # Application des coûts de transaction
            transaction_costs = total_changes * transaction_cost / 2  # Division par 2 car les coûts sont par position
            strategy_returns = strategy_returns - transaction_costs
        
        # Calcul de la valeur du portefeuille
        self.portfolio_values = self.calculate_portfolio_values(strategy_returns, benchmark_returns)
        
        # Calcul des métriques de performance
        self.performance_metrics = self.calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # Sauvegarde des résultats
        self.results.update({
            'benchmark_returns': benchmark_returns,
            'portfolio_values': self.portfolio_values,
            'performance_metrics': self.performance_metrics,
            'transaction_costs': transaction_costs if transaction_cost > 0 else None
        })
        
        print("Backtest terminé avec succès.")
        return self.results
    
    def calculate_portfolio_values(self, strategy_returns, benchmark_returns):
        """
        Calcule la valeur du portefeuille au fil du temps.
        
        Parameters
        ----------
        strategy_returns : pandas.Series
            Rendements de la stratégie
        benchmark_returns : pandas.Series
            Rendements du benchmark
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les valeurs du portefeuille
        """
        # Calcul des rendements cumulatifs
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        
        # Valeurs du portefeuille
        portfolio_values = pd.DataFrame({
            'Strategy': strategy_cum_returns * self.initial_capital,
            'Benchmark': benchmark_cum_returns * self.initial_capital
        })
        
        return portfolio_values
    
    def calculate_performance_metrics(self, strategy_returns, benchmark_returns):
        """
        Calcule les métriques de performance avancées.
        
        Parameters
        ----------
        strategy_returns : pandas.Series
            Rendements de la stratégie
        benchmark_returns : pandas.Series
            Rendements du benchmark
            
        Returns
        -------
        dict
            Dictionnaire contenant les métriques de performance
        """
        # Calcul des métriques de base
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        
        # Rendement total
        total_return = strategy_cum_returns.iloc[-1] - 1
        
        # Rendement annualisé (252 jours de trading par an)
        years = len(strategy_returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatilité annualisée
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Ratio de Sharpe (rendement sans risque = 0 pour simplifier)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Ratio de Sortino (rendement sans risque = 0 pour simplifier)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        
        # Drawdown maximum
        drawdown = 1 - strategy_cum_returns / strategy_cum_returns.cummax()
        max_drawdown = drawdown.max()
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        # Métriques relatives au benchmark
        alpha, beta = self.calculate_alpha_beta(strategy_returns, benchmark_returns)
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (strategy_returns.mean() - benchmark_returns.mean()) * 252 / tracking_error if tracking_error != 0 else 0
        
        # Métriques de drawdown
        avg_drawdown = drawdown[drawdown > 0].mean() if len(drawdown[drawdown > 0]) > 0 else 0
        drawdown_duration = self.calculate_drawdown_duration(drawdown)
        
        # Création du dictionnaire de métriques
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Average Drawdown': avg_drawdown,
            'Max Drawdown Duration': drawdown_duration['max'],
            'Average Drawdown Duration': drawdown_duration['avg'],
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Alpha': alpha,
            'Beta': beta,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Correlation with Benchmark': strategy_returns.corr(benchmark_returns)
        }
        
        return metrics
    
    def calculate_alpha_beta(self, strategy_returns, benchmark_returns):
        """
        Calcule l'alpha et le bêta de la stratégie par rapport au benchmark.
        
        Parameters
        ----------
        strategy_returns : pandas.Series
            Rendements de la stratégie
        benchmark_returns : pandas.Series
            Rendements du benchmark
            
        Returns
        -------
        tuple
            Tuple contenant (alpha, beta)
        """
        # Régression pour calculer alpha et beta
        X = benchmark_returns.values.reshape(-1, 1)
        X = np.concatenate([np.ones_like(X), X], axis=1)
        y = strategy_returns.values
        
        # Calcul par la méthode des moindres carrés
        try:
            beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = alpha * 252  # Annualisation
        except:
            alpha, beta = 0, 0
        
        return alpha, beta
    
    def calculate_drawdown_duration(self, drawdown, threshold=0.0):
        """
        Calcule la durée des drawdowns.
        
        Parameters
        ----------
        drawdown : pandas.Series
            Série contenant les drawdowns
        threshold : float, optional
            Seuil à partir duquel considérer un drawdown
            
        Returns
        -------
        dict
            Dictionnaire contenant les durées de drawdown
        """
        # Identification des périodes de drawdown
        in_drawdown = drawdown > threshold
        
        if not in_drawdown.any():
            return {'max': 0, 'avg': 0, 'current': 0}
        
        # Calcul des changements d'état (entrée ou sortie de drawdown)
        state_change = in_drawdown.astype(int).diff().fillna(0)
        
        # Identification des entrées en drawdown
        drawdown_start = drawdown.index[state_change == 1].tolist()
        
        # Identification des sorties de drawdown
        drawdown_end = drawdown.index[state_change == -1].tolist()
        
        # Gestion des cas particuliers
        if in_drawdown.iloc[0]:
            drawdown_start.insert(0, drawdown.index[0])
        
        if in_drawdown.iloc[-1]:
            drawdown_end.append(drawdown.index[-1])
        
        # Calcul des durées
        durations = []
        for start, end in zip(drawdown_start, drawdown_end):
            duration = (drawdown.index.get_indexer([end])[0] - drawdown.index.get_indexer([start])[0])
            durations.append(duration)
        
        # Résultats
        result = {
            'max': max(durations) if durations else 0,
            'avg': sum(durations) / len(durations) if durations else 0,
            'current': durations[-1] if durations and in_drawdown.iloc[-1] else 0
        }
        
        return result
    
    def plot_performance(self, figsize=(15, 10), save_path=None):
        """
        Visualise les performances du backtest.
        
        Parameters
        ----------
        figsize : tuple, optional
            Taille de la figure
        save_path : str, optional
            Chemin pour sauvegarder la figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if self.results is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Utilisez run() d'abord.")
        
        # Visualisation des performances
        fig = plot_strategy_performance(
            self.results['strategy_returns'], 
            self.results['benchmark_returns'], 
            figsize=figsize
        )
        
        # Sauvegarde de la figure si nécessaire
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_portfolio_evolution(self, figsize=(15, 7), save_path=None):
        """
        Visualise l'évolution de la valeur du portefeuille.
        
        Parameters
        ----------
        figsize : tuple, optional
            Taille de la figure
        save_path : str, optional
            Chemin pour sauvegarder la figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if self.portfolio_values is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Utilisez run() d'abord.")
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracé des valeurs
        self.portfolio_values.plot(ax=ax, linewidth=2)
        
        ax.set_title('Évolution de la Valeur du Portefeuille', fontsize=14)
        ax.set_ylabel('Valeur ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Sauvegarde de la figure si nécessaire
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heat_map(self, figsize=(15, 10), window=30, save_path=None):
        """
        Visualise une heatmap des scores de momentum.
        
        Parameters
        ----------
        figsize : tuple, optional
            Taille de la figure
        window : int, optional
            Fenêtre pour le calcul de la moyenne mobile
        save_path : str, optional
            Chemin pour sauvegarder la figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure matplotlib
        """
        if self.results is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Utilisez run() d'abord.")
        
        # Création de la figure
        fig = plot_momentum_heat_map(
            self.results['momentum_scores'], 
            window=window, 
            figsize=figsize
        )
        
        # Sauvegarde de la figure si nécessaire
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_performance_summary(self):
        """
        Affiche un résumé des performances du backtest.
        """
        if self.performance_metrics is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Utilisez run() d'abord.")
        
        print("\n" + "="*50)
        print("RÉSUMÉ DES PERFORMANCES DU BACKTEST")
        print("="*50)
        
        print(f"\nPériode: {self.results['strategy_returns'].index[0].strftime('%Y-%m-%d')} à {self.results['strategy_returns'].index[-1].strftime('%Y-%m-%d')}")
        print(f"Nombre de jours de trading: {len(self.results['strategy_returns'])}")
        
        print("\nMÉTRIQUES DE PERFORMANCE:")
        print("-"*50)
        
        # Format d'affichage
        metrics_format = {
            'Total Return': '{:.2%}',
            'Annualized Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'Average Drawdown': '{:.2%}',
            'Max Drawdown Duration': '{:.0f} jours',
            'Average Drawdown Duration': '{:.1f} jours',
            'Calmar Ratio': '{:.2f}',
            'Win Rate': '{:.2%}',
            'Alpha': '{:.2%}',
            'Beta': '{:.2f}',
            'Tracking Error': '{:.2%}',
            'Information Ratio': '{:.2f}',
            'Correlation with Benchmark': '{:.2f}'
        }
        
        for metric, value in self.performance_metrics.items():
            print(f"{metric+':':.<30} {metrics_format.get(metric, '{:.4f}').format(value)}")
        
        print("\nRÉSULTATS PAR ANNÉE:")
        print("-"*50)
        
        # Calcul des rendements annuels
        yearly_returns = self.results['strategy_returns'].groupby(pd.Grouper(freq='Y')).apply(lambda x: (1 + x).prod() - 1)
        yearly_benchmark = self.results['benchmark_returns'].groupby(pd.Grouper(freq='Y')).apply(lambda x: (1 + x).prod() - 1)
        
        for year, ret in yearly_returns.items():
            bench_ret = yearly_benchmark.get(year, 0)
            print(f"{year.year}: Stratégie: {ret:.2%}, Benchmark: {bench_ret:.2%}, Alpha: {ret - bench_ret:.2%}")
        
        print("\nPOSITIONS ACTUELLES:")
        print("-"*50)
        
        # Positions longues actuelles
        long_positions = self.results['long_portfolio'].iloc[-1]
        long_tickers = long_positions[long_positions].index.tolist()
        print(f"Positions longues ({len(long_tickers)}): {', '.join(long_tickers)}")
        
        # Positions courtes actuelles
        short_positions = self.results['short_portfolio'].iloc[-1]
        short_tickers = short_positions[short_positions].index.tolist()
        print(f"Positions courtes ({len(short_tickers)}): {', '.join(short_tickers)}")
        
        print("\n" + "="*50)
    
    def save_results(self, directory='results'):
        """
        Sauvegarde les résultats du backtest.
        
        Parameters
        ----------
        directory : str, optional
            Répertoire où sauvegarder les résultats
            
        Returns
        -------
        str
            Chemin du répertoire contenant les résultats
        """
        if self.results is None:
            raise ValueError("Le backtest n'a pas encore été exécuté. Utilisez run() d'abord.")
        
        # Création du répertoire de résultats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(directory, f"backtest_{timestamp}")
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # Sauvegarde des données principales
        self.results['strategy_returns'].to_csv(os.path.join(result_dir, 'strategy_returns.csv'))
        self.results['benchmark_returns'].to_csv(os.path.join(result_dir, 'benchmark_returns.csv'))
        self.portfolio_values.to_csv(os.path.join(result_dir, 'portfolio_values.csv'))
        
        # Sauvegarde des métriques de performance
        pd.Series(self.performance_metrics).to_csv(os.path.join(result_dir, 'performance_metrics.csv'))
        
        # Sauvegarde des positions
        self.results['long_portfolio'].to_csv(os.path.join(result_dir, 'long_positions.csv'))
        self.results['short_portfolio'].to_csv(os.path.join(result_dir, 'short_positions.csv'))
        
        # Génération des visualisations
        self.plot_performance(save_path=os.path.join(result_dir, 'performance.png'))
        self.plot_portfolio_evolution(save_path=os.path.join(result_dir, 'portfolio_evolution.png'))
        self.plot_heat_map(save_path=os.path.join(result_dir, 'momentum_heatmap.png'))
        
        # Création d'un rapport texte
        with open(os.path.join(result_dir, 'backtest_report.txt'), 'w') as f:
            f.write("="*50 + "\n")
            f.write("RAPPORT DE BACKTEST\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Date du backtest: {timestamp}\n")
            f.write(f"Période: {self.start_date} à {self.end_date}\n")
            f.write(f"Capital initial: ${self.initial_capital:.2f}\n\n")
            
            f.write("MÉTRIQUES DE PERFORMANCE:\n")
            f.write("-"*50 + "\n")
            
            for metric, value in self.performance_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
            
            f.write("\nRÉSULTATS FINAUX:\n")
            f.write("-"*50 + "\n")
            
            final_value = self.portfolio_values['Strategy'].iloc[-1]
            benchmark_value = self.portfolio_values['Benchmark'].iloc[-1]
            
            f.write(f"Valeur finale du portefeuille: ${final_value:.2f}\n")
            f.write(f"Profit/Perte: ${final_value - self.initial_capital:.2f} ({(final_value / self.initial_capital - 1) * 100:.2f}%)\n")
            f.write(f"Valeur finale du benchmark: ${benchmark_value:.2f}\n")
            f.write(f"Performance relative: {(final_value / benchmark_value - 1) * 100:.2f}%\n")
        
        print(f"Résultats sauvegardés dans: {result_dir}")
        return result_dir


def parameter_optimization(strategy_class, assets, start_date, end_date, 
                          momentum_periods=range(10, 101, 10),
                          holding_days=range(1, 11),
                          quantile_pairs=[(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)],
                          price_types=['Close', 'Adj Close'],
                          initial_capital=10000):
    """
    Optimise les paramètres de la stratégie.
    
    Parameters
    ----------
    strategy_class : class
        Classe de la stratégie (e.g., MomentumStrategy)
    assets : list
        Liste des symboles d'actions
    start_date : str
        Date de début au format 'YYYY-MM-DD'
    end_date : str
        Date de fin au format 'YYYY-MM-DD'
    momentum_periods : iterable, optional
        Périodes de momentum à tester
    holding_days : iterable, optional
        Jours de détention à tester
    quantile_pairs : iterable, optional
        Paires de quantiles à tester
    price_types : iterable, optional
        Types de prix à tester
    initial_capital : float, optional
        Capital initial pour le backtest
        
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les résultats de l'optimisation
    """
    print(f"Optimisation des paramètres de {strategy_class.__name__}...")
    
    # Création de la liste des paramètres à tester
    param_combinations = [
        (m, h, q, p) 
        for m in momentum_periods 
        for h in holding_days 
        for q in quantile_pairs 
        for p in price_types
    ]
    
    # Initialisation du DataFrame de résultats
    results = []
    
    # Test de chaque combinaison de paramètres
    for momentum_period, holding_days, quantiles, price_type in tqdm(param_combinations, desc="Testing parameters"):
        # Initialisation de la stratégie
        strategy = strategy_class(assets)
        
        # Initialisation du backtest
        backtest = Backtest(strategy, start_date, end_date, initial_capital)
        
        try:
            # Exécution du backtest
            backtest_results = backtest.run(
                momentum_period=momentum_period,
                forward_return_days=holding_days,
                quantiles=quantiles,
                price_type=price_type
            )
            
            # Extraction des métriques clés
            metrics = backtest_results['performance_metrics']
            
            # Ajout des résultats
            results.append({
                'momentum_period': momentum_period,
                'holding_days': holding_days,
                'quantile_low': quantiles[0],
                'quantile_high': quantiles[1],
                'price_type': price_type,
                'sharpe_ratio': metrics['Sharpe Ratio'],
                'sortino_ratio': metrics['Sortino Ratio'],
                'annualized_return': metrics['Annualized Return'],
                'max_drawdown': metrics['Max Drawdown'],
                'win_rate': metrics['Win Rate'],
                'total_return': metrics['Total Return']
            })
            
        except Exception as e:
            print(f"Erreur avec les paramètres: {momentum_period}, {holding_days}, {quantiles}, {price_type}")
            print(f"Message d'erreur: {str(e)}")
            continue
    
    # Conversion en DataFrame
    results_df = pd.DataFrame(results)
    
    # Tri par ratio de Sharpe décroissant
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\nMeilleurs paramètres par ratio de Sharpe:")
    print(results_df.head(5))
    
    return results_df


if __name__ == "__main__":
    # Exemple d'utilisation
    from momentum_strategy import MomentumStrategy
    
    # Liste d'actifs
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'PG', 'V']
    
    # Initialisation de la stratégie
    strategy = MomentumStrategy(assets)
    
    # Période de backtest
    start_date = '2020-01-01'
    end_date = '2025-04-30'
    
    # Initialisation et exécution du backtest
    backtest = Backtest(strategy, start_date, end_date)
    results = backtest.run()
    
    # Affichage du résumé
    backtest.print_performance_summary()
    
    # Visualisation des performances
    backtest.plot_performance()
    plt.show()
