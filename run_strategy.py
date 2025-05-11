#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'interface en ligne de commande pour exécuter la stratégie de trading.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from momentum_strategy import MomentumStrategy
from data_loader import DataLoader
from backtest import Backtest
from visualization import plot_strategy_performance


def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns
    -------
    argparse.Namespace
        Arguments analysés
    """
    parser = argparse.ArgumentParser(description='Exécuter la stratégie de trading long-short basée sur le momentum')
    
    # Arguments obligatoires
    parser.add_argument('assets', type=str, nargs='+',
                        help='Liste des symboles d\'actions (e.g., AAPL MSFT GOOGL)')
    
    # Arguments optionnels
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Date de début au format YYYY-MM-DD (défaut: 1 an avant aujourd\'hui)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='Date de fin au format YYYY-MM-DD (défaut: aujourd\'hui)')
    parser.add_argument('--momentum_period', type=int, default=30,
                        help='Période pour le calcul du momentum en jours (défaut: 30)')
    parser.add_argument('--forward_days', type=int, default=5,
                        help='Jours pour calculer les rendements futurs (défaut: 5)')
    parser.add_argument('--quantile_low', type=float, default=0.2,
                        help='Quantile bas pour les positions longues (défaut: 0.2)')
    parser.add_argument('--quantile_high', type=float, default=0.8,
                        help='Quantile haut pour les positions courtes (défaut: 0.8)')
    parser.add_argument('--price_type', type=str, default='Close', choices=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                        help='Type de prix à utiliser (défaut: Close)')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                        help='Coût de transaction en pourcentage (défaut: 0.001)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='Capital initial pour le backtest (défaut: 10000)')
    parser.add_argument('--benchmark', type=str, default='SPY',
                        help='Symbole du benchmark (défaut: SPY)')
    parser.add_argument('--cache_dir', type=str, default='data/cache',
                        help='Répertoire pour stocker les données en cache (défaut: data/cache)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Répertoire pour sauvegarder les résultats (défaut: results)')
    parser.add_argument('--save_results', action='store_true',
                        help='Sauvegarder les résultats du backtest')
    parser.add_argument('--no_plots', action='store_true',
                        help='Ne pas afficher les graphiques')
    parser.add_argument('--template', type=str, choices=['tech_leaders', 'value_stocks', 'growth_stocks', 'dividend_stocks', 'momentum_stocks', 'etf_portfolio'],
                        help='Utiliser un modèle de portfolio prédéfini à la place des symboles fournis')
    parser.add_argument('--industry', type=str, choices=['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy'],
                        help='Utiliser tous les symboles d\'une industrie spécifique')
    parser.add_argument('--optimize', action='store_true',
                        help='Exécuter une optimisation des paramètres')
    
    return parser.parse_args()


def run_strategy(args):
    """
    Exécute la stratégie de trading avec les arguments fournis.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments de ligne de commande
    """
    print(f"\n{'-'*80}")
    print(f"EXÉCUTION DE LA STRATÉGIE LONG-SHORT BASÉE SUR LE MOMENTUM")
    print(f"{'-'*80}\n")
    
    # Création du répertoire de cache si nécessaire
    if args.cache_dir and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    # Création du répertoire de résultats si nécessaire
    if args.results_dir and not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Obtention des symboles d'actions
    assets = args.assets
    
    # Si un modèle de portfolio est spécifié
    if args.template:
        loader = DataLoader(cache_dir=args.cache_dir)
        data = loader.create_portfolio_from_template(args.template, args.start_date, args.end_date)
        assets = data.columns.tolist()
        print(f"Utilisation du modèle de portfolio '{args.template}' avec {len(assets)} symboles: {', '.join(assets)}")
    
    # Si une industrie est spécifiée
    if args.industry:
        loader = DataLoader(cache_dir=args.cache_dir)
        assets = loader.get_industry_symbols(args.industry)
        print(f"Utilisation des symboles de l'industrie '{args.industry}': {', '.join(assets)}")
    
    # Création de l'instance de la stratégie
    strategy = MomentumStrategy(assets)
    
    # Téléchargement des données
    data = strategy.download_data(args.start_date, args.end_date, args.price_type)
    
    # Exécution du backtest
    print(f"\nExécution du backtest avec les paramètres suivants:")
    print(f"- Période de momentum: {args.momentum_period} jours")
    print(f"- Jours de rendement futurs: {args.forward_days} jours")
    print(f"- Quantiles: {args.quantile_low} (bas) et {args.quantile_high} (haut)")
    print(f"- Coût de transaction: {args.transaction_cost*100:.2f}%")
    print(f"- Capital initial: ${args.initial_capital:.2f}")
    
    backtest = Backtest(strategy, args.start_date, args.end_date, args.initial_capital)
    results = backtest.run(
        momentum_period=args.momentum_period,
        forward_return_days=args.forward_days,
        quantiles=(args.quantile_low, args.quantile_high),
        price_type=args.price_type,
        benchmark_symbol=args.benchmark,
        transaction_cost=args.transaction_cost
    )
    
    # Affichage du résumé des performances
    backtest.print_performance_summary()
    
    # Sauvegarde des résultats si demandé
    if args.save_results:
        result_dir = backtest.save_results(args.results_dir)
        print(f"\nRésultats sauvegardés dans: {result_dir}")
    
    # Affichage des graphiques si demandé
    if not args.no_plots:
        print("\nAffichage des graphiques...")
        
        # Visualisation des performances
        backtest.plot_performance()
        plt.figure()
        
        # Visualisation de l'évolution du portefeuille
        backtest.plot_portfolio_evolution()
        plt.figure()
        
        # Visualisation des scores de momentum
        backtest.plot_heat_map()
        
        plt.show()
    
    return results


def optimize_strategy(args):
    """
    Exécute une optimisation des paramètres de la stratégie.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments de ligne de commande
    """
    from backtest import parameter_optimization
    
    print(f"\n{'-'*80}")
    print(f"OPTIMISATION DES PARAMÈTRES DE LA STRATÉGIE")
    print(f"{'-'*80}\n")
    
    # Obtention des symboles d'actions (même logique que run_strategy)
    assets = args.assets
    
    if args.template:
        loader = DataLoader(cache_dir=args.cache_dir)
        data = loader.create_portfolio_from_template(args.template, args.start_date, args.end_date)
        assets = data.columns.tolist()
        print(f"Utilisation du modèle de portfolio '{args.template}' avec {len(assets)} symboles: {', '.join(assets)}")
    
    if args.industry:
        loader = DataLoader(cache_dir=args.cache_dir)
        assets = loader.get_industry_symbols(args.industry)
        print(f"Utilisation des symboles de l'industrie '{args.industry}': {', '.join(assets)}")
    
    # Paramètres pour l'optimisation
    momentum_periods = [5, 10, 20, 30, 60, 90]
    holding_days = [1, 2, 3, 5, 10, 15]
    quantile_pairs = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    price_types = ['Close', 'Adj Close']
    
    print(f"\nExécution de l'optimisation avec les paramètres suivants:")
    print(f"- Périodes de momentum: {momentum_periods}")
    print(f"- Jours de détention: {holding_days}")
    print(f"- Paires de quantiles: {quantile_pairs}")
    print(f"- Types de prix: {price_types}")
    print(f"- Nombre total de combinaisons: {len(momentum_periods) * len(holding_days) * len(quantile_pairs) * len(price_types)}")
    
    # Exécution de l'optimisation
    results_df = parameter_optimization(
        MomentumStrategy, 
        assets, 
        args.start_date, 
        args.end_date, 
        momentum_periods=momentum_periods,
        holding_days=holding_days,
        quantile_pairs=quantile_pairs,
        price_types=price_types,
        initial_capital=args.initial_capital
    )
    
    # Sauvegarde des résultats
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.results_dir, f"optimization_results_{timestamp}.csv")
    results_df.to_csv(results_path)
    print(f"\nRésultats d'optimisation sauvegardés dans: {results_path}")
    
    # Affichage des meilleurs paramètres
    print("\nMeilleurs paramètres par ratio de Sharpe:")
    print(results_df.head(10)[['momentum_period', 'holding_days', 'quantile_low', 'quantile_high', 'price_type', 'sharpe_ratio', 'annualized_return']])
    
    # Visualisation des résultats si demandé
    if not args.no_plots:
        print("\nAffichage des graphiques d'optimisation...")
        
        # Graphique des ratios de Sharpe par période de momentum
        plt.figure(figsize=(15, 7))
        sharpe_by_period = results_df.groupby('momentum_period')['sharpe_ratio'].mean()
        sharpe_by_period.plot(kind='bar')
        plt.title('Ratio de Sharpe Moyen par Période de Momentum')
        plt.ylabel('Ratio de Sharpe')
        plt.grid(True, axis='y')
        
        # Graphique des rendements annualisés par jour de détention
        plt.figure(figsize=(15, 7))
        returns_by_days = results_df.groupby('holding_days')['annualized_return'].mean()
        returns_by_days.plot(kind='bar')
        plt.title('Rendement Annualisé Moyen par Jour de Détention')
        plt.ylabel('Rendement Annualisé')
        plt.grid(True, axis='y')
        
        # Heatmap des ratios de Sharpe par période et jour de détention
        plt.figure(figsize=(15, 10))
        heatmap_data = results_df.pivot_table(
            values='sharpe_ratio', 
            index='momentum_period', 
            columns='holding_days', 
            aggfunc='mean'
        )
        import seaborn as sns
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Ratio de Sharpe par Période de Momentum et Jour de Détention')
        plt.tight_layout()
        
        plt.show()
    
    return results_df


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.optimize:
        results = optimize_strategy(args)
    else:
        results = run_strategy(args)
