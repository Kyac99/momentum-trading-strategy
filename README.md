# Stratégie de Trading Long-Short basée sur le Momentum

Cette plateforme implémente une stratégie de trading "long-short" basée sur le momentum des actions. La stratégie utilise le momentum sur 30 jours pour classer les actions et prendre des positions longues sur les actions avec le momentum le plus faible et des positions courtes sur les actions avec le momentum le plus élevé.

## Concept

La stratégie repose sur le principe de la réversion à la moyenne :
- Acheter (position longue) les actions ayant connu une baisse récente (faible momentum)
- Vendre (position courte) les actions ayant connu une hausse récente (fort momentum)

L'hypothèse est que les actions qui ont récemment sous-performé ont tendance à rebondir, tandis que celles qui ont surperformé ont tendance à corriger.

## Fonctionnalités

- Téléchargement de données historiques de prix d'actions via l'API Yahoo Finance
- Calcul du momentum sur différentes périodes
- Classement des actions selon leur momentum
- Construction de portefeuilles long-short basés sur des quintiles
- Gestion des risques et optimisation des positions
- Calcul des rendements de la stratégie
- Visualisation des performances
- Backtest complet avec métriques de performance
- Interface en ligne de commande
- Analyse interactive via notebook Jupyter

## Installation

```bash
git clone https://github.com/Kyac99/momentum-trading-strategy.git
cd momentum-trading-strategy
pip install -r requirements.txt
```

## Utilisation

### Via la ligne de commande

```bash
python run_strategy.py AAPL MSFT GOOGL AMZN META --momentum_period 30 --forward_days 5
```

### Via l'API Python

```python
from momentum_strategy import MomentumStrategy

# Initialiser la stratégie avec une liste d'actions
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'PG']
strategy = MomentumStrategy(assets)

# Exécuter le backtest
results = strategy.backtest(start_date='2020-01-01', 
                           end_date='2025-01-01',
                           momentum_period=30,
                           forward_return_days=5)

# Afficher les performances
strategy.plot_performance()
```

### Avec gestion des risques

```python
from momentum_strategy import MomentumStrategy
from risk_management import RiskManager

# Initialiser la stratégie
strategy = MomentumStrategy(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
strategy.download_data('2020-01-01', '2025-01-01')

# Configurer le gestionnaire de risques
risk_manager = RiskManager(strategy)
risk_manager.auto_detect_sectors()

# Créer le portefeuille
strategy.calculate_momentum(30)
strategy.calculate_forward_returns(5)
ranks = strategy.rank_assets()
long_portfolio, short_portfolio = strategy.create_long_short_portfolio(ranks)

# Optimiser les positions
long_weights, short_weights = risk_manager.optimize_position_sizes(long_portfolio, short_portfolio)

# Afficher les métriques de risque
risk_metrics = risk_manager.calculate_risk_metrics(strategy.strategy_returns)
print(risk_metrics)

# Visualiser le profil de risque
risk_manager.plot_risk_profile(strategy.strategy_returns)
```

## Structure du projet

- `momentum_strategy.py` : Classe principale implémentant la stratégie
- `data_loader.py` : Gestion du téléchargement et du traitement des données
- `backtest.py` : Fonctions de backtest
- `visualization.py` : Fonctions de visualisation
- `risk_management.py` : Gestion des risques et optimisation des positions
- `run_strategy.py` : Interface en ligne de commande
- `examples/` : Exemples d'utilisation
  - `example_stocks.py` : Exemple simple avec quelques actions
  - `reproduce_document_results.py` : Reproduction des résultats du document original
- `notebooks/` : Notebooks Jupyter pour l'analyse
- `tests/` : Tests unitaires
- `data/` : Dossier pour stocker les données (créé automatiquement)
- `results/` : Dossier pour stocker les résultats (créé automatiquement)

## Fonctionnalités avancées

### Backtest complet

```python
from backtest import Backtest
from momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
backtest = Backtest(strategy, '2020-01-01', '2025-01-01')
results = backtest.run()
backtest.print_performance_summary()
backtest.plot_performance()
```

### Optimisation des paramètres

```python
from backtest import parameter_optimization
from momentum_strategy import MomentumStrategy

assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
results_df = parameter_optimization(
    MomentumStrategy, 
    assets, 
    '2020-01-01', 
    '2025-01-01'
)
```

### Visualisations avancées

```python
from visualization import plot_momentum_heat_map, plot_correlation_matrix

# Heatmap des scores de momentum
heat_fig = plot_momentum_heat_map(momentum_scores)

# Matrice de corrélation des rendements
corr_fig = plot_correlation_matrix(returns)
```

### Analyse de risque

```python
from risk_management import RiskManager

# Analyse du profil de risque
risk_manager = RiskManager(strategy)
risk_metrics = risk_manager.calculate_risk_metrics(strategy_returns)
risk_fig = risk_manager.plot_risk_profile(strategy_returns)

# Exposition sectorielle
risk_manager.auto_detect_sectors()
sector_fig = risk_manager.plot_sector_exposure(weights)
```

## Tests

Pour exécuter les tests unitaires :

```bash
cd momentum-trading-strategy
python -m unittest discover tests
```

## Guide d'utilisation

Pour plus de détails sur l'utilisation de la plateforme, consultez le [Guide d'utilisation](GUIDE_UTILISATION.md).

## Résultats

Les tests ont montré que cette stratégie peut générer des rendements positifs dans diverses conditions de marché, particulièrement dans des périodes de volatilité élevée.

## Licence

[MIT](LICENSE)
