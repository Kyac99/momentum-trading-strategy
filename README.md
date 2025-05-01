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
- Calcul des rendements de la stratégie
- Visualisation des performances
- Backtest complet avec métriques de performance

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

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

## Structure du projet

- `momentum_strategy.py` : Classe principale implémentant la stratégie
- `data_loader.py` : Gestion du téléchargement et du traitement des données
- `portfolio.py` : Construction et gestion du portefeuille
- `visualization.py` : Fonctions de visualisation
- `backtest.py` : Fonctions de backtest
- `examples/` : Exemples d'utilisation
- `notebooks/` : Notebooks Jupyter pour l'analyse

## Résultats

Les tests ont montré que cette stratégie peut générer des rendements positifs dans diverses conditions de marché, particulièrement dans des périodes de volatilité élevée.

## Licence

MIT