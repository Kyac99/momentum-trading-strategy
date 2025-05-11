# Guide d'Utilisation de la Plateforme de Trading Long-Short

Ce guide explique comment installer, configurer et utiliser la plateforme de trading long-short basée sur le momentum que nous avons développée.

## Table des matières

1. [Installation](#1-installation)
2. [Structure du projet](#2-structure-du-projet)
3. [Utilisation en ligne de commande](#3-utilisation-en-ligne-de-commande)
4. [Utilisation via le notebook Jupyter](#4-utilisation-via-le-notebook-jupyter)
5. [Reproduction des résultats du document](#5-reproduction-des-résultats-du-document)
6. [Optimisation des paramètres](#6-optimisation-des-paramètres)
7. [Personnalisation avancée](#7-personnalisation-avancée)
8. [Aide supplémentaire](#8-aide-supplémentaire)

## 1. Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/Kyac99/momentum-trading-strategy.git
cd momentum-trading-strategy
pip install -r requirements.txt
```

## 2. Structure du projet

Le projet est organisé comme suit :

- `momentum_strategy.py` : Classe principale implémentant la stratégie
- `data_loader.py` : Module pour le chargement de données financières
- `backtest.py` : Module pour le backtesting de la stratégie
- `visualization.py` : Module pour la visualisation des résultats
- `run_strategy.py` : Interface en ligne de commande
- `examples/` : Exemples d'utilisation
  - `example_stocks.py` : Exemple simple avec quelques actions
  - `reproduce_document_results.py` : Reproduction des résultats du document
- `notebooks/` : Notebooks Jupyter pour l'analyse interactive
- `data/` : Dossier pour stocker les données (créé automatiquement)
- `results/` : Dossier pour stocker les résultats (créé automatiquement)

## 3. Utilisation en ligne de commande

Le script `run_strategy.py` fournit une interface en ligne de commande complète pour exécuter la stratégie.

### Exemples d'utilisation

**Exemple 1** : Stratégie de base avec quelques actions

```bash
python run_strategy.py AAPL MSFT GOOGL AMZN META --momentum_period 30 --forward_days 5
```

**Exemple 2** : Utilisation d'un modèle de portfolio prédéfini

```bash
python run_strategy.py --template tech_leaders --start_date 2024-01-01 --end_date 2025-04-30
```

**Exemple 3** : Utilisation d'une industrie spécifique

```bash
python run_strategy.py --industry Financial --start_date 2024-01-01 --end_date 2025-04-30
```

**Exemple 4** : Sauvegarde des résultats

```bash
python run_strategy.py AAPL MSFT GOOGL AMZN META --save_results --results_dir mes_resultats
```

**Exemple 5** : Optimisation des paramètres

```bash
python run_strategy.py AAPL MSFT GOOGL AMZN META --optimize
```

### Options disponibles

Voici la liste complète des options disponibles :

```
positional arguments:
  assets                Liste des symboles d'actions (e.g., AAPL MSFT GOOGL)

optional arguments:
  -h, --help            show this help message and exit
  --start_date START_DATE
                        Date de début au format YYYY-MM-DD (défaut: 1 an avant aujourd'hui)
  --end_date END_DATE   Date de fin au format YYYY-MM-DD (défaut: aujourd'hui)
  --momentum_period MOMENTUM_PERIOD
                        Période pour le calcul du momentum en jours (défaut: 30)
  --forward_days FORWARD_DAYS
                        Jours pour calculer les rendements futurs (défaut: 5)
  --quantile_low QUANTILE_LOW
                        Quantile bas pour les positions longues (défaut: 0.2)
  --quantile_high QUANTILE_HIGH
                        Quantile haut pour les positions courtes (défaut: 0.8)
  --price_type {Open,High,Low,Close,Adj Close}
                        Type de prix à utiliser (défaut: Close)
  --transaction_cost TRANSACTION_COST
                        Coût de transaction en pourcentage (défaut: 0.001)
  --initial_capital INITIAL_CAPITAL
                        Capital initial pour le backtest (défaut: 10000)
  --benchmark BENCHMARK
                        Symbole du benchmark (défaut: SPY)
  --cache_dir CACHE_DIR
                        Répertoire pour stocker les données en cache (défaut: data/cache)
  --results_dir RESULTS_DIR
                        Répertoire pour sauvegarder les résultats (défaut: results)
  --save_results        Sauvegarder les résultats du backtest
  --no_plots            Ne pas afficher les graphiques
  --template {tech_leaders,value_stocks,growth_stocks,dividend_stocks,momentum_stocks,etf_portfolio}
                        Utiliser un modèle de portfolio prédéfini à la place des symboles fournis
  --industry {Technology,Healthcare,Financial,Consumer,Industrial,Energy}
                        Utiliser tous les symboles d'une industrie spécifique
  --optimize            Exécuter une optimisation des paramètres
```

## 4. Utilisation via le notebook Jupyter

Le notebook Jupyter `notebooks/momentum_strategy_analysis.ipynb` fournit une interface interactive pour analyser la stratégie.

Pour lancer le notebook :

```bash
cd momentum-trading-strategy
jupyter notebook notebooks/momentum_strategy_analysis.ipynb
```

Le notebook est organisé en sections claires qui vous guident à travers l'analyse :

1. Configuration et chargement des modules
2. Définition des paramètres de la stratégie
3. Téléchargement et préparation des données
4. Calcul du momentum et des rendements futurs
5. Analyse de la corrélation entre momentum et rendements futurs
6. Mise en œuvre de la stratégie long-short
7. Calcul et analyse des rendements de la stratégie
8. Backtest complet avec benchmark
9. Analyse de sensibilité aux paramètres
10. Test sur différents univers d'actions
11. Conclusion et observations

## 5. Reproduction des résultats du document

Pour reproduire les résultats présentés dans le document original, utilisez l'exemple `examples/reproduce_document_results.py` :

```bash
python examples/reproduce_document_results.py
```

Ce script :
- Utilise les mêmes actifs que dans le document (BA, PEP, LMT, AAPL, JNJ, JPM, BAC, LLY, CAR)
- Utilise la même période (2024-12-01 à 2025-04-30)
- Utilise les mêmes paramètres (momentum sur 30 jours, rendements futurs sur 5 jours, quantiles à 20% et 80%)

Le script affiche également des analyses supplémentaires qui n'étaient pas présentes dans le document original :
- Analyse de sensibilité aux paramètres
- Test sur différentes périodes
- Analyse sectorielle

## 6. Optimisation des paramètres

Pour trouver les meilleurs paramètres pour votre univers d'actions, vous pouvez utiliser l'option `--optimize` :

```bash
python run_strategy.py AAPL MSFT GOOGL AMZN META --optimize
```

L'optimisation teste différentes combinaisons de :
- Périodes de momentum (5, 10, 20, 30, 60, 90 jours)
- Jours de détention (1, 2, 3, 5, 10, 15 jours)
- Paires de quantiles ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))
- Types de prix ('Close', 'Adj Close')

Les résultats de l'optimisation sont sauvegardés dans un fichier CSV et les meilleurs paramètres sont affichés.

## 7. Personnalisation avancée

### Création d'une nouvelle stratégie

Vous pouvez étendre la classe `MomentumStrategy` pour créer votre propre variante de la stratégie :

```python
from momentum_strategy import MomentumStrategy

class MyCustomStrategy(MomentumStrategy):
    def __init__(self, assets):
        super().__init__(assets)
        
    def calculate_momentum(self, period=30):
        # Votre méthode personnalisée de calcul du momentum
        pass
        
    def create_long_short_portfolio(self, ranks, quantiles=(0.2, 0.8)):
        # Votre méthode personnalisée de création de portefeuille
        pass
```

### Intégration avec d'autres sources de données

Vous pouvez étendre la classe `DataLoader` pour intégrer d'autres sources de données :

```python
from data_loader import DataLoader

class MyCustomDataLoader(DataLoader):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir)
        
    def download_from_custom_source(self, symbols, start_date, end_date):
        # Votre méthode personnalisée de téléchargement
        pass
```

## 8. Aide supplémentaire

Si vous avez des questions ou rencontrez des problèmes, n'hésitez pas à :

1. Consulter les docstrings détaillées dans le code source
2. Examiner les exemples dans le dossier `examples/`
3. Explorer le notebook Jupyter pour comprendre la logique de la stratégie
4. Ouvrir une issue sur GitHub pour obtenir de l'aide

---

Bonne trading ! 📈