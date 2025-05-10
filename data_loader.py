#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement et le prétraitement des données financières.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
from tqdm import tqdm


class DataLoader:
    """
    Classe pour charger et prétraiter des données financières à partir de différentes sources.
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialise le chargeur de données.
        
        Parameters
        ----------
        cache_dir : str, optional
            Répertoire pour stocker les données en cache
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def download_from_yahoo(self, symbols, start_date, end_date, interval='1d', price_type='Close', 
                           use_cache=True, cache_filename=None):
        """
        Télécharge les données historiques depuis Yahoo Finance.
        
        Parameters
        ----------
        symbols : list
            Liste des symboles d'actions
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        interval : str, optional
            Intervalle des données ('1d', '1wk', '1mo')
        price_type : str, optional
            Type de prix ('Open', 'High', 'Low', 'Close', 'Adj Close')
        use_cache : bool, optional
            Utiliser le cache si disponible
        cache_filename : str, optional
            Nom du fichier de cache
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les prix historiques
        """
        # Création du nom de fichier cache si non spécifié
        if use_cache and self.cache_dir and not cache_filename:
            symbols_str = '-'.join(sorted(symbols))
            cache_filename = f"{symbols_str}_{start_date}_{end_date}_{interval}.csv"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            # Vérification si le cache existe
            if os.path.exists(cache_path):
                print(f"Chargement des données depuis le cache: {cache_path}")
                data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return data
        
        # Téléchargement des données
        print(f"Téléchargement des données pour {len(symbols)} symboles de {start_date} à {end_date}...")
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
        
        # Sélection du type de prix
        if price_type in data.columns.levels[0]:
            data = data[price_type]
        else:
            raise ValueError(f"Type de prix {price_type} non disponible. Choix disponibles: {data.columns.levels[0].tolist()}")
        
        # Sauvegarde en cache si nécessaire
        if use_cache and self.cache_dir and cache_filename:
            cache_path = os.path.join(self.cache_dir, cache_filename)
            data.to_csv(cache_path)
            print(f"Données sauvegardées en cache: {cache_path}")
        
        return data
    
    def load_from_csv(self, file_path, date_column=None, symbol_column=None):
        """
        Charge des données à partir d'un fichier CSV.
        
        Parameters
        ----------
        file_path : str
            Chemin vers le fichier CSV
        date_column : str, optional
            Nom de la colonne contenant les dates
        symbol_column : str, optional
            Nom de la colonne contenant les symboles
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les données
        """
        # Chargement des données
        data = pd.read_csv(file_path)
        
        # Conversion des dates si nécessaire
        if date_column:
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
        
        # Pivot de la table si nécessaire (format long vers format large)
        if symbol_column:
            data = data.pivot(columns=symbol_column)
        
        return data
    
    def get_benchmark_returns(self, benchmark_symbol, start_date, end_date, interval='1d'):
        """
        Télécharge les rendements d'un benchmark.
        
        Parameters
        ----------
        benchmark_symbol : str
            Symbole du benchmark (e.g., 'SPY', '^GSPC')
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        interval : str, optional
            Intervalle des données ('1d', '1wk', '1mo')
            
        Returns
        -------
        pandas.Series
            Série contenant les rendements du benchmark
        """
        # Téléchargement des données du benchmark
        benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date, interval=interval)
        
        # Calcul des rendements
        benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
        
        return benchmark_returns
    
    def get_industry_symbols(self, industry, min_market_cap=None, max_symbols=None):
        """
        Obtient une liste de symboles d'actions pour une industrie spécifique.
        
        Parameters
        ----------
        industry : str
            Nom de l'industrie (e.g., 'Technology', 'Healthcare')
        min_market_cap : float, optional
            Capitalisation boursière minimale en milliards de dollars
        max_symbols : int, optional
            Nombre maximum de symboles à renvoyer
            
        Returns
        -------
        list
            Liste des symboles d'actions
        """
        # Note: Cette fonction est actuellement une implémentation simplifiée
        # Elle pourrait être améliorée en utilisant des API comme Alpha Vantage ou IEX
        
        # Exemple d'industries et de leurs tickers
        industries = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'INTC', 'CSCO', 'ORCL', 'IBM'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA'],
            'Consumer': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'TGT'],
            'Industrial': ['BA', 'GE', 'HON', 'UNP', 'CAT', 'LMT', 'RTX', 'MMM', 'UPS', 'FDX'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'OXY', 'KMI', 'WMB']
        }
        
        if industry not in industries:
            raise ValueError(f"Industrie {industry} non disponible. Choix disponibles: {list(industries.keys())}")
        
        symbols = industries[industry]
        
        # Filtrage par capitalisation boursière si nécessaire
        if min_market_cap:
            filtered_symbols = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if 'marketCap' in info and info['marketCap'] / 1e9 >= min_market_cap:
                        filtered_symbols.append(symbol)
                except:
                    continue
            symbols = filtered_symbols
        
        # Limitation du nombre de symboles si nécessaire
        if max_symbols and len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
        
        return symbols
    
    def get_index_constituents(self, index_symbol):
        """
        Obtient la liste des constituants d'un indice boursier.
        
        Parameters
        ----------
        index_symbol : str
            Symbole de l'indice (e.g., '^GSPC' pour S&P 500, '^NDX' pour Nasdaq-100)
            
        Returns
        -------
        list
            Liste des constituants de l'indice
        """
        # Indices les plus courants et leurs constituants
        indices = {
            '^GSPC': 'S&P 500',
            '^NDX': 'Nasdaq-100',
            '^DJI': 'Dow Jones Industrial Average',
            '^RUT': 'Russell 2000'
        }
        
        if index_symbol not in indices:
            raise ValueError(f"Indice {index_symbol} non disponible. Choix disponibles: {list(indices.keys())}")
        
        # Cette fonction utiliserait normalement une API pour obtenir les constituants
        # Pour l'instant, retournons quelques constituants communs comme exemple
        
        constituents = {
            '^GSPC': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ', 'V', 
                     'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'CMCSA', 'KO'],
            '^NDX': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'ADBE', 'PYPL', 'INTC',
                    'CMCSA', 'NFLX', 'PEP', 'CSCO', 'AVGO', 'COST', 'TXN', 'TMUS', 'QCOM', 'AMGN'],
            '^DJI': ['AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'CAT', 'V', 'MCD', 'CRM', 'HON',
                    'BA', 'MMM', 'JNJ', 'AXP', 'JPM', 'PG', 'TRV', 'CVX', 'WMT', 'IBM'],
            '^RUT': ['ENPH', 'CROX', 'GME', 'AMC', 'PLUG', 'SMCI', 'CELH', 'ETSY', 'TRIP', 'PZZA',
                    'SFIX', 'APPF', 'IRBT', 'IIPR', 'APPS', 'TRUE', 'CARG', 'HUBS', 'FIVN', 'NTRA']
        }
        
        return constituents[index_symbol]
    
    def create_portfolio_from_template(self, template, start_date, end_date, price_type='Close'):
        """
        Crée un portfolio à partir d'un modèle prédéfini.
        
        Parameters
        ----------
        template : str
            Nom du modèle ('tech_leaders', 'value_stocks', 'growth_stocks', 'dividend_stocks')
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        price_type : str, optional
            Type de prix ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les prix du portfolio
        """
        # Modèles de portfolio prédéfinis
        templates = {
            'tech_leaders': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'CRM', 'INTC'],
            'value_stocks': ['BRK-B', 'JPM', 'JNJ', 'PG', 'KO', 'VZ', 'T', 'IBM', 'CVX', 'XOM'],
            'growth_stocks': ['TSLA', 'NVDA', 'SHOP', 'SQ', 'ZM', 'ROKU', 'CRWD', 'DDOG', 'NET', 'SNOW'],
            'dividend_stocks': ['JNJ', 'PG', 'KO', 'PEP', 'VZ', 'T', 'MO', 'PM', 'O', 'ABBV'],
            'momentum_stocks': ['AAPL', 'NVDA', 'AMD', 'TSLA', 'SHOP', 'SQ', 'ROKU', 'ZM', 'SPOT', 'ETSY'],
            'etf_portfolio': ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'GLD', 'TLT', 'LQD', 'VNQ']
        }
        
        if template not in templates:
            raise ValueError(f"Modèle {template} non disponible. Choix disponibles: {list(templates.keys())}")
        
        symbols = templates[template]
        
        # Téléchargement des données
        data = self.download_from_yahoo(symbols, start_date, end_date, price_type=price_type)
        
        return data


if __name__ == "__main__":
    # Exemple d'utilisation
    loader = DataLoader(cache_dir='data/cache')
    
    # Dates de test
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Téléchargement d'un portfolio de leaders technologiques
    tech_data = loader.create_portfolio_from_template('tech_leaders', start_date, end_date)
    print("\nAperçu des données technologiques:")
    print(tech_data.head())
    
    # Téléchargement des rendements d'un benchmark
    spy_returns = loader.get_benchmark_returns('SPY', start_date, end_date)
    print("\nAperçu des rendements du S&P 500:")
    print(spy_returns.head())
