#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement et le traitement des données financières.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm


class DataLoader:
    """
    Classe pour le chargement et le traitement des données financières.
    """
    
    def __init__(self, cache_dir='./data_cache'):
        """
        Initialise le chargeur de données.
        
        Parameters
        ----------
        cache_dir : str, optional
            Répertoire pour mettre en cache les données téléchargées
        """
        self.cache_dir = cache_dir
        
        # Création du répertoire de cache s'il n'existe pas
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def download_data(self, tickers, start_date, end_date, interval='1d', price_type='Close', use_cache=True):
        """
        Télécharge les données historiques de prix.
        
        Parameters
        ----------
        tickers : list
            Liste des symboles d'actions
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        interval : str, optional
            Intervalle des données ('1d', '1wk', '1mo', etc.)
        price_type : str, optional
            Type de prix à utiliser ('Open', 'High', 'Low', 'Close', 'Adj Close')
        use_cache : bool, optional
            Si True, utilise le cache local si les données existent
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les prix historiques
        """
        # Construction du nom de fichier pour le cache
        cache_file = os.path.join(
            self.cache_dir, 
            f"{'_'.join(sorted(tickers))}_{start_date}_{end_date}_{interval}.parquet"
        )
        
        # Vérification du cache
        if use_cache and os.path.exists(cache_file):
            try:
                data = pd.read_parquet(cache_file)
                print(f"Données chargées depuis le cache: {cache_file}")
                return data[price_type]
            except Exception as e:
                print(f"Erreur lors du chargement depuis le cache: {e}")
                print("Téléchargement des données...")
        
        # Téléchargement des données
        print(f"Téléchargement des données pour {len(tickers)} actions...")
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        
        # Sauvegarde dans le cache
        if use_cache:
            try:
                data.to_parquet(cache_file)
                print(f"Données sauvegardées dans le cache: {cache_file}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde dans le cache: {e}")
        
        return data[price_type]
    
    def download_data_chunked(self, tickers, start_date, end_date, chunk_size=20, price_type='Close'):
        """
        Télécharge les données historiques de prix par morceaux pour éviter les limitations de l'API.
        
        Parameters
        ----------
        tickers : list
            Liste des symboles d'actions
        start_date : str
            Date de début au format 'YYYY-MM-DD'
        end_date : str
            Date de fin au format 'YYYY-MM-DD'
        chunk_size : int, optional
            Nombre de tickers à télécharger en une seule fois
        price_type : str, optional
            Type de prix à utiliser ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns
        -------
        pandas.DataFrame
            DataFrame contenant les prix historiques
        """
        result = []
        
        # Traitement par morceaux
        for i in tqdm(range(0, len(tickers), chunk_size)):
            chunk = tickers[i:i+chunk_size]
            data = yf.download(chunk, start=start_date, end=end_date)[price_type]
            result.append(data)
            
        # Concaténation des résultats
        if len(result) == 1:
            return result[0]
        else:
            return pd.concat(result, axis=1)
    
    def preprocess_data(self, data, fillna_method='ffill', min_periods=None):
        """
        Prétraite les données financières.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Données à prétraiter
        fillna_method : str, optional
            Méthode pour remplir les valeurs manquantes ('ffill', 'bfill', 'interpolate')
        min_periods : int, optional
            Nombre minimum de valeurs non-NA pour qu'une colonne soit conservée
            
        Returns
        -------
        pandas.DataFrame
            Données prétraitées
        """
        # Copie des données pour éviter de modifier l'original
        processed = data.copy()
        
        # Suppression des colonnes avec trop de valeurs manquantes
        if min_periods is not None:
            non_na_counts = processed.count()
            valid_columns = non_na_counts[non_na_counts >= min_periods].index
            processed = processed[valid_columns]
        
        # Remplissage des valeurs manquantes
        if fillna_method == 'ffill':
            processed = processed.fillna(method='ffill')
        elif fillna_method == 'bfill':
            processed = processed.fillna(method='bfill')
        elif fillna_method == 'interpolate':
            processed = processed.interpolate(method='linear')
        
        # Suppression des lignes avec des valeurs manquantes restantes
        processed = processed.dropna()
        
        return processed
    
    def calculate_returns(self, prices, periods=(1, 5, 20, 60, 120, 252)):
        """
        Calcule les rendements sur différentes périodes.
        
        Parameters
        ----------
        prices : pandas.DataFrame
            Données de prix
        periods : tuple, optional
            Périodes pour lesquelles calculer les rendements
            
        Returns
        -------
        dict
            Dictionnaire contenant les rendements pour chaque période
        """
        returns = {}
        
        for period in periods:
            period_name = f"{period}d"
            returns[period_name] = prices.pct_change(periods=period)
        
        return returns
    
    def calculate_momentum(self, prices, periods=(30, 60, 90, 180, 252)):
        """
        Calcule les scores de momentum sur différentes périodes.
        
        Parameters
        ----------
        prices : pandas.DataFrame
            Données de prix
        periods : tuple, optional
            Périodes pour lesquelles calculer le momentum
            
        Returns
        -------
        dict
            Dictionnaire contenant les scores de momentum pour chaque période
        """
        momentum = {}
        
        for period in periods:
            period_name = f"{period}d"
            momentum[period_name] = (prices - prices.shift(period)) / prices.shift(period)
        
        return momentum
    
    def calculate_volatility(self, prices, windows=(20, 60, 120)):
        """
        Calcule la volatilité sur différentes fenêtres.
        
        Parameters
        ----------
        prices : pandas.DataFrame
            Données de prix
        windows : tuple, optional
            Fenêtres pour lesquelles calculer la volatilité
            
        Returns
        -------
        dict
            Dictionnaire contenant la volatilité pour chaque fenêtre
        """
        # Calcul des rendements quotidiens
        returns = prices.pct_change().dropna()
        
        volatility = {}
        
        for window in windows:
            window_name = f"{window}d"
            # Volatilité = écart-type des rendements sur la fenêtre (annualisé)
            volatility[window_name] = returns.rolling(window=window).std() * np.sqrt(252)
        
        return volatility
    
    def download_benchmark_data(self, benchmark_ticker='SPY', start_date=None, end_date=None):
        """
        Télécharge les données du benchmark pour comparaison.
        
        Parameters
        ----------
        benchmark_ticker : str, optional
            Symbole du benchmark
        start_date : str, optional
            Date de début au format 'YYYY-MM-DD'
        end_date : str, optional
            Date de fin au format 'YYYY-MM-DD'
            
        Returns
        -------
        pandas.Series
            Série contenant les prix du benchmark
        """
        # Utilisation de la période par défaut si non spécifiée
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
        
        return data


if __name__ == "__main__":
    # Exemple d'utilisation
    loader = DataLoader()
    
    # Liste d'actions du S&P 500
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'JNJ', 'JPM', 'PG', 'UNH']
    
    # Période: 2 ans
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    # Téléchargement des données
    prices = loader.download_data(tickers, start_date, end_date)
    
    # Prétraitement
    prices = loader.preprocess_data(prices)
    
    # Calcul des rendements et du momentum
    returns_dict = loader.calculate_returns(prices)
    momentum_dict = loader.calculate_momentum(prices)
    
    print("Données téléchargées et traitées avec succès.")
    print(f"Dimensions des données: {prices.shape}")
    print(f"Périodes de momentum calculées: {list(momentum_dict.keys())}")
