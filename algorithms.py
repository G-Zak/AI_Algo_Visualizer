# algorithms.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import make_blobs, make_classification
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

from style import (
    create_web_button,
    create_modern_button,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    CARD_COLOR,
    TEXT_COLOR,
    BORDER_COLOR,
    FONT_MEDIUM,
    FONT_NORMAL,
    HIGHLIGHT
)

class BaseTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style='Card.TFrame')
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Frame(self, style='Card.TFrame')
        header.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        tk.Label(
            header,
            text=self.tab_name,
            font=FONT_MEDIUM,
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR
        ).pack(side='left', padx=10)
        btn_frame = ttk.Frame(header, style='Card.TFrame')
        btn_frame.pack(side='right', padx=10)
        self.run_btn = create_modern_button(
            btn_frame, "Run Analysis", self.run_algorithm,
            bg=ACCENT_COLOR, fg="#005D2E" , padx=20, pady=6
        )
        self.run_btn.pack(side='left', padx=5)
        self.info_btn = create_modern_button(
            btn_frame, "Learn More", self.show_info,
            bg=SECONDARY_COLOR, fg="#005D2E" , padx=20, pady=6
        )
        self.info_btn.pack(side='left')
        # Output text
        output_frame = ttk.Frame(self, style='Card.TFrame')
        output_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0,10))
        output_frame.config(height=100)
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side='right', fill='y')
        self.output_text = tk.Text(
            output_frame, wrap='word', font=FONT_NORMAL,
            bg=CARD_COLOR, fg=TEXT_COLOR,
            padx=10, pady=10, yscrollcommand=scrollbar.set
        )
        self.output_text.pack(expand=True, fill='both')
        scrollbar.config(command=self.output_text.yview)
        # Plot area
        self.plot_frame = ttk.Frame(self, style='Card.TFrame')
        self.plot_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0,10))

    def clear_output(self):
        self.output_text.delete("1.0", "end")

    def show_info(self):
        webbrowser.open(self.info_url)

    def show_plot(self, fig):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        fig.patch.set_facecolor(CARD_COLOR)
        for ax in fig.axes:
            ax.set_facecolor(CARD_COLOR)
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOR)
            ax.title.set_color(TEXT_COLOR)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(alpha=0.2)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

class RegressionTab(BaseTab):
    tab_name = "Price Prediction"
    info_url = "https://en.wikipedia.org/wiki/Linear_regression"

    def run_algorithm(self):
        self.clear_output()
        np.random.seed(42)
        n = 200

        # Simuler des caractéristiques réalistes (ma5 = moyenne mobile, vol5 = volatilité)
        ma5 = np.random.normal(loc=100, scale=10, size=n)        # ex : prix moyen autour de 100
        vol5 = np.random.normal(loc=5, scale=2, size=n)          # ex : volatilité entre 3 et 7

        # Générer un "retour" réaliste basé sur ces features + bruit
        returns = 0.0008 * ma5 - 0.002 * vol5 + np.random.normal(0, 0.01, size=n)

        # Créer le DataFrame
        df = pd.DataFrame({
            'ma5': ma5,
            'vol5': vol5,
            'return': returns
        })

        # Séparation features / target
        X = df[['ma5', 'vol5']]
        y = df['return']

        # Régression linéaire
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        self.output_text.insert('end',
            f"Regression Results:\n"
            f"• R²: {r2:.4f}\n"
            f"• MSE: {mse:.6f}\n"
        )

        # Graphique amélioré
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(y, y_pred, color=PRIMARY_COLOR, alpha=0.7, s=40, label="Pred vs Actual")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color=HIGHLIGHT, label="Ideal")

        ax.set_title("Return Prediction", fontsize=13, weight='bold')
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False)

        self.show_plot(fig)

class ClusteringTab(BaseTab):
    tab_name = "Return Clustering"
    info_url = "https://en.wikipedia.org/wiki/K-means_clustering"

    def run_algorithm(self):
        self.clear_output()
        # Simulate returns for 5 synthetic stocks
        np.random.seed(42)
        n = 300
        returns = np.random.normal(loc=[0.001,0.0005,0,-0.0005,-0.001], scale=0.01, size=(n,5))
        df = pd.DataFrame(returns, columns=[f'S{i}' for i in range(1,6)])
        # Features: mean and volatility per sample across stocks
        X = pd.DataFrame({
            'mean_ret': df.mean(axis=1),
            'vol_ret' : df.std(axis=1)
        })
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        self.output_text.insert('end',
            f"Clustering Results:\n"
            f"• Clusters: 3\n"
            f"• Inertia: {inertia:.4f}\n"
        )
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(X['mean_ret'], X['vol_ret'], c=labels, cmap='viridis', s=40)
        ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                   c='red', s=100, marker='X')
        ax.set(title="Return Clusters", xlabel="Mean Return", ylabel="Volatility")
        self.show_plot(fig)

class ClassificationTab(BaseTab):
    tab_name = "Direction Classification"
    info_url = "https://en.wikipedia.org/wiki/Random_forest"

    def run_algorithm(self):
        self.clear_output()
        # Simulate features and up/down label
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'momentum': np.random.randn(n),
            'volatility': np.abs(np.random.randn(n))
        })
        y = (X['momentum'] > 0).astype(int)
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        self.output_text.insert('end',
            f"Classification Results:\n"
            f"• Accuracy: {acc:.2%}\n"
        )
        # Plot decision boundary
        x_min, x_max = X['momentum'].min(), X['momentum'].max()
        y_min, y_max = X['volatility'].min(), X['volatility'].max()
        xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X['momentum'], X['volatility'], c=y, edgecolor='k', s=40)
        ax.set(title="Momentum vs Volatility", xlabel="Momentum", ylabel="Volatility")
        self.show_plot(fig)

class TimeSeriesTab(BaseTab):
    tab_name = "Price Forecast"
    info_url = "https://en.wikipedia.org/wiki/ARIMA"

    def run_algorithm(self):
        self.clear_output()
        # Simulate stock price series
        np.random.seed(42)
        n = 100
        dt = 1/252
        mu, sigma = 0.08, 0.15
        prices = [50]
        for _ in range(n-1):
            prices.append(prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn()))
        prices = np.array(prices)
        model = ARIMA(prices, order=(2,1,2)).fit()
        forecast = model.predict(start=1, end=n-1)
        self.output_text.insert('end', "ARIMA model fitted.\n")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(prices, label="Actual", color=PRIMARY_COLOR)
        ax.plot(forecast, '--', label="Forecast", color=HIGHLIGHT)
        ax.set(title="Price Forecast", xlabel="Time", ylabel="Price")
        ax.legend(frameon=False)
        self.show_plot(fig)

class ValidationTab(BaseTab):
    tab_name = "Regression CV"
    info_url = "https://en.wikipedia.org/wiki/Cross-validation_(statistics)"

    def run_algorithm(self):
        self.clear_output()

        # Génération des données
        np.random.seed(42)
        n = 150
        X = pd.DataFrame({
            'ma10': np.random.rand(n),
            'vol10': np.random.rand(n)
        })
        y = 0.5*X['ma10'] - 0.3*X['vol10'] + np.random.randn(n)*0.2
        # Modèle
        model = LinearRegression()
        kf = KFold(5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

        # Texte dans la zone de sortie
        self.output_text.insert('end',
            f"Cross-Validation R² Scores:\n" +
            "\n".join(f"Fold {i+1}: {s:.3f}" for i, s in enumerate(scores)) +
            f"\nMean: {scores.mean():.3f}"
        )

        # Graphique stylisé
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        bars = ax.bar(range(1, 6), scores, color="#4CAF50", edgecolor="#2E7D32", width=0.5)

        # Ligne horizontale de la moyenne
        ax.axhline(scores.mean(), color="#FFA500", linestyle="--", linewidth=2, label="Mean R²")

        # Étiquettes
        ax.set_title("Cross-Validation R² Scores", fontsize=14, weight='bold', color="#2C3E50")
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_ylim(0, 1.05)

        # Valeur au-dessus de chaque barre
        for i, b in enumerate(bars):
            height = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, height + 0.02, f"{height:.2f}",
                    ha='center', fontsize=10, color="#333")

        ax.legend(frameon=False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        self.show_plot(fig)