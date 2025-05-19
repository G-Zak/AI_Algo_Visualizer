import os
import webbrowser
from ttkthemes import ThemedTk
import tkinter as tk
from tkinter import ttk

from style import (
    apply_style,
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    CARD_COLOR,
    TEXT_COLOR,
    FONT_LARGE,
    FONT_MEDIUM,
    FONT_NORMAL
)
from algorithms import (
    RegressionTab,
    ClusteringTab,
    ClassificationTab,
    TimeSeriesTab,
    ValidationTab
)

class AIApp:
    def __init__(self, root):
        self.root = root
        self._build_ui()

    def _build_ui(self):
        # ----- Header -----
        header = tk.Frame(self.root, bg="#005D2E", height=60)
        header.pack(fill='x')
        tk.Label(
            header,
            text="EMSI Finance Explorer",
            font=FONT_LARGE,
            bg="#005D2E",
            fg=CARD_COLOR
        ).pack(padx=20, pady=10, anchor='w')

        # ----- Notebook (Tabs) -----
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both', padx=20, pady=20)

        # Algorithm tabs
        tabs = [
            ("Price Prediction", RegressionTab),
            ("Return Clustering", ClusteringTab),
            ("Direction Classification", ClassificationTab),
            ("Price Forecast", TimeSeriesTab),
            ("Regression CV", ValidationTab)
        ]
        for title, TabClass in tabs:
            tab = TabClass(notebook)
            notebook.add(tab, text=title)

        # ----- About Tab -----
        about = ttk.Frame(notebook, style='Card.TFrame')
        notebook.add(about, text="About")
        self.setup_about_tab(about)

    def setup_about_tab(self, frame):
        frame.configure(style="Card.TFrame")

        # EMSI Logo
        logo_path = os.path.join(os.path.dirname(__file__), "emsiLogo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image, ImageTk
                pil_img = Image.open(logo_path)
                pil_img = pil_img.resize((550, 100), Image.Resampling.LANCZOS)
                logo_img = ImageTk.PhotoImage(pil_img)
            except ImportError:
                logo_img = tk.PhotoImage(file=logo_path)
            logo_lbl = tk.Label(frame, image=logo_img, bg=CARD_COLOR)
            logo_lbl.image = logo_img
            logo_lbl.pack(pady=(20, 10))

        # 🔷 Titre centré
        tk.Label(
            frame,
            text="EMSI Finance Explorer – À propos",
            font=FONT_LARGE,
            bg=CARD_COLOR,
            fg="#005D2E",
            justify="center"
        ).pack(pady=(10, 5))

        # 🔷 Infos personnelles centrées
        info_lines = [
            "👨‍💻 Zakaria GUENNANI – Étudiant ingénieur (3e année EMSI)",
            "👩‍🏫 Encadrante : Dr. Mouna El Mkhalet",
            "✉️ Email : guennanizakaria69@gmail.com",
            "🔗 GitHub : https://github.com/G-Zak"
        ]
        for line in info_lines:
            tk.Label(
                frame,
                text=line,
                font=FONT_MEDIUM,
                bg=CARD_COLOR,
                fg=TEXT_COLOR,
                justify='center'
            ).pack(pady=2)

        # Séparateur horizontal
        ttk.Separator(frame, orient="horizontal").pack(fill='x', padx=40, pady=15)

        # Zone de texte explicative
        content = tk.Text(frame, wrap="word", font=FONT_NORMAL, bg=CARD_COLOR, relief="flat", bd=0)
        content.pack(padx=40, pady=(0, 30), fill="both", expand=True)

        texte = (
            "🧠 EMSI Finance Explorer – IA Algo Visualizer\n\n"
            "Cette application interactive illustre l’utilisation d’algorithmes d’intelligence artificielle "
            "dans le domaine de l’analyse financière, à travers des données simulées réalistes.\n\n"

            "🔍 Fonctionnement :\n"
            "• Génération de données synthétiques (prix, rendements, séries temporelles…)\n"
            "• Application d’un algorithme par onglet\n"
            "• Visualisation graphique + métriques explicites\n\n"

            "📊 Algorithmes intégrés :\n"
            "1. 📈 Price Prediction (Régression Linéaire)\n"
            "   → Prédiction du rendement à partir d’indicateurs passés\n"
            "2. 📊 Return Clustering (K-Means)\n"
            "   → Regroupement de profils de rendement en 3 clusters\n"
            "3. 🎯 Direction Classification (Random Forest)\n"
            "   → Classification des tendances marché (hausse / baisse)\n"
            "4. 📉 Price Forecast (ARIMA)\n"
            "   → Prévision du prix sur base de séries temporelles simulées\n"
            "5. 🔁 Regression CV (Validation Croisée)\n"
            "   → Évaluation via validation croisée (R² sur 5 sous-ensembles)\n\n"

            "⚙️ Technologies : Python, Tkinter, scikit-learn, matplotlib, statsmodels, pandas\n"
            "Interface graphique moderne et stylisée (EMSI)"
        )

        content.insert("1.0", texte)
        content.config(state="disabled")

def main():
    root = ThemedTk(theme="plastik")
    root.geometry("1200x800")
    apply_style(root, theme="plastik")
    app = AIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()