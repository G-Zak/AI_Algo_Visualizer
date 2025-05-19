# main.py

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

        # EMSI logo (place emsiLogo.png next to main.py)
        logo_path = os.path.join(os.path.dirname(__file__), "emsiLogo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image, ImageTk
                pil_img = Image.open(logo_path)
                pil_img = pil_img.resize((550, 100), Image.Resampling.LANCZOS)
                logo_img = ImageTk.PhotoImage(pil_img)
            except ImportError:
                logo_img = tk.PhotoImage(file=logo_path)
            logo_lbl = tk.Label(about, image=logo_img, bg=CARD_COLOR)
            logo_lbl.image = logo_img
            logo_lbl.pack(pady=(20, 10))

        # Title
        tk.Label(
            about,
            text="EMSI Finance Explorer",
            font=FONT_LARGE,
            bg=CARD_COLOR,
            fg="#005D2E" 
        ).pack(pady=(20, 10))

        # Description
        desc = (
            "Explore key financial analytics techniques:\n"
            "• Price Prediction (Linear Regression)\n"
            "• Return Clustering (K-Means)\n"
            "• Direction Classification (Random Forest)\n"
            "• Price Forecasting (ARIMA)\n"
            "• Regression Cross-Validation\n\n"
            "Built with Python, Tkinter, Scikit-learn, Matplotlib, Statsmodels\n"
            "Author: Guennani Zakaria\n"
            "Encadrant: Dr. Mouna El Mkhalet\n"

        )
        tk.Label(
            about,
            text=desc,
            font=FONT_NORMAL,
            bg=CARD_COLOR,
            fg=TEXT_COLOR,
            justify='left',
            padx=20,
            pady=10
        ).pack(fill='both', expand=True)

        # Links
        links = ttk.Frame(about, style='Card.TFrame')
        links.pack(pady=(0, 20), fill='x')

        def make_link(text, url):
            lbl = tk.Label(
                links,
                text=text,
                font=FONT_NORMAL,
                bg=CARD_COLOR,
                fg=ACCENT_COLOR,
                cursor='arrow'
            )
            lbl.pack(anchor='w', padx=20)
            lbl.bind("<Button-1>", lambda e: webbrowser.open(url))

        make_link("🔗 GitHub : https://github.com/G-Zak", "https://github.com/G-Zak")
        make_link("✉️ Email  : guennanizakaria69@gmail.com", "mailto:guennanizakaria69@gmail.com")

        # Prevent About frame from shrinking
        about.pack_propagate(False)


def main():
    # Create themed root window
    root = ThemedTk(theme="plastik")
    root.geometry("1200x800")

    # Apply EMSI custom styling
    apply_style(root, theme="plastik")

    # Launch application
    app = AIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()