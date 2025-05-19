import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

# EMSI Color Palette
PRIMARY_COLOR    = "#007A3D"  # EMSI Green
SECONDARY_COLOR  = "#005D2E"  # Darker EMSI Green
ACCENT_COLOR     = "#FFA500"  # Orange
HIGHLIGHT        = ACCENT_COLOR
BACKGROUND_COLOR = "#F4F6F8"  # Light gray
CARD_COLOR       = "#FFFFFF"  # White
TEXT_COLOR       = "#2C3E50"  # Dark blue-gray
BORDER_COLOR     = "#D1D5DB"  # Light gray
HOVER_COLOR      = "#005D2E"  # Light EMSI Green
DISABLED_COLOR   = "#95A5A6"  # Gray
ACTIVE_COLOR     = "#004D27"  # Dark EMSI Green

# Fonts
FONT_LARGE    = ("Segoe UI", 25, "bold")
FONT_MEDIUM   = ("Segoe UI", 16)
FONT_NORMAL   = ("Segoe UI", 13)
FONT_BUTTON   = ("Segoe UI", 12, "bold")


def apply_style(root, theme: str = None):
    """
    Applies a ttkthemes theme (if available) then overrides
    with the EMSI color palette and custom tab/button styling.
    """
    # Try to use ThemedStyle if ttkthemes is installed
    try:
        from ttkthemes import ThemedStyle
        style = ThemedStyle(root)
        if theme:
            style.set_theme(theme)
    except ImportError:
        style = ttk.Style(root)

    # Set default font
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="Segoe UI", size=10)

    # Window background
    root.configure(bg=BACKGROUND_COLOR)

    # Card-style frames
    style.configure(
        'Card.TFrame',
        background=CARD_COLOR,
        borderwidth=1,
        relief='solid',
        bordercolor=BORDER_COLOR
    )

    # Notebook container
    style.configure('TNotebook', background=BACKGROUND_COLOR, borderwidth=0)

    # Default Tab styling
    style.configure(
        'TNotebook.Tab',
        font=FONT_MEDIUM,
        padding=(15, 8),
        background=CARD_COLOR,
        foreground=TEXT_COLOR,
        bordercolor=BORDER_COLOR
    )
    # Tab state-based overrides
    style.map(
        'TNotebook.Tab',
        background=[
            ('selected', PRIMARY_COLOR),  # Active tab
            ('hover',    HOVER_COLOR),    # Mouse over
            ('!selected', CARD_COLOR)     # Inactive
        ],
        foreground=[
            ('selected', TEXT_COLOR),     # White text on active
            ('hover',    TEXT_COLOR),     # Dark text on hover
            ('!selected', TEXT_COLOR)     # Dark text inactive
        ]
    )

    # TTK Button minimal overrides (we use custom buttons for full control)
    style.configure(
        'TButton',
        font=FONT_BUTTON,
        padding=8,
        relief='flat'
    )
    style.map(
        'TButton',
        background=[
            ('hover',    "#005D2E" ),
            ('disabled', "#005D2E" ),
            ('pressed',  "#005D2E" )
        ],
        foreground=[
            ('disabled', 'white')
        ]
    )


def create_modern_button(parent, text, command, **kwargs):
    """
    Create a flat, web-style Tk button with:
      - EMSI colors
      - hover effect
      - custom padding and font
    """
    bg   = kwargs.get('bg', PRIMARY_COLOR)
    fg   = kwargs.get('fg', 'white')
    font = kwargs.get('font', FONT_BUTTON)
    padx = kwargs.get('padx', 16)
    pady = kwargs.get('pady', 8)

    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=ACTIVE_COLOR,
        activeforeground=fg,
        relief='flat',
        font=font,
        padx=padx,
        pady=pady,
        bd=0,
        cursor='arrow',
        highlightthickness=0
    )
    # Hover effect
    btn.bind('<Enter>', lambda e: btn.config(bg=HOVER_COLOR))
    btn.bind('<Leave>', lambda e: btn.config(bg=bg))
    return btn


# Alias for backward compatibility
create_web_button = create_modern_button