import os
import sys
from datetime import datetime

def get_app_dir():
    """Get absolute path relative to the executable location."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path


def get_abs_path(relative_path):
    """Get absolute path to a resource, relative to the base directory."""
    return os.path.join(get_app_dir(), relative_path)


def time_based_dir(prefix, base_path, multiple_cocultures=False):  # used to be get_output_dir()
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    time_dir_name = "_".join([prefix, timestamp])
    if os.getcwd() == "/app":
        time_dir = os.path.join("/media", time_dir_name)
    else:
        time_dir = os.path.join(base_path, time_dir_name)
    if os.path.exists(time_dir) and multiple_cocultures is False:
        base, ext = os.path.splitext(time_dir)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        time_dir = f"{base}_{counter}{ext}"
    return time_dir


def button_style(font_size=12, padding=5, color="black", bck_col="#90EE90",
                 bck_col_hov="#7FCF7F", bck_col_clicked="#72B572", radius=5):
    style = f"""
    QPushButton {{
        font-size: {font_size}px;
        font-weight: bold;
        padding: {padding}px;
        color: {color};
        background-color: {bck_col};  /* Light green color */
        border-radius: {radius}px;
    }}
    QPushButton:hover {{
        background-color: {bck_col_hov};  /* Slightly darker green on hover */
    }}
    QPushButton:pressed {{
        background-color: {bck_col_clicked};  /* Even darker when pressed */
    }}
    """
    return style
