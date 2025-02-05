import tkinter as tk
from tkinter import filedialog


def select_files_from_gui(title=None, defaultextension='.mat'):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilenames(title=title, defaultextension=defaultextension)  # Open file dialog
    return file_path


def select_save_dir_from_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askdirectory(title="Save Directory")  # Open folder dialog
    return file_path

