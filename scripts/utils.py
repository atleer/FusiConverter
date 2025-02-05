import tkinter as tk
from tkinter import filedialog


def select_files_from_gui(title=None, defaultextension='.mat') -> tuple[str, ...]:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilenames(title=title, defaultextension=defaultextension)  # Open file dialog
    if not file_path:
        return ()
    return file_path


def select_save_dir_from_gui() -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askdirectory(title="Save Directory")  # Open folder dialog
    if not file_path:
        return None
    return file_path

