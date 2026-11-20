from pathlib import Path


def registration_file_path(source_path, suffix):
    """Get path to where the registration files should be loaded from or where they should be saved to.
    Registrations files are written to Processed/<animal>/<session>/registration_files/ for a recording (stored in Raw/<animal>/<session>/), 
    if the file that's aligned is an atlas, it is stored next to the source (i.e., next to the atlas)"""
    source_path = Path(source_path)
    parts = source_path.parts
    if 'Raw' in parts[:-3]: # Raw/<animal>/<session>/<file>
        i = len(parts) - 1 - parts[::-1].index('Raw')
        folder = Path(*parts[:i], 'Processed', parts[i + 1], parts[i + 2], 'registration_files')
    else:
        folder = source_path.parent
    return folder / f'{source_path.name}.{suffix}'
