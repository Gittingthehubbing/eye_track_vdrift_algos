
from matplotlib import font_manager
all_fonts = [x.name for x in font_manager.fontManager.ttflist]
try:
    font_dirs = ['fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    "done"
except Exception as e:
    print(f"Adding fonts failed for {font_dirs}, please add font files to ./fonts")
