import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define figure size in inches for PLOS ONE template (width: 7.08 inches for full width, 3.35 inches for single column)

def set_ieee_style(width=3.5, height=None, fontsize=8,linewidth=2, column='single'):
    """
    Set figure style to match IEEE double-column paper requirements.
    
    Parameters:
    width (float): Figure width in inches.
                   Default is 3.5 (single column).
                   Use 7.16 for double column.
    height (float): Figure height in inches. If None, use golden ratio.
    fontsize (int): Font size for labels and text. Default is 8.
    column (str): 'single' (3.5 in) or 'double' (7.16 in).
    """
    
    # if column == 'double':
    #     width = 7.16
    # else:
    #     width = 3.5
    
    if height is None:
        height = width / 1.618  # Golden ratio
    
    plt.rcParams.update({
        'figure.figsize': (width, height),
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'lines.linewidth': linewidth,
        'axes.linewidth': 0.8,
        'xtick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.major.size': 3,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'legend.frameon': False,
        'savefig.dpi': 1200,
        'savefig.format': 'pdf',  # IEEE prefers vector graphics
    })
    
    # Clean, professional style
    #sns.set_style("whitegrid")








def set_plos_one_style(width=6.9, height=None, fontsize=12,linewidth=1.0):
    """
    Set figure style to match PLOS ONE template.
    
    Parameters:
    width (float): Figure width in inches. Default is full-width (7.08 inches).
    height (float): Figure height in inches. If None, use golden ratio.
    fontsize (int): Font size for labels and text. Default is 8.
    """
    if height is None:
        height = width / 1.618  # Use golden ratio for height
    
    plt.rcParams.update({
        'figure.figsize': (width, height),
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'lines.linewidth': linewidth,
        'axes.linewidth': 0.8,
        'xtick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.major.size': 3,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'legend.frameon': False,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',  # Change to 'pdf' for vector graphics
    })
    
    # Use seaborn style for better aesthetics
    sns.set_style("ticks")



def save_fig(name, path):
    formats = ['pdf', 'eps', 'svg', 'png']
    
    for fmt in formats:
        # Create folder path: e.g., path/figures_pdf
        folder_path = os.path.join(path, f'figures_{fmt}')
        os.makedirs(folder_path, exist_ok=True)
        
        # Define filename
        file_name = os.path.join(folder_path, f'{name}.{fmt}')
        
        # Save (add dpi only for png)
        if fmt == 'png':
            plt.savefig(file_name, bbox_inches='tight', dpi=300)
        else:
            plt.savefig(file_name, bbox_inches='tight')

