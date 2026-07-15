
import matplotlib.pyplot as plt

params = {
	'font.size': 10,
	'text.usetex': True,
	'lines.linewidth': 1,
	'lines.markersize': 3,
	'errorbar.capsize': 2,
	'legend.frameon': False,
	'legend.columnspacing': 1,
	'savefig.format': 'png',
	'savefig.dpi': 600,
	'savefig.bbox': 'tight',
	'figure.subplot.hspace': 0.1,
}

params['text.latex.preamble'] = ' '.join([
    r'\usepackage{amsmath, amsthm, amssymb}',
    r'\usepackage[mathcal]{euscript}',
])

#plt.style.use('seaborn-dark-palette')
plt.rcParams.update(params)

