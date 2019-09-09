from matplotlib import rc
from matplotlib import rcParams


rc('xtick', labelsize=16) 
rc('ytick', labelsize=16) 

rc('legend', fontsize=15) 


title_font = {'fontname':'Arial', 'size':'12', 'color':'black', 'weight':'normal'} 

axis_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal'} 

legend_font = {'fontname':'Arial', 'size':'16'} 

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

print("Fonts loaded")