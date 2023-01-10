import numpy as np
import os
import pandas as pd



# --------------------------------------------------------------------
#  This file is converting a np array to a latex table
# --------------------------------------------------------------------


def export_tex_table(np_array, col_names, savepath, filename, printTable = False):
	
	dt_frame = pd.DataFrame(np.round(np_array, 5), columns = col_names)
	
	column_format = ""
	for i in range(len(col_names)):
		column_format += "r"

	with open(savepath+"/"+filename,"w") as file:

		dt_frame.style.set_properties(**{"font-weight": "bold /* --dwrap */",
										"Huge": "--latex--rwrap"})
		s = dt_frame.style
		sformat = {}


		sformat[col_names[i]] = '{}'

		for i in range(1, len(col_names)):
			sformat[col_names[i]] = '{:.4f}'
		
		s = s.format(sformat).format(subset=col_names[0], precision=0).hide()

		
		#s = s.set_properties(**{"font-weight": "bold /* --dwrap */", "Huge": "--latex--rwrap"})
		#s = s.set_properties(**{hrules=True})

		tabel_str = s.to_latex(hrules = True)
		
		file.write(tabel_str)

	if printTable:
		print(tabel_str)

if __name__ == "__main__":

	np_array = np.array([[1,2,3,4,5,6],2*np.array([1,2,3,4,5,6])])
	col_names = ["Size hidden layer","m0","m1","m2","m3","m4"]
	export_tex_table(np_array, col_names, os.getcwd(), "test.tex")