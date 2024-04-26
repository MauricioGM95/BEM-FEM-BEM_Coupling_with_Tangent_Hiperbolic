# BEM-FEM-BEM_Coupling_with_Tangent_Hiperbolic
An alternative to calculate the coupling of BEM-FEM-BEM with variable permittivity using the hyperbolic tangent as a model, applied for different types of solutes.

\begin{itemize}
\item To create the .pqr files for each solute, the code Creation_of_PQR_files_General.ipynb must be executed.
\item After generating the .pqr files, the code Creation_of_Surface_Meshes_General.ipynb is used to generate the surface meshes of each solute.
\item Then of generating the surface meshes, the code Creation_of_Volumetric_Meshes_General.ipynb is used to generate the volumetric meshes of each solute.
\item Finally with the volumetric meshes, the code Creation_Alpha_files_General.ipynb must be executed to generate the text files of the alpha geometric parameter of each solute to work with the variable permittivity model.
\end{itemize}

The Iterative Code_HT_Mobley.py and Iterative Code_HT_TanH_Sphere.py codes are an alternative to calculate the energy of each solute iteratively instead of individually.

Pendiente 
1 Hacer el README
2 Corroborar resultados de cavidad de BB y BFB en el paper
3 .py para codigo de Union

Opcional
1 Excel
2 Grafico opcional 1
3 Visualizacion corte BFB
