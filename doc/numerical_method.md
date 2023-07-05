# Numerical Method Implemented
Unless the case of particular Brunt-Vaisala frequency profiles (see LaCasce, 2012 [^2]), the above [eigenvalues/eigenvectors problem](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}) may not be analytically solved for a realistic one. Thus, it is necessary to solve it numerically  employing a change of variable:

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w=\frac{1}{S}\frac{d\Phi}{dz}\\\\\Rightarrow\Phi&space;=\int_{0}^{z}Swdz+\Phi_0\quad\Big\(\Phi_0=\Phi\vert_{z=0}=const\Big\)\qquad(6))

so that

![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{dw}{dz}=-\lambda\Phi=-\lambda\Big\(\int_{0}^{z}Swdz+\Phi_0\Big\)\\\\\Rightarrow\boxed{\frac{d^2w}{dz^2}=-\lambda{S}{w}}\quad\Big\(B.C.\quad{w=0}\quad{at}\quad{z=0,1}\Big\)\qquad(7))

obtaining a simple eigenvalues/eigenvectors problem of known resolution. 
The numerical method implemented here aims to find the eigenvalues  and eigenvectors in eq. [(4)](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}), exploiting relation [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0) and numerically solving the well known problem [(7)](https://latex.codecogs.com/gif.image?\dpi{110}\frac{d^2w}{dz^2}=-\lambda{S}{w}) for $\lambda_n$ and $w_n$. This is done through few steps, starting from the Brunt-Vaisala frequency vertical profile:
1. The Brunt-Vaisala frequency is linearly interpolated on a new equally spaced  depth grid (1 m grid step).
2. Problem parameter S (depth-dependent) is computed as in eq. [(2)](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,).
3. The *left* finite difference matrix  corresponding to operator $\frac{d^2}{dz^2}$ and the *right* diagonal matrix related to *S* are computed. 
The eigenvalues **discretized problem** is solved:

![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{1}{12dz^{2}}\begin{bmatrix}0&0&\0&0&0&0&\dots&0\\\12&-24&12&0&\dots&0&\dots&0\\\\-1&16&-30&16&-1&0&\dots&0\\\0&-1&16&-30&16&\dots&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\dots&\dots&\vdots\\\0&\dots&\dots&0&0&12&-24&12&space;\\\0&\dots&0&0&0&0&0&0\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix}=-\lambda\begin{bmatrix}S_0&0&0&0&\dots&\dots&0\\\0&S_1&0&0&\dots&\dots&0\\\0&0&S_2&0&\dots&\dots&0\\\0&0&0&S_3&\dots&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\ddots&\vdots\\\0&\dots&\dots&\dots&0&S_{n-1}&0\\\0&\dots&\dots&\dots&0&0&S_n\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix})

where *n* is the number of points along depth axis, *dz* is the scaled grid step. Boundary Conditions are implemented setting the first and last lines of the finite difference matrix (L.H.S.) equal to 0.

4. Eigenvectors are found integrating eq. [(7)](https://latex.codecogs.com/gif.image?\dpi{110}\frac{d^2w}{dz^2}=-\lambda{S}{w}\quad\Big\(B.C.\quad{w=0}\quad{at}\quad{z=0,1}\Big\)) through *Numerov's* numerical method

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w_{n&plus;1}=\left(\frac{2-\frac{10\Delta{t^2}}{12}\lambda{S_n}}{1+\frac{\Delta{t^2}}{12}{\lambda}S_{n&plus;1}}\right)w_n-\left(\frac{1+\frac{\Delta{t^2}}{12}{\lambda}S_{n-1}}{1+&space;\frac{\Delta{t^2}}{12}{\lambda}S_{n&plus;1}}&space;\right)w_{n-1})

where each eigenvalue is used for computing the corresponding eigenvector.
The first value of each eigenvector is computed as

![equation](https://latex.codecogs.com/gif.image?\dpi{110}w_{1}=\frac{\Delta{z}\frac{dw}{dz}\vert_{z=0}}{(1&plus;\lambda\frac{S_1\Delta&space;z^2}{6})}\quad\text{with}\quad\frac{dw}{dz}\vert_{z=0}=-\lambda\Phi_0\quad\Big(\Phi_0=\Phi\vert_{z=0}=1\Big))

where $\Phi_0$ is the surface value, equal to the modes maximum amplitude (and equal to the barotropic mode value). Here, it is set equal to 1.

5. The *baroclinic Rossby radii* $R_n$ are computed as in eq. [(5)](https://latex.codecogs.com/gif.image?\dpi{110}R_n&space;=&space;\frac{1}{\sqrt{\lambda_n}}) while the *vertical structure functions* are obtained integrating *S, w* as in [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0). The integration constant is set $\Phi_0=1$, as already discussed.

[^2]: J. H. LaCasce (2012), "_Surface Quasigeostrophic Solutions and Baroclinic Modes with Exponential Stratification_".
