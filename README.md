# ocean-baroclinic-modes
This project is for computing the *baroclinic vertical structure function* and the *baroclinic Rossby radius* for  each mode of motion, in an ocean region of interest to the user.

## Baroclinic Modes [^1]
Ocean vertical stratification is a complex issue which may be studied through the analysis of the **dynamic baroclinic modes of motion** and their respective **baroclinic Rossby deformation radii**.
For deriving these quantities, we start from the linearized *QuasiGeostrophic (QG)* equation of motion, describing the motion of the field geostrophic component resulting from the balance between pressure and Coriolis forces:
![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2p_0}{\partial&space;x^2}&plus;\frac{\partial^2p_0}{\partial&space;y^2}&plus;\frac{\partial}{\partial&space;z}\Big(\frac{1}{S}\frac{\partial&space;p_0}{\partial&space;z}\Big)\Big\]&plus;\beta\frac{\partial&space;p_0}{\partial&space;x}=0\quad,\qquad(1))
with Boundary Conditions
![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big(\frac{\partial&space;p_0}{\partial&space;z}\Big)&space;=&space;0\text{&space;at&space;}z=0,1\quad;)
where $p_0$ is the pressure term expanded at the first order of the Rossby number $\epsilon$, while $\beta$ is the *beta Rossby number*. *S* is the stratification parameter, defined as
![equation](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,\qquad(2))
with the Brunt-Vaisala frequency *N* obtained as
![equation](https://latex.codecogs.com/gif.image?\dpi{100}N&space;=&space;\Big\[-\frac{g}{\rho_s}\frac{\partial&space;\rho_s}{\partial&space;z}\Big]^\frac{1}{2}\quad(\rho_s\text{basic&space;mean&space;stratification).}\qquad(3)) 
Here, *L* and *H* are respectively the horizontal and vertical scales of motion, assuming $L = 100 km$ and H equal to the region maximum depth ( $O(10^3 km)$ ). The *Coriolis parameter* is assumed $f_0= 10^{-4} 1/s$ while the gravitational acceleration is $g=9.806 m/s^2$.
If we consider a solution to the QG equation of type 
![equation](https://latex.codecogs.com/gif.image?\dpi{100}p_0(x,y,z,t)=f(x,y,t)\Phi(z)\quad\Big\(f=\Re\Big\[e^{i(kx+ly-\sigma{t})}\Big\]\Big\))
where $\Phi(z)$ is the _vertical structure function_, it follows

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2f}{\partial&space;x^2}&plus;\frac{\partial^2f}{\partial&space;y^2}-\lambda\Big\]&plus;\beta\frac{\partial&space;f}{\partial&space;x}=0\quad\quad,\quad\lambda=-\Big\[\frac{\beta{k}}{\sigma}+k^2+l^2\Big\]\quad.)
This necessary leads to the eigenvalues/eigenvector equation
![equation](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}\qquad(4))
where the eigenvectors $\Phi_n$ are the vertical structure functions, each corresponding to a mode of motion $n=0,1,2\dots$, while the eigenvalues $\lambda_n$ may be used for computing the *baroclinic Rossby radius*
![equation](https://latex.codecogs.com/gif.image?\dpi{110}R_n(z)&space;=&space;\frac{L}{\sqrt{S(z)\lambda_n}}\qquad(5))
for each mode *n*. As shown in Grilli, Pinardi (1999) [^1], eigenvalues are null or positive and negative values should be rejected. The trivial eigenvalue $\lambda=0$ corresponds to the **barotropic mode** $(\Phi_0(z)=1)$, while $\lambda=1,2\dots$ correspond to the **baroclinic modes** $(\Phi_{1,2\dots}(z))$. 

This project aims to compute the *baroclinic Rossby radius* and the *vertical structure function* for each mode of motion.

[^1]: F. Grilli, N. Pinardi (1999), "_Le Cause Dinamiche della Stratificazione Verticale nel Mediterraneo_".

## Numerical Method Implemented
Unless the case of particular Brunt-Vaisala frequency profiles (see LaCasce, 2012 [^2]), the above [eigenvalues/eigenvectors problem](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}) may not be analytically solved for a realistic one. Thus, it is necessary to solve it numerically  employing a change of variable:
![equation](https://latex.codecogs.com/gif.image?\dpi{110}w=\frac{1}{S}\frac{d\Phi}{dz}\\\\\Rightarrow\Phi&space;=\int_{0}^{z}Swdz+\Phi_0\quad\Big\(\Phi_0=\Phi\vert_{z=0}=const\Big\)\qquad(6))
so that
![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{dw}{dz}=-\lambda\Phi=-\lambda\Big\(\int_{0}^{z}Swdz+\Phi_0\Big\)\\\ \Rightarrow\boxed{\frac{d^2w}{dz^2}=-\lambda{S}{w}}\qquad(7))
obtaining a simple eigenvalues/eigenvectors problem of known resolution. 
The numerical method implemented here aims to find the eigenvalues  and eigenvectors in eq. [(4)](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}), exploiting relation [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0) and numerically solving the well known problem [(7)](https://latex.codecogs.com/gif.image?\dpi{110}\frac{d^2w}{dz^2}=-\lambda{S}{w}) for $\lambda_n$ and $w_n$. This is done through few steps, starting from the Brunt-Vaisala frequency vertical profile:
1. The Brunt-Vaisala frequency is linearly interpolated on a new equally spaced and scaled depth grid (going from 0 to 1).
2. The problem parameter S (depth-dependent) is computed as in eq. [(2)](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,).
3. The *left* finite difference matrix  corresponding to operator $\frac{d^2}{dz^2}$ and the *right* diagonal matrix related to $S$ are computed. The eigenvalues/eigenvectors **discretized problem** 
![equation](https://latex.codecogs.com/gif.image?\dpi{110}\frac{1}{dz^{2}}\begin{bmatrix}\frac{2}{dz}&\frac{-5}{dz}&\frac{4}{dz}&\frac{-1}{dz}&0&\dots&0\\\1&-2&1&0&\dots&\dots&0\\\0&1&-2&1&0&\dots&0\\\0&0&1&-2&1&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\dots&\vdots\\\0&\dots&\dots&0&1&-2&1&space;\\\0&\dots&0&\frac{2}{dz}&\frac{-5}{dz}&\frac{4}{dz}&\frac{-1}{dz}\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix}=-\lambda\begin{bmatrix}S_0&0&0&0&\dots&\dots&0\\\0&S_1&0&0&\dots&\dots&0\\\0&0&S_2&0&\dots&\dots&0\\\0&0&0&S_3&\dots&\dots&0\\\\\vdots&\ddots&\ddots&\ddots&\ddots&\ddots&\vdots\\\0&\dots&\dots&\dots&0&S_{n-1}&0\\\0&\dots&\dots&\dots&0&0&S_n\end{bmatrix}\begin{bmatrix}w_0\\\w_1\\\w_2\\\\\vdots\\\\\vdots\\\w_{n-1}\\\w_n\end{bmatrix})
is solved, where *n* is the number of points along depth axis (equal to the number of eigenvalues/eigenvectors).
4. Negative eigenvalues and the corresponding eigenvectors are discarded.
5. The *baroclinic Rossby radii* $R_n$ are computed as in eq. [(5)](https://latex.codecogs.com/gif.image?\dpi{110}R_n(z)&space;=&space;\frac{L}{\sqrt{S(z)\lambda_n}}) while the *vertical structure functions* are obtained integrating *S, w* as in [(6)](https://latex.codecogs.com/gif.image?\dpi{110}\Phi&space;=\int_{0}^{z}Swdz+\Phi_0). The integration constant is set $\Phi_0=1$, corresponding to the surface (barotropic) mode. 
The functions $\Phi_n(z)$ are then normalized dividing them by their norm
![equation](https://latex.codecogs.com/gif.image?\dpi{110}\vert\vert\Phi_n\vert\vert=\int_{0}^{1}\Phi^2dz)
6. Lastly, $R_n$ and $Phi_n$ are re-interpolated on the original depth grid.
[^2]: J. H. LaCasce (2012), "_Surface Quasigeostrophic Solutions and Baroclinic Modes with Exponential Stratification_".






