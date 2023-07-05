## Baroclinic Modes 
[^1] Ocean vertical stratification is a complex issue which may be studied through the analysis of the **dynamic baroclinic modes of motion** and their respective **baroclinic Rossby deformation radii**.
For deriving these quantities, we start from the linearized *QuasiGeostrophic (QG)* equation of motion, describing the motion of the field geostrophic component resulting from the balance between pressure and Coriolis forces:

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2p_0}{\partial&space;x^2}&plus;\frac{\partial^2p_0}{\partial&space;y^2}&plus;\frac{\partial}{\partial&space;z}\Big(\frac{1}{S}\frac{\partial&space;p_0}{\partial&space;z}\Big)\Big\]&plus;\beta\frac{\partial&space;p_0}{\partial&space;x}=0\quad,\qquad(1))

with Boundary Conditions

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big(\frac{\partial&space;p_0}{\partial&space;z}\Big)&space;=&space;0\text{&space;at&space;}z=0,1\quad;)

where $p_0$ is the pressure term expanded at the first order of the Rossby number $\epsilon$, while $\beta$ is the *beta Rossby number*. *S* is the stratification parameter, defined as

![equation](https://latex.codecogs.com/gif.image?\dpi{100}S(z)=\frac{N^2(z)H^2}{f_0^2L^2}\quad,\qquad(2))

with the Brunt-Vaisala frequency *N* obtained as

![equation](https://latex.codecogs.com/gif.image?\dpi{100}N&space;=&space;\Big\[-\frac{g}{\rho_0}\frac{\partial&space;\rho_s}{\partial&space;z}\Big]^\frac{1}{2}\quad(\rho_s\text{basic&space;mean&space;stratification),}\qquad(3)) 

where $\rho_0=1025 kg/m^3$ is the reference density value.
Here, *L* and *H* are respectively the horizontal and vertical scales of motion, assuming $L = 100 km$ and H equal to the region maximum depth ( $O(10^3 km)$ ). The *Coriolis parameter* is assumed $f_0= 10^{-4} 1/s$ while the gravitational acceleration is $g=9.806 m/s^2$.
If we consider a solution to the QG equation of type 

![equation](https://latex.codecogs.com/gif.image?\dpi{100}p_0(x,y,z,t)=f(x,y,t)\Phi(z)\quad\Big\(f=\Re\Big\[e^{i(kx+ly-\sigma{t})}\Big\]\Big\))

where $\Phi(z)$ is the _vertical structure function_, it follows

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\frac{\partial}{\partial&space;t}\Big\[\frac{\partial^2f}{\partial&space;x^2}&plus;\frac{\partial^2f}{\partial&space;y^2}-\lambda\Big\]&plus;\beta\frac{\partial&space;f}{\partial&space;x}=0\quad\quad,\quad\lambda=-\Big\[\frac{\beta{k}}{\sigma}+k^2+l^2\Big\]\quad.)

This necessary leads to the eigenvalues/eigenvector equation

![equation](https://latex.codecogs.com/gif.image?\dpi{100}\boxed{\frac{d}{d&space;z}\Big\(\frac{1}{S}\frac{d\Phi_n}{dz}\Big\)=-\lambda_n\Phi_n\quad\Big\(B.C.\quad{\frac{d\Phi_n}{dz}\vline}_{z=0}^{z=1}=0\Big\)}\qquad(4))

where the eigenvectors $\Phi_n$ are the vertical structure functions, each corresponding to a mode of motion $n=0,1,2\dots$, while the eigenvalues $\lambda_n$ may be used for computing the *baroclinic Rossby radius*

![equation](https://latex.codecogs.com/gif.image?\dpi{110}R_n&space;=&space;\frac{1}{\sqrt{\lambda_n}}\qquad(5))

for each mode *n*. As shown in Grilli, Pinardi (1999) [^1], eigenvalues are null or positive and negative values should be rejected. The trivial eigenvalue $\lambda=0$ corresponds to the **barotropic mode** $(\Phi_0(z)=1)$, while $\lambda=1,2\dots$ correspond to the **baroclinic modes** $(\Phi_{1,2\dots}(z))$. 

This project aims to compute the *baroclinic Rossby radius* and the *vertical structure function* for each mode of motion.

[^1]: F. Grilli, N. Pinardi (1999), "_Le Cause Dinamiche della Stratificazione Verticale nel Mediterraneo_".
