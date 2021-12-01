import matplotlib.pyplot as plt
import numpy as np

__helicopter = [
    [[-0.8538541 , 0.15147194],
     [-0.06540896, 0.1060361 ],
     [-0.03200028, 0.30648828],
     [ 0.03348076, 0.30381558],
     [ 0.06822578, 0.11939964],
     [ 0.20319694, 0.11138154],
     [ 0.3207955 , 0.08064552],
     [ 0.40364908, 0.00447374],
     [ 0.42102156,-0.03695308],
     [ 0.41166718,-0.07570714],
     [ 0.37825844,-0.10644314],
     [ 0.32614092,-0.12916106],
     [ 0.21789676,-0.12782466],
     [-0.04269104,-0.11045212],
     [-0.13089   ,-0.10109772],
     [-0.22443434,-0.07036172],
     [-0.28857902, 0.04055516],
     [-0.8578631 , 0.11271794],
     [-0.8842559 , 0.06026624],
     [-0.9273531 , 0.02886212],
     [-0.9627663 , 0.0181713 ],
     [-1.0028567 , 0.0185051 ],
     [-1.0389381 , 0.02919586],
     [-1.0786945 , 0.06360684],
     [-1.0977375 , 0.09033376],
     [-1.1100987 , 0.1250788 ],
     [-1.1127713 , 0.1591557 ],
     [-1.1044193 , 0.19122802],
     [-1.0860445 , 0.22330038],
     [-1.0539721 , 0.25203184],
     [-1.0195611 , 0.26773392],
     [-0.9757957 , 0.27642018],
     [-0.9367075 , 0.26706576],
     [-0.9086443 , 0.25102958],
     [-0.8805809 , 0.22730942],
     [-0.8662153 , 0.19824386],
     [-0.8568609 , 0.17118282],
     [-0.8538541 , 0.15147194]],
    [[ 0.3172041 ,-0.1674975 ],
     [ 0.28838906,-0.1669129 ],
     [ 0.27327162,-0.1796082 ],
     [ 0.14840666,-0.17894   ],
     [ 0.14740446,-0.12414976],
     [ 0.11432984,-0.12147706],
     [ 0.12201382,-0.18027636],
     [-0.08344958,-0.18061016],
     [-0.08378338,-0.10577472],
     [-0.11618982,-0.10377018],
     [-0.10817172,-0.18061016],
     [-0.14859624,-0.18996462],
     [-0.10850578,-0.2003213 ],
     [ 0.2740237 ,-0.1993191 ],
     [ 0.30175288,-0.18762606],
     [ 0.3172041 ,-0.1674975 ]],
    [[ 0.01632178, 0.30475262],
     [ 0.01632178, 0.3671187 ],
     [-0.01415256, 0.3671187 ],
     [-0.01398456, 0.3059768 ],
     [ 0.01632178, 0.30475262]],
    [[ 0.8833978 , 0.34791506],
     [-0.8457001 , 0.34791506],
     [-0.8457001 , 0.33505272],
     [ 0.8836484 , 0.33505272],
     [ 0.8833978 , 0.34791506]],
    [[ 0.00047876, 0.08595646],
     [ 0.00047876,-0.09404354]],
    [[-0.08952124,-0.00404354],
     [ 0.09047876,-0.00404354]]
]


def rotmat(ϕ):
    return np.array([
        [np.cos(ϕ), -np.sin(ϕ)], 
        [np.sin(ϕ), np.cos(ϕ)], 
    ])


def transform(pts, dx, dy, scale, ϕ):
    R = rotmat(ϕ) * scale
    d = np.array([[dx], [dy]])
    return (R @ np.array(pts).T + d).T


def plot_cont(c, dx, dy, ϕ, *wargs, **kwargs):
    ct = transform(c, dx, dy, 1, ϕ)
    return plt.plot(ct[:,0], ct[:,1], *wargs, **kwargs)


def plot_helicopter(x, y, ϕ, scale=1):
    plots = []
    for c in __helicopter:
        c = np.array(c)
        p = plot_cont(c * scale, x, y, ϕ, 'b')
        plots += [p]
    return plots


if __name__ == '__main__':
    plt.axis('equal')
    plot_helicopter(0, 0, 0)
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()