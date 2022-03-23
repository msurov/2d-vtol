import svgpathtools
import numpy as np
from matplotlib.transforms import Affine2D,IdentityTransform
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
import matplotlib.pyplot as plt
from copy import copy, deepcopy

def complex2pair(z):
	return (z.real, z.imag)

def get_segment_vertcmd(obj):
	if isinstance(obj, svgpathtools.path.Line):
		v = [complex2pair(obj.start), complex2pair(obj.end)]
		c = [Path.MOVETO, Path.LINETO]
	elif isinstance(obj, svgpathtools.path.Arc):
		ccgen = obj.as_cubic_curves()
		v = []
		c = []
		for cc in ccgen:
			v += [complex2pair(cc.start), complex2pair(cc.control1), complex2pair(cc.control2), complex2pair(cc.end)]
			c += [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
	elif isinstance(obj, svgpathtools.path.CubicBezier):
		v = [complex2pair(obj.start), complex2pair(obj.control1), complex2pair(obj.control2), complex2pair(obj.end)]
		c = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
	else:
		assert False, f'Unimplemented object type {type(obj)}'

	return v,c

def get_pathobj(path):
	v = []
	c = []

	for i in range(len(path)):
		vi,ci = get_segment_vertcmd(path[i])

		if i > 0 and np.allclose(path[i-1].end, path[i].start):
			v += vi[1:]
			c += ci[1:]
		else:
			v += vi
			c += ci

	return Path(v, c)

def get_style_props(attrs):
    if 'style' in attrs:
        entries = attrs['style'].split(';')
        return dict([e.split(':') for e in entries])
    return None

def get_fill_clolor(style):
    if 'fill' not in style: return None
    r,g,b = svgpathtools.hex2rgb(style['fill'])
    a = float(style['fill-opacity']) if 'fill-opacity' in style else 1.
    return (r/255.,g/255.,b/255.,a)

def load_path_patches(svgpath):
    paths,attributes = svgpathtools.svg2paths(svgpath)
    objs = []

    for p,a in zip(paths,attributes):
        if len(p) == 0:
            continue
        path = get_pathobj(p)
        style = get_style_props(a)
        fillcolor = get_fill_clolor(style)
        if fillcolor is not None:
            obj = PathPatch(path, fill = True, fc = fillcolor, ec='white', lw=0.2)
        else:
            obj = PathPatch(path, fill = False)
        objs += [obj]

    return objs

def clone_pathpatch(pp):
    res = PathPatch(pp.get_path())
    res.update_from(pp)
    return res

def clone_patches(patches):
    return [clone_pathpatch(pp) for pp in patches]

class Quadrotor:
    # TODO: configurable dihedral. 
    # Load separately quadrotor body and propellers,
    # transform propellers. 
    # how to define propellers positions in svg? with a group?

    def __init__(self, ax, svgpath=None, patches=None):
        self.ax = ax
        self.t0 = Affine2D()
        self.t = Affine2D()
        if svgpath is not None:
            patches = load_path_patches(svgpath)
        self.patches = [] if patches is None else patches
        for p in self.patches:
            ax.add_patch(p)

    def set_default_transform(self, x, y, theta, sx=1, sy=1):
        self.t0.clear()
        self.t0.scale(sx, sy)
        self.t0.rotate(theta)
        self.t0.translate(x, y)
        t = self.t0 + self.ax.transData
        for p in self.patches: p.set_transform(t)

    def move(self, x, y, theta):
        self.t.clear()
        self.t.rotate(theta)
        self.t.translate(x, y)
        t = self.t0 + self.t + self.ax.transData
        for p in self.patches: p.set_transform(t)
    
    def set_alpha(self, alpha=1):
        for p in self.patches:
            p.set_alpha(alpha)

    def __copy__(self):
        patches = clone_patches(self.patches)
        q = Quadrotor(self.ax, patches=patches)
        q.t0 = deepcopy(self.t0)
        q.t = deepcopy(self.t)
        return q

def test():
    plt.figure()
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    quad = Quadrotor(ax, 'fig/drawing.svg')
    quad.set_default_transform(0, 0, 0, 0.5, -0.5)
    quad2 = copy(quad)
    quad2.move(-1, -5, 0)
    quad.move(6, 4, 1)

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test()
