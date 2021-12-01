from lxml import etree
import matplotlib.pyplot as plt
import re
import numpy as np
import argparse


def extract_float(s):
    '''
        val, rest of string
    '''
    ans = re.match(r'\s*([\d\.eE\+\-]+)', s)
    if ans is None:
        return None
    _,end = ans.span(0)
    val = float(ans.group(1))
    return val, s[end:]


def extract_x_comma_y(s):
    ans = re.match(r'\s*([\d\.eE\+\-]+),\s*([\d\.eE\+\-]+)', s)
    if ans is None:
        return None
    _,end = ans.span(0)
    x = float(ans.group(1))
    y = float(ans.group(2))
    return np.array([x,y]), s[end:]


def extract_points(s):
    ans = extract_x_comma_y(s)
    if ans is None:
        return None

    p,s = ans
    pts = [p]

    while True:
        ans = extract_x_comma_y(s)
        if ans is None:
            break
        p,s = ans
        pts += [p]

    return pts,s


def extract_cmd_double(s, cmd):
    '''
        cmd (X1,Y1) (X2,Y2) ... (Xn,Yn)
    '''
    s = s.lstrip()
    if len(s) == 0:
        return None

    if s[0] != cmd:
        return None

    val,s = extract_x_comma_y(s[1:])
    vals = [val]
    while True:
        ans = extract_x_comma_y(s[1:])
        if ans is None:
            break
        val,s = ans
        vals += [val]

    return vals,s


def extract_cmd_single(s, cmd):
    '''
        cmd A1 A2 ... An
    '''
    s = s.lstrip()
    if len(s) == 0:
        return None

    if s[0] != cmd:
        return None

    val,s = extract_float(s[1:])
    vals = [val]
    while True:
        ans = extract_float(s[1:])
        if ans is None:
            break
        val,s = ans
        vals += [val]

    return vals,s


def parse_path(s):
    s = s.lstrip()

    path = []
    cp = None

    while True:
        ans = extract_cmd_double(s, 'M')
        if ans is not None:
            vals,s = ans
            # print('extracted M:', vals)
            path = vals
            cp = path[-1]
            continue

        ans = extract_cmd_double(s, 'm')
        if ans is not None:
            vals,s = ans
            # print('extracted m:', vals)
            for i in range(1,len(vals)):
                vals[i] += vals[i-1]
            path = vals
            cp = path[-1]
            continue

        ans = extract_cmd_double(s, 'l')
        if ans is not None:
            vals,s = ans
            # print('extracted l:', vals)

            vals[0] += cp
            for i in range(1, len(vals)):
                vals[i] += vals[i-1]

            for i in range(0, len(vals)):
                path += [vals[i]]

            cp = path[-1]
            continue

        ans = extract_cmd_single(s, 'V')
        if ans is not None:
            vals,s = ans
            # print('extracted V:', vals)
            x = cp[0]
            for i in range(0, len(vals)):
                p = np.array([x, vals[i]])
                path += [p]
            cp = path[-1]
            continue

        ans = extract_cmd_single(s, 'h')
        if ans is not None:
            vals,s = ans
            # print('extracted h:', vals)
            x,y = cp
            vals[0] += x
            for i in range(1, len(vals)):
                vals[i] += vals[i-1]

            for i in range(0, len(vals)):
                p = np.array([vals[i], y])
                path += [p]

            cp = path[-1]
            continue

        # ans = extract_cmd_double(s, 'L')
        # if ans is not None:
        #     vals,s = ans
        #     print('extracted L:', vals)
        #     continue

        ans = extract_cmd_single(s, 'v')
        if ans is not None:
            vals,s = ans
            # print('extracted l:', vals)
            x,y = cp
            vals[0] += y
            for i in range(1, len(vals)):
                vals[i] += vals[i-1]

            for i in range(0, len(vals)):
                p = np.array([x, vals[i]])
                path += [p]

            cp = path[-1]
            continue

        ans = extract_cmd_single(s, 'H')
        if ans is not None:
            vals,s = ans
            # print('extracted V:', vals)
            y = cp[1]
            for i in range(0, len(vals)):
                p = np.array([vals[i], y])
                path += [p]
            cp = path[-1]
            continue

        s = s.lstrip()
        if len(s) == 0:
            break

        if s[0] == 'z' or s[0] == 'Z':
            path += [path[0]]
            break

        assert False, 'undef cmd %s' % s[0]

    return np.array(path)


def get_pathes(root):
    pathes = {}
    elems = root.findall('{http://www.w3.org/2000/svg}path')
    for e in elems:
        id = e.attrib['id']
        path = parse_path(e.attrib['d'])
        pathes[id] = path

    return pathes


def main():
    parser = argparse.ArgumentParser(description='Extract pathes from SVG file')
    parser.add_argument('--svg', metavar='svg', type=str, required=True, help='path to svg file to parse')
    parser.add_argument('--sx', metavar='sx', type=float, default=1., help='scale x')
    parser.add_argument('--sy', metavar='sy', type=float, default=1., help='scale y')
    parser.add_argument('--dx', metavar='dx', type=float, default=0., help='traslate image by x')
    parser.add_argument('--dy', metavar='dy', type=float, default=0., help='traslate image by y')
    parser.add_argument('--show', metavar='show', type=bool, default=False, help='render extracted pathes')

    args = parser.parse_args()
    root = etree.parse(args.svg)
    
    dx = args.dx
    dy = args.dy
    sx = args.sx
    sy = args.sy
    show = args.show

    pathes = get_pathes(root)
    arr = []
    for id in pathes:
        path = pathes[id]
        path[:,0] = sx * (path[:,0] + dx)
        path[:,1] = sy * (path[:,1] + dy)
        s = np.array2string(path, separator=',', suppress_small=True)
        print(s)

    if show:
        plt.axis('equal')
        for id in pathes:
            path = pathes[id]
            plt.plot(path[:,0], path[:,1], color='b')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
