import re
from manimlib.constants import *
from manimlib.utils.bezier import *
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject


def get_tangent_equation(dirction, point):
    if dirction[1] == 0:
        t = point[0]
        return lambda i: abs(i[0]-t)
    elif dirction[0] == 0:
        t = point[1]
        return lambda i: abs(i[1]-t)
    else:
        s = -dirction[1] / dirction[0]
        t1 = point[0]*s-point[1]
        t2 = 1 / np.sqrt(s**2+1)
        return lambda i: abs(i[1]-s*i[0]+t1)*t2


def one_trim(mob, dirction=LEFT, alpha=0.5, start_point=None, dis_max=None):
    def trim_line(data):
        dis = [get_dis(i) for i in data[::3]]
        flag = [i <= now_dis for i in dis]
        num = len(dis)
        now_line = []
        for i in range(num):
            j = i + 1
            k = j % num
            if flag[i] and flag[k]:
                if False not in (data[i*3] == data[j*3-1]):
                    continue
                now_line.append(list(data[i*3:j*3]))
            elif flag[i] and not flag[k]:
                past = partial_quadratic_bezier_points(
                    data[i*3:j*3], 0, (now_dis-dis[i])/(dis[k]-dis[i]),
                )
                dot.append(past[2])
                now_line.append(past)
            elif flag[k]:
                past = partial_quadratic_bezier_points(
                    data[i*3:j*3], 1-(now_dis-dis[k])/(dis[i]-dis[k]), 1,
                )
                dot.append(past[0])
                now_line.append(past)
        if len(now_line):
            line.extend(now_line)
    x_dis = (lambda i: i[0]) if dirction[0] == 0 else (lambda i: i[1])
    b_point = mob.get_corner(dirction) if start_point is None else start_point
    get_dis = get_tangent_equation(dirction, b_point)
    if dis_max is None:
        dis_max = get_dis(mob.get_corner(-dirction))
    now_dis = dis_max * alpha
    for i in mob:
        one_trim(i, dirction, alpha, b_point, dis_max)
    points = mob.get_points()
    if len(points) == 0:
        return mob
    line = []
    dot = []
    cut = []
    out = []
    last = 0
    flag = True
    for i in range(1, len(points)//3):
        if False in (points[i*3-1] == points[i*3]):
            trim_line(points[last*3:i*3])
            last = i
    trim_line(points[last*3:])
    if len(line) == 0:
        mob.set_points(np.zeros((0, 3)))
        return mob
    if len(dot):
        dot.sort(key=x_dis)
        for i in range(len(dot)//2):
            a = dot[i*2]
            b = dot[i*2+1]
            cut.append([a, mid(a, b), b])
    while True:
        if flag:
            if len(line) + len(cut) == 0:
                break
            out.extend(line.pop(0) if len(line) > 0 else cut.pop(0))
        flag = True
        for index, i in enumerate(line):
            if False not in (i[-1] == out[-1]):
                i.reverse()
            elif False in (i[0] == out[-1]):
                continue
            out.extend(i)
            line.pop(index)
            flag = False
            break
        for index, i in enumerate(cut):
            if False not in (i[-1] == out[-1]):
                i.reverse()
            elif False in (i[0] == out[-1]):
                continue
            out.extend(i)
            cut.pop(index)
            flag = False
            break
    mob.set_points(out)
    return mob


def two_trim(mob, dirction=LEFT, alpha=(0.25, 0.75), start_point=None, dis_max=None):
    def trim_line(data):
        dis = [get_dis(i) for i in data[::3]]
        l_flag = [i < lower_dis for i in dis]
        u_flag = [i <= upper_dis for i in dis]
        num = len(dis)
        now_line = []
        for i in range(num):
            j = i + 1
            k = j % num
            f1 = int(l_flag[i])+u_flag[i]
            f2 = int(l_flag[k])+u_flag[k]
            if f1 + f2 & 3 == 0:
                continue
            elif f1 + f2 == 2:
                if False not in (data[i*3] == data[j*3-1]):
                    continue
                if f1 == f2:
                    now_line.append(list(data[i*3:j*3]))
                    continue
                elif f1 == 2:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3],
                        (lower_dis-dis[i])/(dis[k]-dis[i]),
                        (upper_dis-dis[i])/(dis[k]-dis[i]),
                    )
                    l_dot.append(past[0])
                    u_dot.append(past[2])
                else:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3],
                        1-(upper_dis-dis[k])/(dis[i]-dis[k]),
                        1-(lower_dis-dis[k])/(dis[i]-dis[k]),
                    )
                    l_dot.append(past[2])
                    u_dot.append(past[0])
            elif f1 + f2 == 1:
                if f1:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3], 0, (upper_dis-dis[i])/(dis[k]-dis[i]),
                    )
                    u_dot.append(past[2])
                else:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3], 1-(upper_dis-dis[k])/(dis[i]-dis[k]), 1,
                    )
                    u_dot.append(past[0])
            elif f1 + f2 == 3:
                if f1 == 2:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3], (lower_dis-dis[i])/(dis[k]-dis[i]), 1,
                    )
                    l_dot.append(past[0])
                else:
                    past = partial_quadratic_bezier_points(
                        data[i*3:j*3], 0, 1-(lower_dis-dis[k])/(dis[i]-dis[k]),
                    )
                    l_dot.append(past[2])
            now_line.append(past)
        if len(now_line):
            line.extend(now_line)
    x_dis = (lambda i: i[0]) if dirction[0] == 0 else (lambda i: i[1])
    b_point = mob.get_corner(dirction) if start_point is None else start_point
    get_dis = get_tangent_equation(dirction, b_point)
    if dis_max is None:
        dis_max = get_dis(mob.get_corner(-dirction))
    lower_dis = dis_max * alpha[0]
    upper_dis = dis_max * alpha[1]
    for i in mob:
        two_trim(i, dirction, alpha, b_point, dis_max)
    points = mob.get_points()
    if len(points) == 0:
        return mob
    line = []
    l_dot, u_dot = [], []
    dot = []
    cut = []
    out = []
    last = 0
    flag = True
    for i in range(1, len(points)//3):
        if False in (points[i*3-1] == points[i*3]):
            trim_line(points[last*3:i*3])
            last = i
    trim_line(points[last*3:])
    if len(line) == 0:
        mob.set_points(np.zeros((0, 3)))
        return mob
    if len(l_dot)+len(u_dot):
        l_dot.sort(key=x_dis)
        u_dot.sort(key=x_dis)
        dot = l_dot + u_dot
        for i in range(len(dot)//2):
            a = dot[i*2]
            b = dot[i*2+1]
            cut.append([a, mid(a, b), b])
    while True:
        if flag:
            if len(line) + len(cut) == 0:
                break
            out.extend(line.pop(0) if len(line) > 0 else cut.pop(0))
        flag = True
        for index, i in enumerate(line):
            if False not in (i[-1] == out[-1]):
                i.reverse()
            elif False in (i[0] == out[-1]):
                continue
            out.extend(i)
            line.pop(index)
            flag = False
            break
        for index, i in enumerate(cut):
            if False not in (i[-1] == out[-1]):
                i.reverse()
            elif False in (i[0] == out[-1]):
                continue
            out.extend(i)
            cut.pop(index)
            flag = False
            break
    mob.set_points(out)
    return mob


def more_trim(mob, dirction=LEFT, alpha=[0, 1, 2, 3, 4], start_point=None, dis_max=None):
    if isinstance(alpha, int):
        alpha = [i for i in range(alpha+1)]
    v = alpha[0]
    if v != 0:
        alpha = [i-v for i in alpha]
    v = 1 / alpha[-1]
    alpha = [i*v for i in alpha]
    out = VGroup()
    b_point = mob.get_corner(dirction) if start_point is None else start_point
    get_dis = get_tangent_equation(dirction, b_point)
    if dis_max is None:
        dis_max = get_dis(mob.get_corner(-dirction))
    for i in mob:
        out.add(*more_trim(i, dirction, alpha, b_point, dis_max))
    if mob.get_num_points():
        for i in range(len(alpha)-1):
            out.add(two_trim(mob.copy(), dirction, [alpha[i], alpha[i+1]], b_point, dis_max))
    return out


def c_style_set_color(mob, code, code_dict={}):
    def func(mode, value, text):
        tmp = re.finditer(mode, text)
        for i in tmp:
            b, e = i.span()
            mob[b:e].set_color(color[value])
            text = text[:b]+' '*(e-b)+text[e:]
        return text
    color = {
        'var': '#9cdcfe',
        'fun': '#dcdcaa',
        'num': '#b5cea8',
        'str': '#ce9178',
        'def': '#569cd6',
        'pre': '#c586c0',
        'type': '#4ec9b0',
        'notes': '#6a993e',
    }
    word = {
        'break': color['pre'], 'case': color['pre'], 'continue': color['pre'], 'default': color['pre'],
        'do': color['pre'], 'else': color['pre'], 'for': color['pre'], 'goto': color['pre'], 'if': color['pre'],
        'return': color['pre'], 'switch': color['pre'], 'auto': color['def'], 'char': color['def'],
        'const': color['def'], 'double': color['def'], 'enum': color['def'], 'extern': color['def'],
        'float': color['def'], 'int': color['def'], 'long': color['def'], 'register': color['def'],
        'short': color['def'], 'signed': color['def'], 'sizeof': color['def'], 'static': color['def'],
        'struct': color['def'], 'typedef': color['def'], 'union': color['def'], 'unsigned': color['def'],
        'void': color['def'], 'volatile': color['def'],
    }
    word.update(code_dict)
    flag = False
    defi = False
    code = func(r'//.*|/\*.*?\*/', 'notes', code)
    if code[0] == '#':
        code = func(r'#[a-z]+', 'pre', code)
        code = func(r'<.*>', 'str', code)
        flag = True
    code = func(r'".*?(?<!(?<!\\)\\)"', 'str', code)
    code = func(r"'.*?(?<!(?<!\\)\\)'", 'str', code)
    func(r'[0-9]+', 'num', code)
    tmp = re.finditer(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
    for i in tmp:
        b, e = i.span()
        t = i.group()
        if t in word:
            mob[b:e].set_color(word[t])
        else:
            last = re.compile(r' *\(').match(code, e)
            value = 'def' if flag else ('var' if last is None else 'fun')
            mob[b:e].set_color(color[value])
            if defi == False or flag == False:
                code_dict[t] = color[value]
            defi = True


def get_text_split_points(mob):
    def point_in_polygon(point, points):
        flag = False
        for i, j in zip(points[::3], points[2::3]):
            if point[1] < j[1]:
                if i[1] <= point[1]:
                    if (point[1]-i[1])*(j[0]-i[0]) > (point[0]-i[0])*(j[1]-i[1]):
                        flag = not flag
            elif point[1] < i[1]:
                if (point[1]-i[1])*(j[0]-i[0]) < (point[0]-i[0]) * (j[1]-i[1]):
                    flag = not flag
        return flag
    out = []
    if len(mob):
        for i in mob:
            out.extend(get_text_split_points(i))
    points = mob.get_points()
    if len(points) == 0:
        return out
    rings = []
    last = 0
    for i in range(1, len(points)//3):
        if False in (points[i*3-1] == points[i*3]):
            rings.append(list(points[last*3:i*3]))
            last = i
    rings.append(list(points[last*3:]))
    index = 0
    while index < len(rings):
        point = rings[index][0]
        for i in [*rings[:index], *rings[index+1:]]:
            if point_in_polygon(point, i) & 1:
                i.extend(rings[index])
                del rings[index]
                index -= 1
                break
        index += 1
    out = rings
    return out


def text_split_submob(mob):
    points = get_text_split_points(mob)
    group = VGroup(*[VMobject() for i in range(len(points))])
    group.fade(0).set_stroke(width=0)
    for i, j in zip(group, points):
        i.set_points(j)
    return group
