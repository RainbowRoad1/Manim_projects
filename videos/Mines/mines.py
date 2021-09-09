from manimlib import *
MYNUM_MOBJECT_DATA = {}
FTEXT_MOBJECT_DATA = {}
FTEXT_WORD_SPACE = 0.3688304874239998
FTEXT_ROW_SPACE = 0.670654537728


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
    x_dis = (lambda i: i[0]) if dirction[0] == 0 else (lambda i: i[1])
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


class MyGrid(VMobject):
    CONFIG = {
        "m_line": {
            "stroke_width": 2,
            "stroke_color": WHITE,
        },
        "m_box": {
            "side_length": 1,
            "stroke_width": 0,
            "fill_opacity": 1,
            "fill_color": GREY,
        },
    }

    def __init__(self, col=5, row=5, width=5, build=0, **kwargs):
        digest_config(self, kwargs)
        super().__init__(**kwargs)
        self.n_cols = col
        self.n_rows = row
        self.re_build(build)
        if width != None:
            self.set_width(width)

    def add_number(self, to=ORIGIN, buff=0.05, size=0.5, init=None):
        num = VGroup()
        size = self.box[0].get_width()*size
        if init == None:
            func = lambda i: i
        elif type(init) == list:
            func = lambda i: init[i]
        else:
            func = init
        for i, m in enumerate(self.box):
            num.add(Ftext('%d'%func(i)).scale(size).move_to(m.get_corner(to)-to*buff, to))
        return num

    def sort_func(self, mob=None, mode=0):
        if mob == None:
            mob = self.box
        if mode == 0:
            mob.sort(lambda i: i[0]-i[1]*self.n_cols)
        elif mode == 1:
            mob.sort(lambda i: i[0]**2+i[1]**2)
        elif mode == 2:
            mob.sort(submob_func=lambda i: random.random())
        return self

    def re_build(self, build=0):
        col = self.n_cols
        row = self.n_rows
        box = VGroup()
        for i in range(row):
            for j in range(col):
                box.add(Square(**self.m_box).shift([j, -i, 0]))
        self.box = box
        self.add(box.center())
        if build == 0:
            line = VGroup()
            for i in range(row+1):
                line.add(Line(np.array([0, -i, 0]), np.array([row, -i, 0]), **self.m_line))
            for i in range(col+1):
                line.add(Line(np.array([i, 0, 0]), np.array([i, -col, 0]), **self.m_line))
            self.line = line
            self.add(line.center())
        elif build == 1:
            line = VGroup()
            edge_rect = Rectangle(col, row, **self.m_line)
            for i in range(1, row):
                line.add(Line(np.array([0, -i, 0]), np.array([row, -i, 0]), **self.m_line))
            for i in range(1, col):
                line.add(Line(np.array([i, 0, 0]), np.array([i, -col, 0]), **self.m_line))
            self.line = line
            self.rect = edge_rect
            self.add(line.center(), edge_rect)
        return self


class Game_MS(MyGrid):
    CONFIG = {
        "m_box": {
            "side_length": 0.8,
            "stroke_width": 0,
            "fill_opacity": 1,
            "fill_color": GREY_A,
        },
        "num_color": [
            WHITE, GREEN, BLUE, RED, YELLOW, GREY_BROWN, PURPLE, PINK, LIGHT_BROWN, RED_A,
        ],
    }

    def __init__(self, col=9, row=9, bomb=10, width=5, **kwargs):
        digest_config(self, kwargs)
        super().__init__(col, row, None, 2, **kwargs)
        self.rect = Rectangle(col, row, color=WHITE, stroke_width=5*width/col)
        self.m_num = VGroup()
        self.add(self.rect).set_width(width)
        self.d_index = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        self.n_bomb = bomb
        self.dig_buf = []
        self.n_air = col*row
        self.v_map = col*row*[1]
        self.v_mask = col*row*[1]
        self.not_dig = [i for i in range(col*row)]
        self.dig_func = self.set_bomb

    def edge(self, x, y):
        return 0 <= x and x < self.n_cols and 0 <= y and y < self.n_rows

    def run_func(self, index, anim=False):
        self.dig_buf.clear()
        x, y = index % self.n_cols, index // self.n_cols
        if self.v_mask[index] == 0:
            for dx, dy in [[x+i, y+j] for i, j in self.d_index]:
                if self.edge(dx, dy):
                    self.dig_func(dx, dy)
        else:
            self.dig_func(x, y)
        if anim:
            return FadeOut(VGroup(*[self.box[j] for j in self.dig_buf]),
                run_time=min(3, 0.05*len(self.dig_buf)), lag_ratio=0.2,
            )
        return self.dig_buf

    def set_bomb(self, x=0, y=0, area=2):
        col = self.n_cols
        n_max = col*self.n_rows
        self.n_air = n_max+self.n_bomb
        self.v_map = n_max*[0]
        self.v_mask = n_max*[1]
        self.not_dig = [i for i in range(n_max)]
        self.dig_func = self.set_bomb_func
        for i in range(self.n_bomb):
            index = random.randint(0, n_max-1)
            while self.v_map[index] == 9 or (abs(x-index % col) < area and abs(y-index//col) < area):
                index = random.randint(0, n_max-1)
            self.v_map[index] = 0
            self.dig(index % col, index//col)
            self.v_map[index] = 9
            self.v_mask[index] = 1
        self.dig_func = self.dig
        m_num = self.add_number(size=1, init=self.v_map)
        wid = self.get_width()/col*0.2
        for i, m in enumerate(m_num):
            num = self.v_map[i]
            m.set_color(self.num_color[num])
            if num == 0:
                m.re_text(' ')
            if num == 9:
                m.become(Text('💣', stroke_width=1, fill_color=BLACK).replace(self.box[i]))
        self.dig_buf.clear()
        self.dig(x, y)
        self.m_num = m_num
        self.add_to_back(self.m_num)
        return self.dig_buf

    def set_bomb_func(self, x, y):
        i = x+self.n_cols*y
        self.dig_buf.append(i)
        if self.v_map[i] < 9:
            self.v_map[i] += 1

    def set_bomb_custom(self, bomb):
        self.n_bomb = len(bomb)
        col = self.n_cols
        n_max = col*self.n_rows
        self.n_air = n_max+self.n_bomb
        self.v_map = n_max*[0]
        self.v_mask = n_max*[1]
        self.not_dig = [i for i in range(n_max)]
        self.dig_func = self.set_bomb_func
        for i in bomb:
            self.v_map[i] = 0
            self.dig(i % col, i//col)
            self.v_map[i] = 9
            self.v_mask[i] = 1
        self.dig_func = self.dig
        self.m_num = self.add_number(size=1, init=self.v_map)
        wid = self.get_width()/col*0.2
        for i, m in enumerate(self.m_num):
            num = self.v_map[i]
            m.set_color(self.num_color[num])
            if num == 0:
                m.re_text(' ')
            if num == 9:
                m.become(Text('💣', stroke_width=1, fill_color=BLACK).replace(self.box[i]))
        self.add_to_back(self.m_num)
        self.dig_buf.clear()
        return self.m_num

    def dig(self, x, y):
        index = x+self.n_cols*y
        if self.v_mask[index] == 1:
            self.dig_buf.append(index)
            self.v_mask[index] = 0
            self.n_air -= 1
            if self.v_map[index] == 0:
                for i in self.d_index:
                    m, n = x+i[0], y+i[1]
                    if 0 <= m and m < self.n_cols and 0 <= n and n < self.n_rows:
                        self.dig_func(m, n)


class MyNum(VMobject):
    CONFIG = {
        "fill_opacity": 1.0,
        "stroke_width": 0,
        "color": WHITE,
        "size": 1,
        "unit": 0.1,
        "n_round": 0,
        "edge_to_fix": LEFT,
    }

    def __init__(self, num=0, n_round=0, **kwargs):
        self.CONFIG.update(kwargs)
        super().__init__(**kwargs)
        if len(MYNUM_MOBJECT_DATA) == 0:
            self.init_mobject_data()
        self.n_round = n_round
        self.text = ''
        self.re_num(num)

    def init_mobject_data(self):
        for i in '0123456789.-':
            MYNUM_MOBJECT_DATA[i] = Text(i)[0]

    def re_num(self, new_num):
        new = str('%%.0%df' % self.n_round % new_num)
        old = self.text
        n_l = len(new)
        o_l = len(old)
        move_to_point = self.get_corner(self.edge_to_fix)
        for i in range(min(n_l, o_l)):
            if new[i] != old[i]:
                self[i].match_points(MYNUM_MOBJECT_DATA[new[i]])
                self[i].scale(self.size)
        if n_l < o_l:
            self.remove(*self[n_l:])
        elif n_l > o_l:
            tmp = VMobject().match_style(self)
            for i in new[o_l:]:
                mob = tmp.copy()
                mob.match_points(MYNUM_MOBJECT_DATA[i])
                self.add(mob.scale(self.size))
        self.arrange(buff=self.unit*self.size, aligned_edge=DOWN, center=False)
        if '-' in new:
            self[new.find('-')].set_y(self.get_y())
        self.move_to(move_to_point, self.edge_to_fix)
        self.text = new
        self.number = round(new_num, self.n_round)
        return self

    def re_num_d(self, num):
        return self.re_num(self.number+num)

    def scale(self, v, min_v=1e-8, **kwargs):
        v = max(v, min_v)
        self.apply_points_function(
            lambda points: v * points,
            works_on_bounding_box=True,
            **kwargs
        )
        self.size *= v
        return self

    def num_track(self, func):
        self.add_updater(lambda i: i.re_num(func()))


class Ftext(VMobject):
    CONFIG = {
        "fill_opacity": 1.0,
        "stroke_width": 0,
        "color": WHITE,
        "size": 1,
        "w_space": FTEXT_WORD_SPACE,
        "r_space": FTEXT_ROW_SPACE,
    }

    def __init__(self, text, center=True, **kwargs):
        digest_config(self, kwargs)
        super().__init__(**kwargs)
        if not FTEXT_MOBJECT_DATA:
            self.init_mobject_data()
        self.re_text(text, center)
        if 't2c' in kwargs:
            self.set_color_by_t2c(kwargs['t2c'])

    def init_mobject_data(self):
        text = '''0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*_+-=)(}{][:;"',.<>/?|\`~'''
        m_past = Text(
            '|0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[1:]
        mob = Text('''!@#$%^&*_+-=)(}{][:;"',.<>/?|\`~''').add_to_back(*m_past)
        for i, m in enumerate(mob):
            FTEXT_MOBJECT_DATA[text[i]] = (m, m.get_y())
        FTEXT_MOBJECT_DATA['\n'] = (
            Dot(radius=0, fill_opacity=0, stroke_opacity=0), 0)
        FTEXT_MOBJECT_DATA[' '] = (
            Dot(radius=0, fill_opacity=0, stroke_opacity=0), 0)

    def re_text(self, text, center=True):
        pos = self.get_center()
        self.remove(*self).add(*self.new_text(text))
        if center:
            self.move_to(pos)
        self.text = text
        return self

    def new_text(self, text):
        if text.find('\t') != -1:
            text = text.replace('\t', '    ')
        out = VMobject()
        tmp = VMobject().match_style(self)
        c, r = 0, 0
        for i in text:
            mob, t = tmp.copy(), FTEXT_MOBJECT_DATA[i]
            mob.set_opacity(0) if i in ' \n' else None
            mob.set_points(t[0].get_all_points())
            c = max(c-(i == '\n'), 0)
            mob.scale(self.size).move_to(
                np.array([c*self.w_space, t[1]*self.size-r*self.r_space, 0])
            )
            c, r = (c+1, r) if i != '\n' else (0, r+1)
            out.add(mob)
        return out

    def scale(self, v, min_v=1e-8, **kwargs):
        v = max(v, min_v)
        self.apply_points_function(
            lambda points: v * points,
            works_on_bounding_box=True,
            **kwargs
        )
        self.size *= v
        self.w_space *= v
        self.r_space *= v
        return self

    def find_indexes(self, word):
        m = re.match(r'\[([0-9\-]{0,}):([0-9\-]{0,})\]', word)
        if m:
            start = int(m.group(1)) if m.group(1) != '' else 0
            end = int(m.group(2)) if m.group(2) != '' else len(self.text)
            start = len(self.text) + start if start < 0 else start
            end = len(self.text) + end if end < 0 else end
            return [(start, end)]

        indexes = []
        index = self.text.find(word)
        while index != -1:
            indexes.append((index, index + len(word)))
            index = self.text.find(word, index + len(word))
        return indexes

    def set_color_by_t2c(self, t2c):
        for word, color in list(t2c.items()):
            for start, end in self.find_indexes(word):
                self[start:end].set_color(color)


class NumberToValue(Animation):
    def __init__(self, mob, value, **kwargs):
        self.v_new = value
        self.v_old = mob.number
        self.n_round = mob.n_round
        super().__init__(mob, **kwargs)

    def interpolate_mobject(self, alpha):
        new = round(interpolate(self.v_old, self.v_new, alpha), self.n_round)
        self.mobject.re_num(new)


class CommandLine(VMobject):
    def __init__(self, col_max=80, width=None, **kwargs):
        super().__init__(**kwargs)
        if width is None:
            width = FRAME_WIDTH
        self.unit_w = width / col_max
        self.init_mobject_data()
        self.basic_point = lambda: np.array([self.unit_w/2-FRAME_X_RADIUS, FRAME_Y_RADIUS-self.unit_h/2, 0])
        self.m_text = VGroup()
        self.cursor = Rectangle(self.unit_w, self.unit_h, fill_opacity=1, stroke_width=0).move_to(self.get_grid_place())
        self.cursor_pos = [0, 0]
        self.text = []
        self.code_dict = {}
        self.add(self.m_text, self.cursor)

    def init_mobject_data(self):
        self.mobject_data = {}
        text = '''0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*_+-=)(}{][:;"',.<>/?|\`~'''
        m_past = Text(
            '|0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[1:]
        mob = Text('''!@#$%^&*_+-=)(}{][:;"',.<>/?|\`~''').add_to_back(*m_past)
        for i in mob:
            i.set_x(0)
        mob.set_width(self.unit_w)
        self.unit_h = mob.get_height()
        for i, m in enumerate(mob):
            self.mobject_data[text[i]] = (m, m.get_y())
        self.mobject_data['\n'] = (
            Dot(radius=0, fill_opacity=0, stroke_opacity=0), 0)
        self.mobject_data[' '] = (
            Dot(radius=0, fill_opacity=0, stroke_opacity=0), 0)

    def new_text(self, text, row=0, col=0, insert=False):
        if text.find('\t') != -1:
            text = text.replace('\t', '    ')
        out = VGroup()
        tmp = VMobject(fill_opacity=1, stroke_width=0)
        for i in text:
            out.add(self.mobject_data[i][0].copy())
        if len(self.text) <= row:
            self.m_text.add(out)
            self.text.append(text)
        elif insert:
            self.m_text[row:].shift(DOWN*self.unit_h)
            self.m_text.submobjects.insert(row, out)
            out.parents.append(self.m_text)
            self.m_text.assemble_family()
            self.text.insert(row, text)
        else:
            self.m_text[row].set_submobjects(out)
            self.text[row] = text
        self.typesetting(row, col)
        return out

    def add_text(self, text):
        return self.new_text(text, len(self.text))

    def new_word(self, word, x=0, y=0):
        out = VGroup()
        for i in word:
            out.add(self.mobject_data[i][0].copy().move_to(self.get_grid_place(x, y)+self.mobject_data[i][1]*UP))
            if i == '\n':
                x, y = 0, y+1
            else:
                x += 1
        return out

    def get_grid_place(self, x=0, y=0):
        return np.array([x*self.unit_w, -y*self.unit_h, 0])+self.basic_point()

    def typesetting(self, row=0, col=0):
        for i, c in enumerate(self.text[row]):
            col = max(col-(c == '\n'), 0)
            self.m_text[row][i].move_to(np.array([col*self.unit_w, self.mobject_data[c][1]-row*self.unit_h, 0])+self.basic_point())
            col, row = (col+1, row) if c != '\n' else (0, row+1)

    def cursor_next(self):
        col, row = self.cursor_pos
        if row < len(self.text):
            text = self.text[row]
            if col < len(text):
                now = self.m_text[row][col]
                if text[col] == '\n':
                    self.cursor_pos = [0, row+1]
                    self.cursor.move_to(self.get_grid_place(y=row+1))
                    return True
                self.cursor_pos[0] += 1
                self.cursor.shift(RIGHT*self.unit_w)
                return now
        return False

    def cursor_shift(self, v):
        self.cursor_pos[0] += int(v[0])
        self.cursor_pos[1] -= int(v[1])
        self.cursor.move_to(self.get_grid_place(*self.cursor_pos))

    def cursor_move_to(self, x=0, y=0):
        self.cursor_pos = [x, y]
        self.cursor.move_to(self.get_grid_place(x, y))

    def cursor_add(self, add):
        col, row = self.cursor_pos
        if row < len(self.text):
            text = self.text[row]
            mob = self.m_text[row]
            if add == '\n':
                text = text[:col]+add
                self.new_text(text[col:])
                self.m_text[row+1:].shift(DOWN*self.unit_h)
                return
            text = text[:col]+add+text[col:]
            mob[col:].shift(RIGHT*self.unit_w)
            tmp = self.mobject_data[add][0].copy()
            tmp.move_to(self.get_grid_place(col, row)+self.mobject_data[add][1]*UP)
            tmp.parents.append(mob)
            mob.submobjects.insert(col, tmp)
            mob.assemble_family()
            self.cursor_shift(RIGHT)

    def cursor_del(self):
        col, row = self.cursor_pos
        if row < len(self.text):
            text = self.text[row]
            mob = self.m_text[row]
            text = text[:col-1]+text[col:]
            mob[col:].shift(LEFT*self.unit_w)
            mob.remove(mob[col])
            self.cursor_shift(LEFT)

    def set_color_c_style(self, index=None):
        if index == None:
            for mob, code in zip(self.m_text, self.text):
                c_style_set_color(mob, code, self.code_dict)
        else:
            c_style_set_color(self.m_text[index], self.text[index], self.code_dict)


class Console(CommandLine):
    def __init__(self, cols=20, lines=8, width=6, **kwargs):
        super().__init__(cols, width, **kwargs)
        self.background = Rectangle(
            cols*self.unit_w, lines*self.unit_h,
            stroke_width=1, stroke_color=WHITE,
            color='#1e1e1e', fill_opacity=1,
        )
        self.bg_title = Rectangle(
            cols*self.unit_w, 0.5,
            stroke_width=1, stroke_color=WHITE,
            color=WHITE, fill_opacity=1,
        ).move_to(self.background.get_top(), DOWN)
        a = Line(LEFT*0.1, RIGHT*0.1, color=BLACK, stroke_width=1.5).shift(LEFT*0.75)
        b = Rectangle(width=0.2, height=0.2, color=BLACK, stroke_width=1.5)
        c = Line(DL*0.1, UR*0.1, color=BLACK, stroke_width=1.5)
        c.add(c.copy().rotate(PI*0.5)).shift(RIGHT*0.75)
        red_button = Dot(radius=0.1, color='#ff5f56').shift(LEFT * 0.3)
        green_button = Dot(radius=0.1, color='#27c93f').shift(RIGHT * 0.3)
        yellow_button = Dot(radius=0.1, color='#ffbd2e')
        self.button_old = VGroup(red_button, yellow_button, green_button)
        self.button_old.move_to(self.bg_title.get_left()+RIGHT*0.15, LEFT)
        self.botton = VGroup(a, b, c).move_to(self.bg_title.get_right()+LEFT*0.25, RIGHT)
        self.basic_point = lambda: self.background.get_corner(UL)+np.array([self.unit_w/2, -self.unit_h/2, 0])
        self.cursor.move_to(self.basic_point())
        self.add_to_back(self.background, self.bg_title, self.button_old, self.botton)


class anim_CLI_types(Animation):
    CONFIG = {
        'rate_func': rush_from,
    }

    def __init__(self, cli, line=0, **kwargs):
        self.submob = list(cli.m_text[line].submobjects)
        self.line = line
        self.last_num = 0
        self.cli = cli
        super().__init__(cli.m_text[line], **kwargs)
    
    def finish(self):
        super().finish()
        self.mobject.set_submobjects(self.submob)
        self.cli.cursor_move_to(y=self.line+1)

    def interpolate_mobject(self, alpha):
        n_submobs = len(self.submob)
        index = int(alpha * n_submobs)
        self.cli.cursor_shift([index-self.last_num, 0])
        self.last_num = index
        self.mobject.set_submobjects(self.submob[:index]+[self.cli.cursor])


class RouteCreation(Animation):
    def __init__(self, mobject, rounds=lambda i: (i, i*2), close_path=None, **kwargs):
        super().__init__(mobject, **kwargs)
        self.get_bounds = rounds
        if close_path is None and mobject.get_fill_opacity() or close_path:
            self.close_path = True
        else:
            self.close_path = False

    def interpolate_submobject(self, a, b, alpha):
        a.set_points(self.get_route(b, alpha))
        if self.close_path and len(a.get_points()):
            a.close_path()

    def modf(self, i):
        d = 0
        while d + 1 < i:
            d += 1
        return d, i-d

    def get_route(self, vmobject, alpha):
        a, b = self.get_bounds(alpha)
        if a > b:
            a, b = b, a
        v_p = vmobject.get_points()
        if len(v_p) == 0:
            return np.zeros((0, 3))
        if a + 1 <= b:
            return v_p
        a, a1 = self.modf(a)
        b, b1 = self.modf(b)
        num_curves = vmobject.get_num_curves()
        a1, a2 = self.modf(a1 * num_curves)
        b1, b2 = self.modf(b1 * num_curves)
        num_curves *= 3
        a1 *= 3
        b1 *= 3
        if a == b and a1 == b1:
            out = v_p.copy()
            out[0:3] = partial_quadratic_bezier_points(v_p[a1:a1+3], a2, b2)
            out[3:] = out[2]
            return out
        else:
            out = []
            out.extend(partial_quadratic_bezier_points(v_p[a1:a1+3], a2, 1))
            a1 += 3
            if a1 < b1:
                out.extend(v_p[a1:b1])
            elif b1 < a1:
                out.extend(v_p[a1:])
                out.extend(v_p[:b1])
            out.extend(partial_quadratic_bezier_points(v_p[b1:b1+3], 0, b2))
            return np.array(out)


class TrimCreation(Animation):
    def __init__(self, mobject, dirction=LEFT, **kwargs):
        super().__init__(mobject, **kwargs)
        self.b_point = mobject.get_corner(dirction)
        self.get_dis = get_tangent_equation(dirction, self.b_point)
        self.dis_max = self.get_dis(mobject.get_corner(-dirction))
        self.dot_dis = (lambda i: i[0]) if dirction[0] == 0 else (lambda i: i[1])
        self.is_fill = mobject.get_fill_opacity()

    def interpolate_submobject(self, a, b, alpha):
        a.set_points(self.trim(b, alpha))
        if self.is_fill:
            a.set_fill(opacity=self.is_fill)


    def trim(self, vmobject, alpha):
        def trim_line(data):
            dis = [self.get_dis(i) for i in data[::3]]
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
        now_dis = self.dis_max * alpha
        points = vmobject.get_points().copy()
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
            return np.zeros((0, 3))
        if len(dot):
            dot.sort(key=self.dot_dis)
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
        return out


class Reveal(Animation):
    CONFIG = {
        "run_time": 2,
        "rate_func": lambda i: 1-(1-i)**4,
    }

    def __init__(self, mobject, dirction=DOWN, buff=0, **kwargs):
        super().__init__(mobject, **kwargs)
        center = mobject.get_center()
        dirction = -dirction
        self.start_point = mobject.get_corner(dirction)
        self.dis_max = get_tangent_equation(dirction, self.start_point)(mobject.get_corner(-dirction))+buff
        self.dis_move = dirction*self.dis_max
        self.dirction = dirction

    def begin(self):
        super().begin()
        self.starting_mobject.shift(-self.dis_move)

    def interpolate(self, alpha):
        self.mobject.become(self.starting_mobject).shift(self.rate_func(alpha)*self.dis_move)
        self.mobject = one_trim(self.mobject, self.dirction, 1, self.start_point, self.dis_max)


class Glitch(Animation):
    def __init__(self, mobject, move1=0.05, move2=0.05, **kwargs):
        super().__init__(mobject, **kwargs)
        self.mob_copy = self.mobject.copy()
        self.move1 = move1
        self.move2 = move2

    def finish(self):
        self.mobject.become(self.mob_copy)

    def interpolate(self, null):
        pos = [random.randint(0, 10000)
                for i in range(random.randint(6, 12))]
        pos.sort(key=lambda i: i)
        d2 = RIGHT
        dirction = UP
        m = more_trim(self.mob_copy.copy(), dirction, pos)
        for i in m:
            i.shift(d2*(random.random()*10-5)*self.move2)
        t = random.random()*PI*2
        d3 = np.array([self.move1*np.cos(t), self.move1*np.sin(t), 0])
        l = m.copy().shift(d3).set_color(BLUE)
        r = m.copy().shift(-d3).set_color(RED)
        self.mobject.become(VGroup(l, r, m))


class FadeOutRandom(AnimationGroup):
    def __init__(self, mobject, move=0.5, **kwargs):
        anim = []
        shift = [UP, DOWN, LEFT, RIGHT]
        for i in mobject.get_family():
            if len(i) == 0:
                anim.append(FadeOut(i, random.random()*move*shift[random.randint(0, 3)]))
        super().__init__(*anim, **kwargs)


class FadeInRandom(AnimationGroup):
    def __init__(self, mobject, move=0.5, **kwargs):
        anim = []
        shift = [UP, DOWN, LEFT, RIGHT]
        for i in mobject.get_family():
            if len(i) == 0:
                anim.append(FadeIn(i, random.random()*move*shift[random.randint(0, 3)]))
        super().__init__(*anim, **kwargs)


class Shift(Animation):
    def __init__(self, mobject, shift=RIGHT, **kwargs):
        self.shift = shift
        self.total = 0
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        self.mobject.shift((alpha-self.total)*self.shift)
        self.total = alpha


class Scene_1(Scene):
    def construct(self):
        mob = Game_MS(5, 5, 4, width=6).to_edge(UP)
        mob.set_bomb_custom([2, 7, 13, 19])
        text1 = VMobject(fill_opacity=1, stroke_width=0, color=BLUE)
        text2 = VMobject(fill_opacity=1, stroke_width=0, color=RED)
        text2.set_points(Text('Game over').next_to(mob).get_all_points())
        text1.set_points(Text('You win').move_to(text2).get_all_points())
        rect = Square(mob.box[0].get_width()/0.8, color=RED, stroke_width=mob.rect.get_stroke_width()).move_to(mob.box[0])
        text3 = Text('大小: 5x5\n数量: 4', font='思源黑体').next_to(mob)
        text3[8:].shift(DOWN*0.2)
        mob.sort_func(mode=2)
        self.play(
            FadeIn(mob.box, lag_ratio=0.2),
            ShowCreation(mob.rect),
            RouteCreation(rect, lambda i: (1, 1-i)),
            Reveal(text3, LEFT, rate_func=squish_rate_func(lambda i: 1-(1-i)**4, 0.5, 1)),
            run_time=2,
        )
        mob.sort_func()
        self.add(mob, rect)
        for i in [22, 8, 4, 14]:
            self.play(ApplyMethod(rect.move_to, mob.box[i], run_time=0.5))
            self.play(FadeOut(
                VGroup(*[mob.box[i] for i in mob.run_func(i)]),
                run_time=0.25*len(mob.dig_buf), lag_ratio=0.5,
            ))
        self.play(
            ApplyMethod(rect.move_to, mob.box[24], rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeOut(mob.box[24], rate_func=squish_rate_func(smooth, 0.5, 1)),
            Reveal(text3, LEFT, rate_func=lambda i: smooth(1-i)),
            run_time=1,
        )
        self.play(Reveal(text1, run_time=1))
        self.wait(1)
        self.play(
            RouteCreation(rect, lambda i: (1-rush_into(i), 2-slow_into(i))),
            text1.animate.set_color(GREY),
            FadeIn(mob.box[24]),
        )
        self.add(rect).play(ApplyMethod(rect.move_to, mob.box[19], run_time=0.5))
        mob.box[19].set_color(RED)
        a = VGroup(*[m for i, m in enumerate(mob.box) if mob.v_map[i] == 9])
        a.sort(lambda i: i[1])
        self.play(FadeOut(a), run_time=1, lag_ratio=0.5)
        self.play(Transform(text1, text2, run_time=1))
        self.wait()
        self.play(
            *[Uncreate(i, lag_ratio=0.05) for i in [mob.m_num, mob.box[24], text1]],
            RouteCreation(mob.rect, lambda i: (1-0.2*rush_into(i), 2-1.2*rush_into(i))),
            RouteCreation(rect, lambda i: (1+2*rush_into(i), 2+rush_into(i))),
            run_time=1.5,
        )
        self.wait(0.5)


class Scene_2(Scene):
    def construct(self):
        text_list = [
            '首先, 先来分析扫雷的游戏规则',
            '雷区中, 数字代表周围有多少颗雷',
            '通过简单的推理, 即可避免踩雷',
            '游戏基本操作有左击, 右击, 双击',
            '(回到上一次状态...)',
            '挖开一个格子的时候, 如果这个格子是空地',
            '意味着周围安全, 会自动挖开周围的格子',
            '如果挖开的格子还是空地, 则重复以上动作',
            '直到没有空地为止...',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        t_text[1][5:7].set_color(BLUE)
        t_text[1][9:11].set_color(RED)
        s_text = t_text[0].copy().fix_in_frame()
        mob = Game_MS(9, 9, 10, width=6).to_edge(UP)
        mob.set_bomb_custom([3, 10, 13, 19, 22, 30, 32, 42, 52, 62])
        box_unit_w = mob.box[0].get_width()/0.8
        box_line_w = mob.rect.get_stroke_width()
        rect = Square(box_unit_w, color=RED, stroke_width=box_line_w).move_to(mob.box[0])
        h_help = Text('大小: 9x9\n数量: 10', font='思源黑体').next_to(mob)
        h_help[8:].shift(DOWN*0.2)
        t_win = Ftext('You win', color=BLUE).next_to(mob)
        frame = self.camera.frame
        line = Line([0, 0, 0], [0, 1.5, 0], color=BLACK)
        line2 = Line([-0.5, 0, 0], [0.5, 0, 0], color=BLACK)
        flag_flag = RegularPolygon(3, color=RED, fill_opacity=1, stroke_width=0).rotate(-PI/2).scale(0.5).move_to(line, UL)
        m_flag = VGroup(line, line2, flag_flag)
        bomb_list = [30, 32, 42, 52, 62, 22, 13, 19, 10, 3]
        g_flag = VGroup(*[m_flag.copy().replace(mob.box[i]).scale(0.75) for i in bomb_list])
        mob.sort_func(mode=2)
        self.play(
            Reveal(h_help, LEFT, rate_func=squish_rate_func(lambda i: 1-(1-i)**4, 0.5, 1)),
            ShowCreation(mob.rect),
            FadeIn(mob.box, lag_ratio=0.2),
            RouteCreation(rect, lambda i: (1, 1-i)),
            Write(s_text),
            run_time=2,
        )
        mob.sort_func()
        self.add(mob, rect, s_text)
        self.play(ApplyMethod(rect.move_to, mob.box[55], run_time=0.5))
        self.play(
            mob.run_func(55, anim=True),
            Reveal(h_help, LEFT, rate_func=lambda i: smooth(1-i), run_time=1),
        )
        
        self.play(
            frame.animate.scale(0.5).move_to(mob.box[38]),
            Transform(s_text, t_text[1]),
        )
        r_air = rect.copy().move_to(mob.box[38]).set_color(BLUE).scale(0.9)
        r_edge = VGroup(*[rect.copy().scale(0.9).move_to(mob.box[i]) for i in [28, 29, 30, 37, 39, 46, 47, 48]])
        r_bomb = r_edge[2]
        t_bomb = Text(
            '必定是雷!', font='思源黑体', color=BLUE_B,
        ).set_width(box_unit_w*1.2).next_to(r_bomb, DOWN, buff=0.01)
        r_safe = r_air.copy().move_to(mob.box[31]).set_color(GREEN)
        t_safe = Text(
            '此处安全.', font='思源黑体', color=GREEN_B,
        ).set_width(box_unit_w*1.2).next_to(r_safe, DOWN, buff=0.01)
        self.play(ShowCreation(r_air))
        self.add(r_edge, r_air)
        self.play(RouteCreation(r_edge, run_time=2, lag_ratio=0.05))
        r_edge.remove(r_bomb)
        self.add(r_bomb)
        self.play(FadeOut(r_edge), Write(t_bomb), run_time=1)
        self.play(
            RouteCreation(g_flag[0], lambda i: (0, i), lag_ratio=0),
            Transform(s_text, t_text[2]),
        )
        self.play(r_air.animate.shift(box_unit_w*RIGHT))
        self.play(Transform(t_bomb, t_safe), ShowCreation(r_safe))
        self.wait()
        
        for i in bomb_list:
            mob.v_mask[i] = 2
        mouse = VGroup(
            Polygon(LEFT, UP, ORIGIN),
            Polygon(RIGHT, UP, ORIGIN),
            Polygon(LEFT, DOWN*2, RIGHT),
        ).set_fill(opacity=1, color=WHITE).set_stroke(color=BLACK)
        mouse[0].data['points'][1][:] = UL
        mouse[1].data['points'][1][:] = UR
        mouse[2].data['points'][1][:] = DL+DOWN
        mouse[2].data['points'][4][:] = DR+DOWN
        mouse.move_to(mob.get_corner(DL)+UR*0.3, DL)
        r_flag = VGroup(*[r_air.copy().set_fill(opacity=0.5).move_to(mob.box[i]) for i in [32, 42]]).set_color(RED)
        r_open = VGroup(*[r_air.copy().set_fill(opacity=0.5).move_to(mob.box[i]) for i in [23, 24, 25, 34, 43]])
        o_text = VGroup(*[Text(i, font='思源黑体') for i in ['左击', '右击', '双击']])
        o_t_1 = Text('挖开格子\n推进游戏进度', font='思源黑体')
        o_t_2 = Text('标记格子\n辅助双击操作', font='思源黑体')
        o_t_3 = Text('当数字等于周围标记格子数量时,\n对周围未标记格子都执行左击操作', font='思源黑体')
        o_text.scale(1.8).arrange(DOWN, buff=1.2).next_to(mob, LEFT, buff=3.2)
        o_t_1[5:11].scale(2/3, about_edge=UL).shift(DOWN*0.3)
        o_t_2[5:11].scale(2/3, about_edge=UL).shift(DOWN*0.3)
        o_t_3[16:].shift(DOWN*0.3)
        o_t_3[7:11].set_color(RED)
        o_t_3[19:24].set_color(BLUE)
        o_t_1.next_to(o_text[0])
        o_t_2.next_to(o_text[1])
        o_t_3.move_to(o_text[2].get_center()+RIGHT).scale(0.6)
        self.play(
            *[Uncreate(i, remover=True) for i in [r_bomb, r_air, r_safe]],
            ApplyMethod(rect.move_to, mob.box[31]),
            frame.animate.scale(2).center(),
            FadeOut(t_bomb, remover=True),
            run_time=2,
        )
        self.play(
            frame.animate.shift(2.5*LEFT+0.5*UP),
            Transform(s_text, t_text[3]),
            FadeIn(mouse),
        )
        self.wait()
        s_text.unfix_from_frame().shift(2.5*LEFT+0.5*UP)
        self.play(
            FadeOut(s_text),
            ReplacementTransform(s_text[7:9], o_text[0]),
            ReplacementTransform(s_text[11:13], o_text[1]),
            ReplacementTransform(s_text[15:17], o_text[2]),
        )
        self.play(
            ApplyMethod(mouse[0].set_fill, ORANGE, rate_func=there_and_back),
            Write(o_t_1, run_time=1),
        )
        self.play(mob.run_func(31, anim=True), run_time=1)
        self.play(
            ApplyMethod(mouse.shift, RIGHT*0.2, rate_func=there_and_back),
            rect.animate.move_to(mob.box[32]),
        )
        self.play(
            ApplyMethod(mouse[1].set_fill, ORANGE, rate_func=there_and_back),
            Write(o_t_2, run_time=1),
        )
        self.play(RouteCreation(g_flag[1], lambda i: (0, i), lag_ratio=0))
        self.play(
            ApplyMethod(mouse.shift, DR*0.2, rate_func=there_and_back),
            rect.animate.move_to(mob.box[42]),
        )
        self.play(
            ApplyMethod(mouse[1].set_fill, ORANGE, rate_func=there_and_back),
            RouteCreation(g_flag[2], lambda i: (0, i), lag_ratio=0, remover=True),
        )
        self.play(
            ApplyMethod(mouse[1].set_fill, ORANGE, rate_func=there_and_back),
            Uncreate(g_flag[2].copy(), lag_ratio=0, remover=True), run_time=1,
        )
        self.play(
            ApplyMethod(mouse[1].set_fill, ORANGE, rate_func=there_and_back),
            RouteCreation(g_flag[2], lambda i: (0, i), lag_ratio=0), run_time=1,
        )
        self.play(
            ApplyMethod(mouse.shift, UP*0.2, rate_func=there_and_back),
            rect.animate.move_to(mob.box[33]),
        )
        self.play(
            ApplyMethod(mouse[0].set_fill, ORANGE, rate_func=there_and_back),
            mob.run_func(33, anim=True), run_time=1,
        )
        self.play(
            ApplyMethod(mouse[:2].set_fill, ORANGE, rate_func=there_and_back),
            Write(o_t_3[:15]), o_text[2].animate.set_opacity(0.3),
        )
        self.play(RouteCreation(r_flag, lag_ratio=0.1), run_time=2)
        self.play(Write(o_t_3[15:]), RouteCreation(r_open, lag_ratio=0.1), run_time=2)
        self.play(
            FadeOut(r_flag, rate_func=rush_into),
            FadeOut(r_open, rate_func=rush_into),
            mob.run_func(33, anim=True),
            FadeOut(rect.copy(), scale=3, remover=True, rate_func=slow_into),
        )
        self.wait(2)
        
        s_text = t_text[4].copy().set_color(GREY)
        dig_buf = VGroup(*[mob.box[j] for j in mob.dig_buf])
        r_air = r_open[0].copy().set_color(GREEN).move_to(mob.box[25])
        r_air_2 = VGroup(*[r_air.copy().set_color(GREEN_B).move_to(mob.box[i]) for i in [15, 16, 17, 26, 35]])
        r_air_3 = VGroup(*[r_air.copy().set_color(GREEN_A).move_to(mob.box[i]) for i in [6, 7, 8]])
        r_air_a = VGroup(r_air, r_air_2, r_air_3)
        for i in [52, 53, 62]:
            mob.m_num[i].set_opacity(0)
        self.play(
            *[Uncreate(i, remover=True) for i in [o_text, o_t_1, o_t_2, o_t_3, mouse]],
            FadeOut(dig_buf, lag_ratio=0.2, rate_func=lambda i: smooth(1-i)),
            frame.animate.move_to(mob.box[33]).scale(0.5),
            Write(s_text),
        )
        self.add(dig_buf.set_fill(opacity=1))
        self.play(
            *[mob.box[i].animate.set_opacity(0.25) for i in [52, 53, 62]],
            mob.rect.animate.set_stroke(opacity=0.25),
            Transform(s_text, t_text[5]),
            RouteCreation(r_open, lag_ratio=0.1, run_time=2),
        )
        self.play(
            *[FadeOut(mob.box[i]) for i in [23, 24, 25, 34, 43]],
            FadeOut(r_open, lag_ratio=0.1),
        )
        self.play(
            s_text[18:].animate.set_color(GREEN),
            ShowCreation(r_air),
        )
        self.play(
            RouteCreation(
                r_air.copy().set_width(3*box_unit_w), lambda i: (1.2*rush_into(i), 1.2*slow_into(i)),
                remover=True, run_time=2, close_path=False,
            ),
            Transform(s_text, t_text[6]),
        )
        self.play(
            FadeOut(r_air.copy(), scale=3, remover=True, rate_func=slow_into),
            *[FadeOut(mob.box[i]) for i in [15, 16, 17, 26, 35]],
        )
        self.play(
            RouteCreation(r_air_2, lag_ratio=0.1, run_time=2),
            Transform(s_text, t_text[7], rate_func=squish_rate_func(smooth, 0.5, 1)),
        )
        self.play(
            *[FadeOut(i.copy(), scale=3, remover=True, rate_func=slow_into) for i in r_air_2],
            *[FadeOut(mob.box[i]) for i in [5, 6, 7, 8, 14, 44]],
        )
        self.play(
            *[mob.box[i].animate.set_opacity(1) for i in [52, 53]],
            RouteCreation(r_air_3, lag_ratio=0.1),
            frame.animate.scale(1.2), run_time=2,
        )
        self.play(
            *[FadeOut(i.copy(), scale=3, remover=True, rate_func=slow_into) for i in r_air_3],
            Transform(s_text, t_text[8]),
        )
        self.play(
            mob.rect.animate.set_stroke(opacity=1),
            mob.box[62].animate.set_opacity(1),
            frame.animate.scale(5/3).center(),
            rect.animate.shift(DR*box_unit_w*2),
            FadeOut(r_air_a, lag_ratio=0.1),
        )
        mob.m_num[53].set_fill(opacity=1)
        for i in [52, 62]:
            mob.m_num[i].set_opacity(1)
        self.play(
            mob.run_func(44, anim=True),
            RouteCreation(g_flag[3:7], lambda i: (0, i), lag_ratio=0.1),
            run_time=1,
        )
        self.play(
            rect.animate.move_to(mob.box[4]),
            mob.run_func(4, anim=True), run_time=1,
        )
        self.play(
            rect.animate.move_to(mob.box[21]),
            mob.run_func(21, anim=True), run_time=0.5,
        )
        self.play(
            FadeOut(rect.copy(), scale=3, remover=True, rate_func=slow_into),
            mob.run_func(21, anim=True), run_time=0.5,
        )
        self.play(
            rect.animate.move_to(mob.box[18]),
            mob.run_func(18, anim=True),
            RouteCreation(g_flag[7:], lambda i: (0, i), lag_ratio=0.1),
            run_time=0.5,
        )
        self.play(
            mob.run_func(9, anim=True), mob.run_func(0, anim=True),
            rect.animate.shift(box_unit_w*2*UP), run_time=0.25,
        )
        self.play(
            mob.run_func(1, anim=True), mob.run_func(2, anim=True),
            rect.animate.shift(box_unit_w*2*RIGHT), run_time=0.25,
        )
        self.play(Reveal(t_win), Uncreate(s_text))
        self.play(
            RouteCreation(rect, lambda i: (1+2*rush_into(i), 2+rush_into(i))),
            *[FadeOut(mob.box[i]) for i in bomb_list],
            Uncreate(g_flag, lag_ratio=0.1),
            Reveal(t_win, rate_func=lambda i: 1-i**4),
            run_time=1,
        )
        self.wait()


class Scene_3(Scene):
    def construct(self):
        def get_game_map(mob):
            mob.remove(mob.m_num)
            mob.set_bomb(random.randint(0, mob.n_cols-1), random.randint(0, mob.n_rows-1))
            return mob.m_num
        text_list = [
            '这些数字和地雷是如何生成的?',
            '数字的规律很明显, 围绕着地雷分布',
            '如果地雷集中, 则会出现更大的数字',
            '地雷的规律, 那肯定得随机分布了',
            '但也不能完全随机...',
            '至少不能还没开始就结束吧',
            '所以要保证初次点击的格子不是地雷',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN)
        t_text[1][10:12].set_color(BLUE)
        t_text[2][12:].set_color(RED)
        s_text = t_text[0].copy()
        grid1 = Game_MS(9, 9, 10, width=6).to_edge(UP)
        grid2 = Game_MS(16, 16, 40, width=6).to_edge(UP)
        box_unit_w = grid1.box[0].get_width()/0.8
        box_line_w = grid1.rect.get_stroke_width()
        rect = Square(box_unit_w, color=RED, stroke_width=box_line_w).move_to(grid1)
        h_help = Text('大小: 16x16\n数量: 40', font='思源黑体').next_to(grid1)
        h_help[10:].shift(DOWN*0.2)
        h_help2 = Text('大小: 9x9\n数量: 10', font='思源黑体').next_to(grid1)
        h_help2[8:].shift(DOWN*0.2)
        over = Ftext('Game over', color=BLUE).next_to(grid1)
        num_tmp = grid1.set_bomb_custom([30, 32, 42, 52, 62, 22, 13, 19, 10, 3])
        self.add(grid1.rect, num_tmp)
        self.play(
            TransformMatchingShapes(num_tmp, get_game_map(grid2)),
            Reveal(h_help, LEFT), Write(s_text), run_time=2,
        )
        for i in range(2):
            num_tmp.remove(*num_tmp)
            num_tmp = grid2.m_num
            self.play(TransformMatchingShapes(num_tmp, get_game_map(grid2)), run_time=2)
        num_tmp.remove(*num_tmp)
        num_tmp = grid2.m_num
        grid1.set_bomb_custom([9, 13, 22, 24, 38, 45, 47, 54, 71, 74])
        self.play(
            ReplacementTransform(h_help, h_help2, run_time=2),
            TransformMatchingShapes(num_tmp, grid1.m_num, run_time=2),
            Transform(s_text, t_text[1], run_time=1),
        )
        b_rect = rect.copy().set_color(BLUE).set_opacity(0.25).scale(3)
        b_rect = VGroup(*[b_rect.copy().move_to(grid1.box[i]) for i in [9, 45, 54, 71, 74, 13, 22, 24, 38, 47]])
        o_rect = VGroup(*[rect.copy().move_to(grid1.box[i]) for i in range(81) if 1 < grid1.v_map[i] % 9])
        self.add(b_rect, s_text)
        self.play(
            ShowCreation(b_rect, lag_ratio=0.2, run_time=2),
            Reveal(h_help2, LEFT, rate_func=lambda i: smooth(1-i)),
        )
        self.play(
            *[b_rect[i].animate.set_width(box_unit_w*2, True).shift(box_unit_w*0.5*RIGHT) for i in [0, 1, 2]],
            b_rect[3].animate.set_width(box_unit_w*2, True).shift(box_unit_w*0.5*LEFT),
            b_rect[4].animate.set_height(box_unit_w*2, True).shift(box_unit_w*0.5*UP),
        )
        self.play(
            Transform(s_text, t_text[2]),
            RouteCreation(o_rect, lag_ratio=0.15, run_time=3),
        )
        self.wait()
        self.play(
            *[i.animate.set_width(box_unit_w*3, True).shift(box_unit_w*0.5*LEFT).scale(1/3).set_opacity(0.5).set_fill(opacity=0) for i in b_rect[:3]],
            b_rect[3].animate.set_width(box_unit_w*3, True).shift(box_unit_w*0.5*RIGHT).scale(1/3).set_opacity(0.5).set_fill(opacity=0),
            b_rect[4].animate.set_height(box_unit_w*3, True).shift(box_unit_w*0.5*DOWN).scale(1/3).set_opacity(0.5).set_fill(opacity=0),
            *[i.animate.scale(1/3).set_opacity(0.5).set_fill(opacity=0) for i in b_rect[5:]],
            FadeOut(o_rect, lag_ratio=0.1),
            Transform(s_text, t_text[3]),
        )
        num_tmp.remove(*num_tmp)
        num_tmp = grid1.m_num
        self.play(
            FadeOut(b_rect ,lag_ratio=0.1),
            TransformMatchingShapes(num_tmp, get_game_map(grid1)),
        )
        for i in range(2):
            num_tmp.remove(*num_tmp)
            num_tmp = grid1.m_num
            self.play(TransformMatchingShapes(num_tmp, get_game_map(grid1)))
        num_tmp.remove(*num_tmp)
        num_tmp = grid1.m_num
        grid1.set_bomb_custom([4, 36, 39, 43, 46, 50, 53, 54, 66, 72])
        self.play(TransformMatchingShapes(num_tmp, grid1.m_num))
        self.play(
            Transform(s_text, t_text[4]),
            FadeIn(grid1.box, lag_ratio=0.2, run_time=2),
            RouteCreation(rect, lambda i: (1, 1-i)),
        )
        self.play(
            rect.animate.shift(box_unit_w*LEFT),
            grid1.run_func(39, anim=True), run_time=1,
        )
        grid1.v_map[39] = 10
        self.play(
            *[FadeOut(grid1.box[i]) for i in range(81) if grid1.v_map[i] == 9],
            Transform(s_text, t_text[5]),
            Reveal(over),
        )
        grid1.m_num.remove(*[grid1.m_num[i] for i in range(81) if grid1.v_map[i] < 9])
        [grid1.box[i].set_opacity(0) for i in range(81) if grid1.v_map[i] > 8]
        self.play(
            RouteCreation(rect, lambda i: (1-rush_into(i), 2-slow_into(i))),
            *[i.animate.set_opacity(0.5) for i in grid1.box],
            over.animate.set_color(GREY),
        )
        self.add(grid1.m_num[2], grid1.box)
        self.play(
            Uncreate(over, lag_ratio=0.2),
            Transform(s_text, t_text[6]),
            grid1.m_num[2].animate.shift(box_unit_w*UL*2),
        )
        self.play(grid1.box.animate.set_opacity(1))
        num_tmp = grid1.m_num
        grid1.remove(grid1.m_num)
        grid1.set_bomb_custom([4, 19, 36, 43, 46, 50, 53, 54, 66, 72])
        self.add(grid1.m_num, grid1.box)
        self.play(grid1.run_func(39, anim=True), run_time=1)
        self.wait()
        self.play(FadeOut(s_text, DOWN))


class Scene_4(Scene):
    def construct(self):
        def map_to_str():
            out = ''
            for i, m in enumerate(zip(mob.v_mask, mob.v_map)):
                out += ' '+('*' if m[0] else ' 12345678@'[m[1]])
                if (i+1) % mob.n_cols == 0:
                    out += '\n'
            return out.split('\n')
        text_list = [
            '关于某些功能实现, 补充一点设定',
            '由于鼠标事件处理比较麻烦...',
            '交互方式, 由键盘代替鼠标操作',
            '通过ADSW移动"光标", 选择位置',
            '空格键对应左击操作, Esc键退出游戏',
            '输出方式, 由字符组成可视化界面',
            '包含了方块 空地 数字 光标 雷元素',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        s_text = t_text[0].copy()
        mob = Game_MS(9, 9, 10, width=6).to_edge(UP)
        mob.set_bomb_custom([4, 19, 36, 43, 46, 50, 53, 54, 66, 72])

        box_unit_w = mob.box[0].get_width()/0.8
        box_line_w = mob.rect.get_stroke_width()
        rect = Square(box_unit_w, color=RED, stroke_width=box_line_w).move_to(mob.box[39])
        mouse = VGroup(
            Polygon(LEFT, UP, ORIGIN),
            Polygon(RIGHT, UP, ORIGIN),
            Polygon(LEFT, DOWN*2, RIGHT),
        ).set_fill(opacity=1, color=WHITE).set_stroke(color=BLACK)
        mouse[0].data['points'][1][:] = UL
        mouse[1].data['points'][1][:] = UR
        mouse[2].data['points'][1][:] = DL+DOWN
        mouse[2].data['points'][4][:] = DR+DOWN
        mouse.next_to(mob)
        keyboard = VGroup(*[Square(0.9) for i in range(5)], Rectangle(2.9, 0.9))
        keyboard.set_fill(BLACK, 1)
        keyboard[0].add(Ftext('Esc').set_width(0.75)).shift(UL)
        keyboard[1].add(Ftext('W')).shift(UP)
        keyboard[2].add(Ftext('A')).shift(LEFT)
        keyboard[3].add(Ftext('S'))
        keyboard[4].add(Ftext('D')).shift(RIGHT)
        keyboard[5].add(Ftext('Space')).next_to(keyboard[3], DOWN, buff=0.1)
        keyboard.next_to(mouse)
        frame = self.camera.frame
        cli = CommandLine()
        con = Console(lines=10).next_to(mob)
        m_text = cli.m_text
        for i in open(os.getcwd()+'\\code_6.txt', 'r').readlines():
            cli.add_text(i)
        cli.next_to(mob, DOWN).set_color_c_style()
        self.add(mob, rect)
        self.play(
            *[FadeIn(mob.box[i]) for i in mob.run_func(39)],
            Write(s_text, run_time=1),
        )
        self.add(rect)
        self.play(ShowCreation(mouse))
        self.play(
            ApplyMethod(mouse.shift, RIGHT*0.2, rate_func=there_and_back),
            rect.animate.shift(box_unit_w*RIGHT),
        )
        self.add(cli.m_text, s_text)
        self.play(
            *[Write(i, run_time=min(3, len(i)*0.05)) for i in cli.m_text],
            ApplyMethod(frame.shift, BOTTOM, run_time=1.5),
            Transform(s_text, t_text[1]),
        )
        self.wait()
        self.play(
            *[Uncreate(i, lag_ratio=0.3, run_time=2) for i in cli.m_text],
            *[RouteCreation(i, lambda i: (0.1*i, 1.1*i), close_path=True, lag_ratio=0.1, run_time=2) for i in keyboard],
            ApplyMethod(frame.move_to, mouse, run_time=1.5),
            Transform(s_text, t_text[2]),
        )
        self.play(FadeOut(mouse, LEFT), keyboard.animate.next_to(mob), frame.animate.center())
        self.wait()
        self.play(
            *[ApplyMethod(i.scale, 1.1, rate_func=there_and_back) for i in keyboard[1:5]],
            Transform(s_text, t_text[3]),
        )
        self.play(
            ApplyMethod(keyboard[1].set_color, RED, rate_func=there_and_back),
            rect.animate.shift(box_unit_w*UP),
        )
        self.play(
            ApplyMethod(keyboard[2].set_color, RED, rate_func=there_and_back),
            rect.animate.shift(box_unit_w*LEFT),
        )
        self.play(
            ApplyMethod(keyboard[3].set_color, RED, rate_func=there_and_back),
            rect.animate.shift(box_unit_w*DOWN),
        )
        self.play(
            ApplyMethod(keyboard[5].set_color, RED, rate_func=there_and_back),
            Transform(s_text, t_text[4]),
        )
        self.play(
            *[FadeOut(mob.box[i]) for i in mob.dig_buf],
        )
        self.wait()
        self.play(
            RouteCreation(keyboard, lambda i: (i, 1)),
            frame.animate.shift(3*RIGHT).scale(1.2),
            FadeIn(con, lag_rato=0.1),
        )
        con_cursor = con.new_word('>', 6, 4)[0]
        mob_rect = rect.copy()
        mob_show = VGroup(
            *[mob.box[i].copy() if m else mob.m_num[i].copy() for i, m in enumerate(mob.v_mask)]
        ).add(mob_rect)
        self.play(
            mob_show.animate.set_height(con.unit_h*8.9).move_to(con.background.get_corner(UL)+DR*0.05, UL),
            con.cursor.animate.shift(9*con.unit_h*DOWN),
            Transform(s_text, t_text[5]),
        )
        mob_show.remove(mob_rect)
        for i in map_to_str():
            con.add_text(i[1::2])
        self.remove(con.m_text)
        self.play(
            *[ReplacementTransform(i, j) for i, j in zip(mob_show, [i for i in con.m_text.get_family() if len(i) == 0])],
            mob_rect.animate.set_width(con.unit_w, True).move_to(con.get_grid_place(3, 4)), run_time=2,
        )
        self.play(
            *[AnimationGroup(*[ApplyMethod(m.shift, (i+1)*con.unit_w*RIGHT) for i, m in enumerate(l)]) for l in con.m_text],
            ReplacementTransform(mob_rect, con_cursor), run_time=1,
        )
        for i in [RIGHT, RIGHT, DOWN, DOWN]:
            con_cursor.shift(np.array([con.unit_w*2, con.unit_h, 0])*i)
            self.play(rect.animate.shift(box_unit_w*i), run_time=0.5)
        con.m_text[6][5].become(con.new_word('1', 11, 6)[0])
        self.play(Transform(s_text, t_text[6]), mob.run_func(59, True), run_time=1)
        con_cursor.shift(con.unit_h*DOWN)
        self.play(rect.animate.shift(box_unit_w*DOWN), run_time=0.5)
        tmp_anim = mob.run_func(68, True)
        self.remove(con.m_text)
        for i, m in enumerate(map_to_str()):
            con.new_text(m, i)
        self.add(con.m_text)
        self.play(tmp_anim, run_time=1)
        for i in range(2):
            con_cursor.shift(con.unit_h*UP)
            self.play(rect.animate.shift(box_unit_w*UP), run_time=0.5)
        for i, m in enumerate(mob.v_map):
            if m == 9:
                mob.v_mask[i] = 0
        self.remove(con.m_text)
        for i, m in enumerate(map_to_str()):
            con.new_text(m, i)
        self.add(con.m_text)
        self.play(*[FadeOut(m) for i, m in enumerate(mob.box) if mob.v_map[i] == 9])
        mob.sort_func(mob.m_num, 2)
        self.play(
            *[FadeOut(m, run_time=min(1, 0.1+random.random())) for i, m in enumerate(mob.box) if mob.v_mask[i] == 1],
            RouteCreation(mob.rect, lambda i: (1-0.2*rush_into(i), 2-1.2*rush_into(i)), run_time=1.5),
            RouteCreation(rect, lambda i: (1+2*rush_into(i), 2+rush_into(i)), run_time=1.5),
            *[Uncreate(i, lag_ratio=0.05, run_time=1.5) for i in con],
            Uncreate(mob.m_num, lag_ratio=0.1, run_time=1.5),
            Uncreate(s_text, lag_ratio=0.1, run_time=1.5),
            Uncreate(con_cursor),
        )
        self.wait()


class Scene_5(Scene):
    def construct(self):
        text_list = [
            '整理一下制作思路...',
            '跟前几期代码的架构差不多',
            '还有很多细节就不列举了',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        s_text = t_text[0].copy()
        t1 = VGroup(*[Text(i, font='思源黑体') for i in ['开始', '游戏循环', '结束']])
        t21 = VGroup(*[Text(i, font='思源黑体') for i in ['变量初值', '内存分配', '随机数初始化']])
        t22 = VGroup(*[Text(i, font='思源黑体') for i in ['输入', '数据处理', '输出']])
        t23 = VGroup(*[Text(i, font='思源黑体') for i in ['打印成功or失败文本', '阻塞']])
        t31 = VGroup(*[Text(i, font='思源黑体') for i in ['监听键盘输入', '清屏&打印界面']])
        t32 = VGroup(*[Text(i, font='思源黑体') for i in ['输入处理', '生成地雷', '挖掘']])
        t_all = VGroup(t1, t21, t22, t23, t31, t32)
        unit_h = (max([i.get_height() for i in t_all])+0.2)*DOWN
        frame = self.camera.frame
        for mob in t_all:
            for i, m in enumerate(mob):
                m.add_to_back(
                    Circle(color=BLUE, fill_opacity=1).set_width(0.1).next_to(m, LEFT)
                ).move_to(i*unit_h, LEFT)
        t_all[1:].scale(0.8)
        t_all[4:].scale(0.8)
        t_all.shift(-t1.get_center())
        for i in t_all:
            i.align_to(t_all[0], UP)
        t_map = t_all.copy()
        t_map[0][0].shift(-3*unit_h)
        t_map[0][2].shift(3*unit_h)
        t_map[2][0].shift(-unit_h)
        t_map[2][2].shift(unit_h)
        t_map[1].next_to(t_map[0][0], buff=1)
        t_map[2].next_to(t_map[0][1], buff=1)
        t_map[3].next_to(t_map[0][2], buff=1)
        t_map[4][0].next_to(t_map[2][0], buff=0.5)
        t_map[4][1].next_to(t_map[2][2], buff=0.5)
        t_map[5].next_to(t_map[2][1], buff=0.5)
        t_map.center()
        line = VGroup()
        def line_add(point, mob, smooth_line=True):
            for i in mob:
                tmp = Line(point, i[0].get_center())
                if 0.1 < abs(tmp.get_start()[1]-tmp.get_end()[1]) and smooth_line:
                    s = tmp.get_start()
                    e = tmp.get_end()
                    m = mid(s, e)
                    m1 = np.array([m[0], s[1], 0])
                    m2 = np.array([m[0], e[1], 0])
                    tmp.set_points([s, m1, m, m, m2, e])
                line.add(tmp)
        line_add(t_map[0][1].get_left()+LEFT, t_map[0])
        line_add(t_map[0][0].get_right()+0.05*RIGHT, t_map[1])
        line_add(t_map[0][1].get_right()+0.05*RIGHT, t_map[2])
        line_add(t_map[0][2].get_right()+0.05*RIGHT, t_map[3], False)
        line_add(t_map[2][0].get_right()+0.05*RIGHT, [t_map[4][0]], False)
        line_add(t_map[2][2].get_right()+0.05*RIGHT, [t_map[4][1]], False)
        line_add(t_map[2][1].get_right()+0.05*RIGHT, t_map[5])
        line.set_stroke(opacity=0.5)

        self.play(Write(s_text, run_time=1))
        self.play(*[Write(i, run_time=2) for i in t1])
        self.play(
            Transform(s_text, t_text[1], rate_func=squish_rate_func(smooth, 0.5, 1)),
            *[Write(i) for i in t21.shift(unit_h)],
            *[Write(i) for i in t22.shift(4.4*unit_h)],
            *[Write(i) for i in t23.shift(7.8*unit_h)],
            t1[1].animate.shift(2.4*unit_h),
            t1[2].animate.shift(4.8*unit_h),
            frame.animate.set_y(t22[1].get_y()).scale(1.2),
            run_time=2,
        )
        self.play(
            *[Write(i) for i in t32.shift(6.64*unit_h)],
            Write(t31[1].shift(8.72*unit_h)),
            Write(t31[0].shift(5.2*unit_h)),
            t22[1].animate.shift(0.64*unit_h),
            t22[2].animate.shift(2.56*unit_h),
            t1[2].animate.shift(3.2*unit_h),
            t23.animate.shift(3.2*unit_h),
            frame.animate.set_y(t32[1].get_y()).scale(1.4),
            run_time=2,
        )
        self.wait()
        Group(frame, t_all).set_y(0)
        self.play(
            Transform(s_text, t_text[2], rate_func=squish_rate_func(smooth, 0.5, 1)),
            Transform(t_all, t_map, rate_func=squish_rate_func(smooth, 0, 0.8)),
            frame.animate.scale(0.8).set_y(-1),
            ShowCreation(line, lag_ratio=0.1),
            run_time=3,
        )
        self.wait()


class Scene_6(Scene):
    def construct(self):
        text_list = [
            '首先, 从头文件开始',
            '简单地过一遍...',
            '都是常见的头文件',
            '关于conio.h...',
            '(不重要, 此处可战术暂停)',
            '接下来就是变量了, 这个得细说',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        frame = self.camera.frame
        d_head = cli.m_text[:4].get_height()*UP
        cli.shift(d_head)
        frame.shift(d_head)
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        ).align_to(cli.get_bottom(), UP)
        t1_h1 = cli.m_text[0][10:17].copy().scale(3).set_x(0).to_edge(UP).set_color(ORANGE)
        t1_h2 = cli.m_text[1][10:18].copy().scale(3).set_x(0).next_to(t1_h1, DOWN).set_color(ORANGE)
        t1_t1 = Ftext('(console input/output)').move_to(t1_h1.get_center()+LEFT*2, LEFT)
        t1_t2 = Ftext('(standard library)').move_to(t1_h2.get_center()+LEFT*2, LEFT)
        for i in [1, 2, 3, 9, 15]:
            t1_t1[i].set_color(ORANGE)
        for i in [1, 2, 8, 10, 11, 12]:
            t1_t2[i].set_color(ORANGE)
        t1_t3 = Text('控制台输入输出 包含控制台io函数', font='思源黑体')
        t1_t4 = Text('标准库函数 包含一些常见的系统函数', font='思源黑体')
        t1_t5 = VGroup(*[Ftext(i, color=GREY_B).scale(0.6) for i in ['getch()', 'cprintf()', 'cputs()']])
        t1_t6 = Ftext('rand() srand() calloc() abs() system()', color=GREY_B).scale(0.6)
        t1_t3[8:].scale(0.8, about_edge=DL).set_color(GREY_A)
        t1_t4[6:].scale(0.8, about_edge=DL).set_color(GREY_A)
        t2_text_list = [
            'conio.h并非标准头文件, 其他平台可能没有conio.h',
            '和stdio.h相似, 不同的是getch()有"无缓冲"的特性',
            '其实也可以用stdio.h, 都有功能相似的函数替代',
            '但getchar()是"行缓冲", 每次输入需要额外回车',
        ]
        t2_text = VGroup(*[Text(i, font='思源黑体').scale(0.8) for i in t2_text_list])
        t2_func = VGroup(*[Ftext(i, color=GREEN).scale(0.6) for i in ['getchar()', 'printf()', 'puts()', '<stdio.h>']])
        unit_h = (max([i.get_height() for i in t2_text])+0.2)*DOWN
        for i, m in enumerate(t2_text):
            m.add_to_back(
                Circle(color=BLUE, fill_opacity=1).set_width(0.1).next_to(m, LEFT)
            ).move_to(i*unit_h, LEFT)
        b_rect = VGroup(
            Rectangle(FRAME_WIDTH, cli.unit_h*3, color=GREEN, fill_opacity=0.6, stroke_width=0),
            Rectangle(FRAME_WIDTH, cli.unit_h*11, color=RED, fill_opacity=0.6, stroke_width=0),
            Rectangle(FRAME_WIDTH, cli.unit_h*10, color=BLUE, fill_opacity=0.6, stroke_width=0),
        ).arrange(DOWN, buff=0).shift(cli.unit_h*4.3*UP)
        b_text = VGroup(*[Text(i, font='思源黑体') for i in ['头文件&变量定义部分, 最先讲解', '函数运用大量技巧, 比较抽象, 最后讲解', 'main()开始运用各种技巧, 其次讲解']])
        for i, j, k in zip(b_text, b_rect, [6, 2, 6]):
            i.to_edge(LEFT).match_y(j)[k:].scale(0.8, about_edge=DL).shift(0.5*RIGHT)
        
        self.play(*[Write(i) for i in cli.m_text], run_time=1)
        self.play(
            AnimationGroup(*[Reveal(i, UP) for i in b_rect], lag_ratio=0.2, run_time=2),
            AnimationGroup(*[Write(i) for i in b_text], lag_ratio=0.2, run_time=2),
        )
        self.wait(2)
        self.play(
            AnimationGroup(*[Reveal(i, DOWN, remover=True, rate_func=lambda i: 1-i**2) for i in b_rect], lag_ratio=0.2),
            AnimationGroup(*[Uncreate(i, lag_ratio=0.2) for i in b_text], lag_ratio=0.1),
        )
        self.add(bg_rect, s_text)
        turn_animation_into_updater(Write(s_text))
        self.play(bg_rect.animate.center(), rate_func=slow_into)
        self.wait()
        self.play(
            TransformFromCopy(cli.m_text[0][10:17], t1_h1),
            TransformFromCopy(cli.m_text[1][10:18], t1_h2),
            Transform(s_text, t_text[1]),
        )
        self.wait(0.5)
        self.play(
            t1_h1.animate.next_to(t1_t1, LEFT),
            t1_h2.animate.next_to(t1_t2, LEFT),
            TransformFromCopy(t1_h1, t1_t1),
            TransformFromCopy(t1_h2, t1_t2),
        )
        t1_t4.next_to(t1_h2, DOWN, aligned_edge=LEFT)
        t1_t6.next_to(t1_t4, DOWN, aligned_edge=LEFT)
        self.play(
            RouteCreation(Underline(t1_h2, stroke_width=10, color=RED), lambda i: (rush_from(i), max(0, 5*i-4)), remover=True),
            ApplyMethod(Group(t1_h1, t1_t1).align_to, t1_h2, LEFT, rate_func=slow_into, remover=True),
            FadeIn(t1_t6, RIGHT*0.5, lag_ratio=0.5, rate_func=slow_into),
            Write(t1_t4), run_time=1.5,
        )
        t1_t3.next_to(t1_h1, DOWN, aligned_edge=LEFT)
        t1_t5.arrange(RIGHT, aligned_edge=DOWN).next_to(t1_t3, DOWN, aligned_edge=LEFT)
        t2_func.arrange(RIGHT).next_to(t1_t5, DOWN, aligned_edge=LEFT)
        t1_all2 = VGroup(t1_h2, t1_t2, t1_t4, t1_t6)
        t2_text.next_to(t2_func, DOWN, aligned_edge=LEFT)
        self.add(t1_h1, t1_t1).wait()
        self.play(
            t1_all2.animate.next_to(t1_t5, DOWN, aligned_edge=LEFT),
            Transform(s_text, t_text[2]),
            frame.animate.center(),
        )
        self.play(
            RouteCreation(Underline(t1_h1, stroke_width=10, color=RED), lambda i: (rush_from(i), max(0, 5*i-4)), remover=True),
            FadeIn(t1_t5, RIGHT*0.5, lag_ratio=0.5, rate_func=slow_into),
            Write(t1_t3), run_time=1.5,
        )
        self.wait()
        self.play(
            Transform(s_text, t_text[3]),
            FadeOut(t1_all2, DOWN*0.1, lag_ratio=0.05),
        )
        self.play(
            *[Write(i, run_time=1) for i in t2_text],
        )
        self.play(
            TransformFromCopy(t1_t5, t2_func[:3]),
            Reveal(t2_func[3]),
        )
        self.wait()
        out = []
        shift = [UP, DOWN, LEFT, RIGHT]
        for j in [t1_h1, t1_t1, t1_t3, *t1_t5, *t2_func, *t2_text]:
            for i in j:
                out.append(FadeOut(i, random.random()*shift[random.randint(0, 3)]))
        t2_rect = Rectangle(
            FRAME_WIDTH, cli.unit_h,
            fill_opacity=0.5, fill_color=BLUE, stroke_width=0,
        ).match_y(cli.m_text[2])
        self.mobjects.insert(0, t2_rect)
        self.play(
            Transform(s_text, t_text[5], rate_func=squish_rate_func(smooth, 0, 0.8)),
            Reveal(t2_rect, LEFT, rate_func=squish_rate_func(smooth, 0.5, 1)),
            frame.animate.shift(d_head),
            *out, run_time=2,
        )


class Scene_7(Scene):
    def construct(self):
        text_list = [
            '接下来就是变量了, 这个得细说',
            '从第一个开始, WHB, 对应宽 长 雷数',
            '作为游戏最基本的属性, 作用很明显',
            '可随意更改, 只要不是太小或者太大',
            'S和s, 总数, 都是记录格子数量',
            'S记录总数量, s记录未挖掘数量',
            'pci, 对应坐标, 字符, 整数',
            'p记录光标位置, c和i临时变量',
            'm和M, 对应地图和掩码',
            'M记录格子是否打开, m记录格子的元素',
            'f, 函数指针, 非常好玩的东西',
            '然后, 就到主函数main()了',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        frame = self.camera.frame
        d_head = cli.m_text[:4].get_height()*UP
        cli.shift(d_head)
        frame.shift(d_head)
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        )
        c_rect = Rectangle(
            FRAME_WIDTH, cli.unit_h,
            fill_opacity=0.5, fill_color=BLUE, stroke_width=0,
        ).match_y(cli.m_text[2])
        game = Game_MS(9, 9, 10).to_edge(UP, buff=0.2)
        game.set_bomb(0, 0, 0)
        game.m_num.submobjects.sort(key=lambda i: random.random())
        game.m_num.assemble_family()
        game_tmp2 = Game_MS(16, 16, 40, 6).align_to(game, UP)
        game_tmp2.set_bomb(0, 0, 0)
        game_tmp3 = Game_MS(30, 16, 99, 10).move_to(game_tmp2.get_left(), LEFT)
        game_tmp3.set_bomb(0, 0, 0)
        t2_all = VGroup(
            *[Ftext(i, color=BLUE) for i in ['Width', 'Height', 'Bomb']],
        ).arrange(DOWN, aligned_edge=RIGHT, buff=1).to_edge(LEFT).match_y(game)
        t2_var = VGroup(*[i[0].set_color(ORANGE).copy().align_to(i, RIGHT) for i in t2_all])
        t2_text = VGroup(*[Text(m, font='思源黑体').next_to(t2_all[i], DOWN, aligned_edge=RIGHT, buff=0.2) for i, m in enumerate(['宽', '长', '雷数'])])
        t2_num = VGroup(*[MyNum(m).next_to(t2_all[i], RIGHT, buff=0.5) for i, m in enumerate([9, 9, 10])])
        t4_low = Text('卡死', color=GREY_B, font='思源黑体').match_width(t_text[3][11:13]).next_to(t_text[3][11:13], UP, buff=0.01).scale(0.7, about_edge=DOWN).fix_in_frame()
        t4_pow = Text('爆栈', color=GREY_B, font='思源黑体').match_width(t_text[3][15:]).next_to(t_text[3][15:], UP, buff=0.01).scale(0.7, about_edge=DOWN).fix_in_frame()
        self.add(cli.m_text, bg_rect, s_text, c_rect)
        self.play(
            c_rect.animate.move_to(cli.m_text[2][7:27]).set_width(cli.m_text[2][7:27].get_width(), True),
            RouteCreation(game.rect, lambda i: (0.2*i, 1.2*i), run_time=1.5),
            FadeIn(game.m_num, run_time=3, lag_ratio=0.05),
        )
        self.play(
            *[TransformFromCopy(cli.m_text[2][m], t2_var[i]) for i, m in enumerate([7, 14, 21])],
            Transform(s_text, t_text[1]),
        )
        self.play(
            *[ReplacementTransform(m, t2_all[i][0]) for i, m in enumerate(t2_var)],
            *[TrimCreation(i[1:], RIGHT) for i in t2_all],
            run_time=1, rate_func=rush_into,
        )
        self.play(
            *[Reveal(i, UP) for i in t2_text],
            *[Write(i) for i in t2_num],
            run_time=2,
        )
        self.play(
            NumberToValue(t2_num[2], 20),
            Transform(s_text, t_text[2]),
        )
        game.n_bomb = 20
        old_game = game.m_num
        game.set_bomb(0, 0, 0)
        self.play(TransformMatchingShapes(old_game, game.m_num), run_time=2)
        self.play(
            frame.animate.center(),
            NumberToValue(t2_num[0], 16),
            NumberToValue(t2_num[1], 16),
            NumberToValue(t2_num[2], 40),
            TransformMatchingShapes(game.rect, game_tmp2.rect),
            TransformMatchingShapes(game.m_num, game_tmp2.m_num), run_time=2,
        )
        self.play(
            Transform(s_text, t_text[3], rate_func=squish_rate_func(smooth, 0, 0.5)),
            Reveal(t4_low, DOWN, rate_func=squish_rate_func(smooth, 0.5, 1)),
            Reveal(t4_pow, DOWN, rate_func=squish_rate_func(smooth, 0.5, 1)),
            NumberToValue(t2_num[0], 30),
            NumberToValue(t2_num[1], 16),
            NumberToValue(t2_num[2], 99),
            ReplacementTransform(game_tmp2.rect, game_tmp3.rect),
            TransformMatchingShapes(game_tmp2.m_num, game_tmp3.m_num), run_time=2,
        )
        game.n_bomb = 10
        game.set_bomb(3, 0, 0)
        game.set_submobjects([game.m_num, game.box, game.rect])
        self.play(
            frame.animate.shift(d_head),
            NumberToValue(t2_num[0], 9),
            NumberToValue(t2_num[1], 9),
            NumberToValue(t2_num[2], 10),
            ReplacementTransform(game_tmp3.rect, game.rect),
            TransformMatchingShapes(game_tmp3.m_num, game.m_num), run_time=2,
        )
        
        t5_all = Ftext('Sum', color=BLUE).move_to(game).to_edge(LEFT).shift(UP)
        t5_var = t5_all[0].set_color(ORANGE).copy().align_to(t5_all, RIGHT)
        t5_num = MyNum(81).next_to(t5_all, RIGHT, buff=0.5)
        t5_text = Text('总数', font='思源黑体').next_to(t5_all, DOWN, aligned_edge=RIGHT, buff=0.2)
        t5_all2 = Ftext('sum', color=BLUE).move_to(game).to_edge(LEFT).shift(DOWN)
        t5_var2 = t5_all2[0].set_color(ORANGE).copy().align_to(t5_all2, RIGHT)
        t5_num2 = MyNum(30).next_to(t5_all2, RIGHT, buff=0.5)
        t5_text2 = t5_text.copy().next_to(t5_all2, DOWN, aligned_edge=RIGHT, buff=0.2)
        t5_grid = MyGrid(9, 9, 5, build=2).move_to(game)
        t5_grid.box.set_fill(opacity=0.5).set_color(RED).set_stroke(width=1)
        t5_grid_c = t5_grid.box.copy().set_opacity(0)
        c_rect2 = c_rect.copy()
        box_unit_w = game.box[0].get_width()/0.8
        cursor = Square(box_unit_w, color=RED, stroke_width=game.rect.get_stroke_width()).move_to(game.box[40])
        self.play(
            c_rect2.animate.move_to(cli.m_text[2][29]).set_width(cli.unit_w, True),
            c_rect.animate.move_to(cli.m_text[2][4]).set_width(cli.unit_w, True),
            *[Reveal(i, DOWN, rate_func=lambda i: 1-i**4) for i in t2_all],
            *[Reveal(i, UP, rate_func=lambda i: 1-i**4) for i in t2_text],
            *[Uncreate(i) for i in t2_num],
            Reveal(t4_low, DOWN, rate_func=lambda i:1-i**4),
            Reveal(t4_pow, DOWN, rate_func=lambda i:1-i**4),
            FadeIn(game.box, lag_ratio=0.1),
            run_time=2, remover=True,
        )
        self.add(c_rect, c_rect2, game.box)
        self.play(
            Transform(s_text, t_text[4]),
            TransformFromCopy(cli.m_text[2][4], t5_var),
            TransformFromCopy(cli.m_text[2][29], t5_var2),
        )
        self.play(
            FadeOut(VGroup(*[game.box[i] for i in game.dig_buf]), lag_ratio=0.1),
            ReplacementTransform(t5_var, t5_all[0]),
            TrimCreation(t5_all[1:], RIGHT),
            Reveal(t5_text, UP),
            Write(t5_num),
            ReplacementTransform(t5_var2, t5_all2[0]),
            TrimCreation(t5_all2[1:], RIGHT),
            Reveal(t5_text2, UP),
            Write(t5_num2),
            run_time=2,
        )
        self.play(*[RouteCreation(m, run_time=(i%9+i//9)*0.1125+0.2) for i, m in enumerate(t5_grid.box)])
        self.play(
            Transform(s_text, t_text[5]),
            t5_num.animate.set_color(RED),
            t5_grid_c.animate.scale(0.2).next_to(t5_num, buff=0.4).set_fill(opacity=0.5).set_stroke(opacity=1),
            *[Uncreate(m) if game.v_mask[i] == 0 else m.animate.set_color(BLUE) for i, m in enumerate(t5_grid.box)],
            run_time=2,
        )
        self.play(
            t5_num2.animate.set_color(BLUE),
            t5_grid.box.animate.scale(0.2).next_to(t5_num2, buff=0.4),
            run_time=2,
        )
        self.play(
            game.run_func(35, anim=True),
            NumberToValue(t5_num2, 24),
            *[m.animate.set_opacity(0) for i, m in enumerate(t5_grid.box) if i in game.dig_buf],
            run_time=2,
        )
        shift = [UP, DOWN, LEFT, RIGHT]
        self.play(
            FadeOut(c_rect),
            Uncreate(t5_num),
            Uncreate(t5_num2),
            ShowCreation(cursor),
            Reveal(t5_text, UP, rate_func=lambda i: 1-i**4),
            Reveal(t5_text2, UP, rate_func=lambda i: 1-i**4),
            Reveal(t5_all, DOWN, rate_func=lambda i: 1-i**4),
            Reveal(t5_all2, DOWN, rate_func=lambda i: 1-i**4),
            c_rect2.animate.move_to(cli.m_text[2][32:47]).set_width(cli.unit_w*15, True),
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t5_grid_c],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t5_grid.box],
            run_time=2, remover=True,
        )
        self.add(c_rect2, cursor)
        
        t7_all = VGroup(
            *[Ftext(i, color=BLUE) for i in ['place', 'char', 'int']],
        ).arrange(DOWN, aligned_edge=RIGHT, buff=1).to_edge(LEFT).match_y(game)
        t7_var = VGroup(*[i[0].set_color(ORANGE).copy().align_to(i, RIGHT) for i in t7_all])
        t7_text = VGroup(*[Text(m, font='思源黑体').next_to(t7_all[i], DOWN, aligned_edge=RIGHT, buff=0.2) for i, m in enumerate(['坐标', '字符', '整数'])])
        t7_num = MyNum(40).next_to(t7_all[0], RIGHT, buff=0.5).set_color(ORANGE)
        t7_axis = VGroup(
            *[MyNum(i).move_to(game.box[i*9+8].get_center()+RIGHT*box_unit_w).scale(0.8) for i in range(9)],
            *[MyNum(i).next_to(game.box[i+72], DOWN, buff=0.1).scale(0.8) for i in range(9)],
        )
        t7_rect = VGroup(
            Rectangle(box_unit_w*9, box_unit_w, color=BLUE, fill_opacity=0.5, stroke_width=0).move_to(game),
            Rectangle(box_unit_w, box_unit_w*9, color=RED, fill_opacity=0.5, stroke_width=0).move_to(game),
        )
        t7_axis[4].set_color(BLUE)
        t7_axis[13].set_color(RED)
        t7_equa = VGroup(
            MyNum(40), Ftext('='), MyNum(4), Ftext('+'), MyNum(4), Ftext('*'), Ftext('9'),
        ).arrange(buff=0.1, center=False).next_to(game, buff=1)
        self.play(
            *[TransformFromCopy(cli.m_text[2][m], t7_var[i]) for i, m in enumerate([32, 39, 46])],
            Transform(s_text, t_text[6]),
        )
        self.play(
            *[ReplacementTransform(m, t7_all[i][0]) for i, m in enumerate(t7_var)],
            *[TrimCreation(i[1:], RIGHT) for i in t7_all],
            run_time=1, rate_func=rush_into,
        )
        self.play(*[Reveal(i, UP) for i in t7_text], Write(t7_num, run_time=2))
        self.play(
            Write(t7_axis, lag_ratio=0.1),
            frame.animate.shift(DOWN*0.5),
            Reveal(t7_rect[0], DOWN),
            Reveal(t7_rect[1], LEFT),
            run_time=2,
        )
        self.play(
            Transform(s_text, t_text[7]),
            *[Write(i) for i in [t7_equa[1], t7_equa[3], t7_equa[5]]],
            TransformFromCopy(t7_num, t7_equa[0].set_color(ORANGE)),
            TransformFromCopy(t7_axis[4], t7_equa[2].set_color(BLUE)),
            TransformFromCopy(t7_axis[13], t7_equa[4].set_color(RED)),
            TransformFromCopy(cli.m_text[2][7:12], t7_equa[6].set_color(GREEN)),
            ApplyMethod(t7_num.scale, 1.5, rate_func=there_and_back),
            ApplyMethod(t7_axis[4].scale, 1.5, rate_func=there_and_back),
            ApplyMethod(t7_axis[13].scale, 1.5, rate_func=there_and_back),
            ApplyMethod(cli.m_text[2][7:12].scale, 1.5, rate_func=there_and_back),
            run_time=2,
        )
        self.play(
            t7_axis[4].animate.set_color(WHITE),
            t7_axis[5].animate.set_color(BLUE),
            NumberToValue(t7_num, 49),
            NumberToValue(t7_equa[0], 49),
            t7_equa[4].animate.re_num(5),
            cursor.animate.shift(DOWN*box_unit_w),
            t7_rect[0].animate.shift(DOWN*box_unit_w),
            run_time=1.5,
        )
        self.play(
            t7_axis[13].animate.set_color(WHITE),
            t7_axis[14].animate.set_color(RED),
            NumberToValue(t7_num, 50),
            NumberToValue(t7_equa[0], 50),
            t7_equa[2].animate.re_num(5),
            cursor.animate.shift(RIGHT*box_unit_w),
            t7_rect[1].animate.shift(RIGHT*box_unit_w),
            run_time=1.5,
        )
        self.play(
            t7_axis[1].animate.set_color(BLUE),
            t7_axis[10].animate.set_color(RED),
            t7_axis[5].animate.set_color(WHITE),
            t7_axis[14].animate.set_color(WHITE),
            NumberToValue(t7_num, 10),
            NumberToValue(t7_equa[0], 10),
            t7_equa[2].animate.re_num(1),
            t7_equa[4].animate.re_num(1),
            cursor.animate.shift(box_unit_w*4*UL),
            t7_rect[0].animate.shift(box_unit_w*4*UP),
            t7_rect[1].animate.shift(box_unit_w*4*LEFT),
            run_time=1.5,
        )
        self.wait(0.5)
        self.play(
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t7_axis],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t7_equa],
            c_rect2.animate.move_to(cli.m_text[2][49:55]).set_width(cli.unit_w*6, True),
            *[Reveal(i, DOWN, rate_func=lambda i: 1-i**4) for i in t7_all],
            *[Reveal(i, UP, rate_func=lambda i: 1-i**4) for i in t7_text],
            Reveal(t7_rect[1], RIGHT, rate_func=lambda i: 1-i**4),
            Reveal(t7_rect[0], UP, rate_func=lambda i: 1-i**4),
            frame.animate.shift(UP*0.5),
            Uncreate(t7_num),
            Uncreate(cursor),
            run_time=2, remover=True,
        )
        self.add(c_rect2)
        
        t9_all2 = Ftext('Mask', color=BLUE).move_to(game).to_edge(LEFT).shift(DOWN)
        t9_all = Ftext('map', color=BLUE).move_to(game).align_to(t9_all2, RIGHT).shift(UP)
        t9_var = t9_all[0].set_color(ORANGE).copy().align_to(t9_all, RIGHT)
        t9_var2 = t9_all2[0].set_color(ORANGE).copy().align_to(t9_all2, RIGHT)
        t9_text = Text('地图', font='思源黑体').next_to(t9_all, DOWN, aligned_edge=RIGHT, buff=0.2)
        t9_text2 = Text('掩码', font='思源黑体').next_to(t9_all2, DOWN, aligned_edge=RIGHT, buff=0.2)
        t9_grid = MyGrid(9, 9, 5).move_to(game.get_center()+RIGHT, LEFT)
        t9_rect = Square(t9_all2.get_width()+0.1, color=RED).move_to(Group(t9_all2, t9_text2))
        t9_grid.sort_func(t9_grid.line, 1)
        t9_grid_n = t9_grid.add_number(size=0.9, init=lambda i: 0)
        self.play(
            Transform(s_text, t_text[8]),
            TransformFromCopy(cli.m_text[2][50], t9_var),
            TransformFromCopy(cli.m_text[2][54], t9_var2),
        )
        self.play(
            ReplacementTransform(t9_var, t9_all[0]),
            TrimCreation(t9_all[1:], RIGHT),
            Reveal(t9_text, UP),
            ReplacementTransform(t9_var2, t9_all2[0]),
            TrimCreation(t9_all2[1:], RIGHT),
            Reveal(t9_text2, UP),
            *[FadeIn(m) for i, m in enumerate(game.box) if game.v_mask[i] == 0],
            run_time=2,
        )
        game.v_mask = 81*[1]
        game.not_dig = [i for i in range(81)]
        self.play(
            ShowCreation(t9_grid.line, lag_ratio=0.1),
            game.animate.next_to(t9_grid, LEFT),
            run_time=2,
        )
        self.play(
            Write(t9_grid_n, lag_ratio=0.1, run_time=1),
            Transform(s_text, t_text[9]),
            RouteCreation(t9_rect),
        )
        self.play(
            game.run_func(3, anim=True),
            *[t9_grid_n[i].animate.re_text('1').set_color(GREY) for i in game.dig_buf],
            run_time=2,
        )
        self.play(
            game.run_func(35, anim=True),
            *[t9_grid_n[i].animate.re_text('1').set_color(GREY) for i in game.dig_buf],
            run_time=2,
        )
        grid_color = [WHITE, GREY_A, GREY_B, GREY_C]+[GREY_D]*6
        self.play(
            t9_rect.animate.move_to(Group(t9_all, t9_text)),
            *[FadeOut(game.box[i]) for i in range(81) if game.v_mask[i] == 1],
            *[m.animate.re_text(str(i)).set_color(grid_color[i]) for m, i in zip(t9_grid_n, game.v_map)],
        )
        for i in range(2):
            old_game = game.m_num.copy()
            self.remove(*game.m_num)
            game.remove(game.m_num).set_bomb()
            self.play(
                TransformMatchingShapes(old_game, game.m_num),
                *[m.animate.re_text(str(i)).set_color(grid_color[i]) for m, i in zip(t9_grid_n, game.v_map)],
                run_time=2,
            )
        old_game = game.m_num.copy()
        self.remove(*game.m_num)
        game.remove(game.m_num).set_bomb()
        self.play(
            Uncreate(t9_rect),
            c_rect2.animate.move_to(cli.m_text[2][57:]).set_width(cli.unit_w*15, True),
            TransformMatchingShapes(old_game, game.m_num),
            Reveal(t9_text, UP, rate_func=lambda i: 1-i**4),
            Reveal(t9_text2, UP, rate_func=lambda i: 1-i**4),
            Reveal(t9_all, DOWN, rate_func=lambda i: 1-i**4),
            Reveal(t9_all2, DOWN, rate_func=lambda i: 1-i**4),
            *[m.animate.re_text(str(i)).set_color(grid_color[i]) for m, i in zip(t9_grid_n, game.v_map)],
            run_time=2, remover=True,
        )
        self.add(c_rect2, t9_grid_n, game.m_num)
        
        te_all = Ftext('function', color=BLUE).to_edge(UP, buff=1).shift(FRAME_WIDTH*0.2*LEFT)
        te_var = te_all[0].set_color(ORANGE).copy().align_to(te_all, RIGHT)
        te_text = Text('函数', font='思源黑体').next_to(te_all, DOWN, aligned_edge=RIGHT, buff=0.2)
        te_f = Ftext('f = ').match_y(Group(te_all, te_text))
        te_l = VGroup(*[Ftext(i, color='#dcdcaa') for i in ['...', 'set', 'tmp', 'dig']]).next_to(te_f)
        d_head_2 = cli.m_text[4:12].get_height()*UP
        d_head = d_head*0.5+d_head_2
        te_rect = Rectangle(FRAME_WIDTH, cli.unit_h*5, color=RED, stroke_width=0).match_y(cli.m_text[11])
        te_f[0].set_color('#9cdcfe')
        self.play(
            Uncreate(game.rect),
            Transform(s_text, t_text[10]),
            TransformFromCopy(cli.m_text[2][59], te_var),
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t9_grid_n],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in game.m_num],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t9_grid.line],
            run_time=2,
        )
        cli.m_text.add(te_rect)
        self.add(cli.m_text, bg_rect, te_var, s_text)
        self.play(
            ReplacementTransform(te_var, te_all[0]),
            TrimCreation(te_all[1:], RIGHT),
            Reveal(te_text, UP),
            FadeIn(te_l[0]),
            Write(te_f),
            frame.animate.shift(d_head_2),
            c_rect2.animate.shift(d_head),
            cli.m_text.animate.shift(d_head),
            run_time=2,
        )
        self.play(
            te_rect.animate.set_fill(opacity=0.5),
            FadeOut(te_l[0], UP*0.6),
            FadeIn(te_l[1], UP*0.6),
        )
        self.play(
            te_rect.animate.set_height(cli.unit_h, True).shift(cli.unit_h*7*UP),
            FadeOut(te_l[1], UP*0.6),
            FadeIn(te_l[2], UP*0.6),
        )
        self.play(
            te_rect.animate.set_height(cli.unit_h*4, True).shift(cli.unit_h*2.5*DOWN),
            FadeOut(te_l[2], UP*0.6),
            FadeIn(te_l[3], UP*0.6),
        )
        cli.m_text.remove(te_rect)
        self.add(cli.m_text, te_rect, bg_rect, te_text, te_f, te_all, te_l[3], s_text)
        self.play(
            Uncreate(te_f),
            FadeOut(te_rect, UP),
            Uncreate(te_l[3]),
            Transform(s_text, t_text[11]),
            frame.animate.set_y(cli.unit_h*4),
            cli.m_text.animate.set_y(cli.unit_h*18.3),
            Uncreate(te_text),
            Uncreate(te_all),
            bg_rect.animate.set_y(cli.unit_h*4-FRAME_HEIGHT/2, UP),
            run_time=2,
        )
        self.wait()


class Scene_8(Scene):
    def construct(self):
        text_list = [
            '然后, 就到主函数main()了',
            '前面都好理解, 简单地过一遍',
            '关于循环条件, 这个需要细说',
            '很明显, c的值由后面的表达式决定',
            '为假时固定返回27, c=27结束循环',
            '因为雷数不能超过剩余格子数(B<=s)',
            '所以只有B=s或B=0时, 表达式为假',
            '两种情况也恰好可以区分成功和失败',
            '此处修改B的值是不是很妙?',
            '回到表达式中, 为真时执行getch()',
            '这个"&95", 可以把小写转成大写',
            '不止是字母, 大部分字符都会被影响',
            '如果结果是27, 同样也会结束循环',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        frame = self.camera.frame
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        ).align_to(cli.get_bottom(), UP)
        cli.set_y(cli.unit_h*18.3)
        frame.set_y(cli.unit_h*4)
        t1_line = VGroup(
            Underline(cli.m_text[15][4:26], stroke_width=10, color=RED),
            Underline(cli.m_text[15][28:48], stroke_width=10, color=GREEN),
            Underline(cli.m_text[15][61:69], stroke_width=10, color=BLUE),
            Underline(cli.m_text[15][50:59], stroke_width=10, color=RED),
        )
        for i in t1_line:
            i.align_to(t1_line, DOWN)
        t1_text = VGroup(
            *[Text(i, font='思源黑体').scale(0.8).next_to(j, DOWN).match_color(j)
              for i, j in zip(['变量初值', '内存分配', '初始化随机数'], t1_line)]
        )
        t1_text2 = VGroup(
            *[Text(i, font='思源黑体').scale(0.5).next_to(j, DOWN, buff=1)
              for i, j in zip(['不解释', 'M和m都需要独立空间', 'm的值不固定(*随机分配地址)'], t1_line)]
        )
        self.add(cli.m_text, bg_rect, s_text)
        self.play(
            cli.m_text[16:].animate.shift(5*cli.unit_h*DOWN),
            ShowCreation(t1_line, lag_ratio=0.1),
            *[Reveal(m, UP, rate_func=squish_rate_func(smooth, 0.5+0.1*i, 1)) for i, m in enumerate(t1_text)],
            *[FadeIn(i, rate_func=squish_rate_func(smooth, 0.6, 1)) for i in t1_text2],
            run_time=2,
        )
        t1_line_t = VGroup(
            Underline(cli.m_text[16][11:17], stroke_width=10, color=RED),
            Underline(cli.m_text[16][21:35], stroke_width=10, color=GREEN),
            Underline(cli.m_text[16][41:46], stroke_width=10, color=BLUE),
            Underline(cli.m_text[16][49:57], stroke_width=10, color=YELLOW),
        )
        for i in t1_line_t:
            i.align_to(t1_line_t, DOWN)
        t1_text2_t = VGroup(*[Text(i, font='思源黑体') for i in ['c=27时结束循环;', 'system()返回0, 所以取反;', '触发成功or失败条件;', '获取字符+大小写处理.']])
        t1_text2_t.scale(0.4).arrange(RIGHT, buff=0.2).next_to(t1_line_t, DOWN, buff=1)
        t1_text_t = VGroup(
            *[Text(i, font='思源黑体').scale(0.6).next_to(k, UP).match_color(j)
              for i, j, k in zip(['循环条件', '清屏命令', '状态判定', '监听输入'], t1_line_t, t1_text2_t)]
        )
        turn_animation_into_updater(Transform(s_text, t_text[1]))
        self.play(
            cli.m_text[17:].animate.set_opacity(0.1),
            ShowCreation(t1_line_t, lag_ratio=0),
            *[Reveal(i, UP) for i in t1_text_t],
            *[FadeIn(i) for i in t1_text2_t],
            run_time=2,
        )
        self.wait()
        t1_rect = Rectangle(
            color=RED, stroke_width=5,
        ).surround(cli.m_text[16][11:17], stretch=True, buff=0.5).shift(5*cli.unit_h*UP)
        t1_t_c1 = t1_text_t[0].copy().scale(1.5).to_edge(UL)
        t1_t_c2 = t1_text2_t[0][:-1].copy().scale(1.5).next_to(t1_t_c1, DOWN, aligned_edge=LEFT)
        shift = [UP, DOWN, LEFT, RIGHT]
        self.add(*cli.m_text, bg_rect, t1_text_t[0], t1_text2_t[0], s_text)
        self.play(
            RouteCreation(t1_rect),
            bg_rect.animate.center(),
            Transform(s_text, t_text[2]),
            ReplacementTransform(t1_text_t[0], t1_t_c1),
            TransformMatchingShapes(t1_text2_t[0], t1_t_c2),
            *[i.animate.shift(5*cli.unit_h*UP).fade(0) for i in cli.m_text[16:]],
            *[Uncreate(i) for i in t1_line],
            *[Uncreate(i) for i in t1_line_t],
            *[FadeOutRandom(i) for i in [t1_text, t1_text2, t1_text_t[1:], t1_text2_t[1:]]],
            run_time=2,
        )
        self.wait()
        self.play(
            t1_rect.animate.surround(cli.m_text[16][37:67], stretch=True, buff=0.2),
            Transform(s_text, t_text[3]),
        )
        self.play(
            cli.m_text[16][37:40].animate.set_color(RED).scale(1.2),
            run_time=2, rate_func=there_and_back,
        )
        t4_code = cli.m_text[16][37:67].copy().scale(1.5).next_to(t1_t_c1, buff=1, aligned_edge=UP)
        t4_line = Underline(t4_code[4:9], stroke_width=8, color=BLUE)
        t4_line.add_updater(lambda m: m.next_to(t4_code[4], DOWN, buff=0.5, aligned_edge=LEFT))
        self.play(
            TransformFromCopy(cli.m_text[16][37:67], t4_code),
            ShowCreation(t4_line),
            FadeOut(t1_rect),
            run_time=1,
        )
        t4_line.clear_updaters()
        t4_if_t = Ftext('if', color=BLUE).next_to(t4_line, DOWN)
        t4_true_l = Underline(t4_code[10:25], stroke_width=8, color=GREEN)
        t4_false_l = Underline(t4_code[26:], stroke_width=8, color=RED).match_y(t4_true_l)
        t4_true_t = Ftext('true', color=GREEN).next_to(t4_true_l, DOWN)
        t4_false_t = Ftext('false', color=RED).next_to(t4_false_l, DOWN)
        t4_rect = Square(t4_false_t.get_width()+0.2, color=RED, stroke_width=5).move_to(t4_false_l)
        self.play(
            Reveal(t4_if_t, UP),
            Reveal(t4_true_t, UP),
            Reveal(t4_false_t, UP),
            FadeIn(t4_true_l, UP*0.1),
            FadeIn(t4_false_l, UP*0.1),
            Transform(s_text, t_text[4]),
        )
        self.play(
            t4_code[10:25].animate.set_opacity(0.1),
            t4_true_l.animate.set_opacity(0.1),
            t4_true_t.animate.set_opacity(0.1),
            ShowCreationThenFadeOut(t4_rect),
            run_time=2,
        )
        game = Game_MS(9, 9, 10, 4).next_to(t4_code, DOWN, buff=1.2)
        game.set_bomb_custom([3, 5, 9, 12, 15, 18, 28, 68, 73, 77])
        t5_equa = Ftext('B % s = 0').to_edge(LEFT, buff=0).scale(0.5)
        t5_var = VGroup(*[Ftext(i, color='#9cdcfe') for i in ['W', 'H', 'B', 's']]).arrange(DOWN).next_to(game)
        t5_num = VGroup(*[MyNum(i).next_to(j) for i, j in zip([9, 9, 10, 81], t5_var)])
        t5_text = VGroup(*[Text(i, font='思源黑体') for i in ['长', '宽', '雷数', '剩余']]).arrange(DOWN).match_height(t5_num).next_to(t5_num)
        t5_e_r = Square(4/9, color=RED).move_to(game.box[28])
        t5_e_t = Text('否则必定踩雷!', font='思源黑体', color=RED_D).set_height(0.4).next_to(t5_e_r, DOWN, aligned_edge=LEFT)
        self.add(game)
        self.play(
            Transform(s_text, t_text[5]),
            Write(t5_equa, lag_ratio=0.1),
            frame.animate.center(),
            ShowCreation(game.rect),
            FadeIn(game.box, lag_ratio=0.2),
            FadeIn(game.m_num, lag_ratio=0.05),
            *[Write(i) for i in t5_num],
            FadeIn(t5_text, LEFT*0.2),
            Reveal(t5_var, LEFT),
            run_time=2,
        )
        self.play(
            *[ApplyMethod(j.set_opacity, 0, run_time=random.random()*1.8+0.2) for i, j in zip(game.v_map, game.box) if i != 9],
            ApplyMethod(game.box[28].set_opacity, 0, run_time=2),
            NumberToValue(t5_num[3], 9, run_time=2),
            FadeOut(t5_text, LEFT*0.2, run_time=2),
        )
        self.play(
            Write(t5_e_t, lag_ratio=0.1),
            FadeIn(t5_e_r, scale=0.75),
            run_time=1.5,
        )
        self.wait(0.5)
        t7_t1 = Ftext('B = s').scale(0.5).move_to(t5_equa.get_center()+UR, LEFT)
        t7_t2 = Ftext('B = 0').scale(0.5).move_to(t5_equa.get_center()+DR, LEFT)
        t7_l1 = Line(t7_t1.get_left(), t5_equa.get_top()).scale(0.7, about_edge=UL)
        t7_l2 = Line(t7_t2.get_left(), t5_equa.get_bottom()).scale(0.7, about_edge=DL)
        t7_l1.data['points'][1][:] = t7_l1.get_corner(UL)
        t7_l2.data['points'][1][:] = t7_l2.get_corner(DL)
        t7_l1.add(Triangle(fill_opacity=1, stroke_width=0).scale(0.1).rotate(PI).move_to(t7_l1.get_end()))
        t7_l2.add(Triangle(fill_opacity=1, stroke_width=0).scale(0.1).move_to(t7_l2.get_end()))
        t7_r1 = Rectangle(color=RED).surround(t7_t1)
        t7_r2 = Rectangle(color=RED).surround(Group(t5_num[2:], t5_var[2:]), stretch=True, buff=0.2)
        self.play(
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t5_e_t],
            game.box[28].animate.set_opacity(1),
            t5_num[3].animate.re_num(10),
            Transform(s_text, t_text[6]),
            FadeOut(t5_e_r, scale=0.75),
            run_time=2,
        )
        t5_num[3].text = '10'
        self.play(
            TransformFromCopy(t5_equa, t7_t1),
            TransformFromCopy(t5_equa, t7_t2),
            ShowCreation(t7_l1),
            ShowCreation(t7_l2),
            run_time=2,
        )
        self.play(
            RouteCreation(t7_r1),
            RouteCreation(t7_r2),
        )
        t8_mask = Square(4, fill_opacity=0.8, color=BLACK, stroke_width=0).move_to(game)
        t8_win = Ftext('You win!', color=BLUE).move_to(game)
        self.play(
            Transform(s_text, t_text[7]),
            FadeIn(t8_mask),
            Reveal(t8_win),
            run_time=1.5,
        )
        self.wait(0.5)
        self.play(
            *[ApplyMethod(j.set_opacity, 1, run_time=random.random()*1.8+0.2) for i, j in zip(game.v_map, game.box) if i != 9],
            t8_win.animate.scale(0.6).next_to(t7_t1),
            NumberToValue(t5_num[3], 81, run_time=2),
            FadeOut(t8_mask, run_time=2),
            Uncreate(t7_r1),
            Uncreate(t7_r2),
        )
        game.run_func(40)
        self.play(
            *[ApplyMethod(j.set_opacity, 0, run_time=random.random()*1.8+0.2) for i, j in zip(game.v_mask, game.box) if i == 0],
            ApplyMethod(game.box[28].set_color(RED).set_opacity, 0, run_time=2),
            NumberToValue(t5_num[3], 20, run_time=2),
        )
        t9_text = Text('正常思路会认为\n"B"是静态值', font='思源黑体').scale(0.5).next_to(t5_num[2])
        t9_text[8:].shift(DOWN*0.1)
        t9_over = Ftext('Game over!', color=RED).move_to(game)
        t9_over_c =t9_over.copy().scale(0.6).next_to(t7_t2)
        t9_rect = Rectangle(
            cli.unit_w*5, cli.unit_h, fill_opacity=0.5, color=RED
        ).move_to(cli.m_text[16][63]).scale(1.1)
        t9_r1 = Rectangle(color=RED).surround(t7_t2)
        t9_r2 = Rectangle(color=RED).surround(Group(t5_num[2], t5_var[2]), stretch=True, buff=0.2)
        game_hide = VGroup(*game.m_num[63:], *[game.box[i] for i in [68, 72, 73, 77]])
        cli.m_text.set_opacity(1)
        self.play(
            FadeIn(t9_rect, scale=0.15),
            Transform(s_text, t_text[8]),
            game_hide.animate.set_opacity(0.1),
            frame.animate.shift(cli.unit_h*4*UP),
            game.rect.animate.set_stroke(opacity=0.1), 
            *[i.animate.shift(cli.unit_h*2*UP) for i in cli.m_text],
            run_time=2,
        )
        self.play(FadeIn(t9_text, LEFT), run_time=0.5)
        self.play(
            NumberToValue(t5_num[2], 0),
            RouteCreation(t9_r1),
            RouteCreation(t9_r2),
            run_time=1.5,
        )
        self.play(
            FadeIn(t8_mask),
            Reveal(t9_over),
            Uncreate(t9_rect),
            frame.animate.center(),
            game_hide.animate.set_opacity(1),
            game.rect.animate.set_stroke(opacity=1),
            run_time=2,
        )
        t10_t1 = t7_t1.copy().scale(1.2).next_to(t1_t_c2, DOWN, aligned_edge=LEFT)
        t10_t2 = t7_t2.copy().scale(1.2).next_to(t10_t1, DOWN, aligned_edge=LEFT)
        t10_t3 = Text('成功条件', font='思源黑体', color=BLUE).scale(0.8).next_to(t10_t1)
        t10_t4 = Text('失败条件', font='思源黑体', color=RED).scale(0.8).next_to(t10_t2)
        t10_t5 = Text('剩余格子等于雷数', font='思源黑体').scale(0.5).next_to(t10_t1, DOWN, aligned_edge=LEFT)
        t10_t6 = Text('踩雷后修改"B"值', font='思源黑体').scale(0.5)
        t10_t4.add_updater(lambda i: i.next_to(t10_t2))
        t10_t6.add_updater(lambda i: i.next_to(t10_t2, DOWN, aligned_edge=LEFT))
        self.play(
            ReplacementTransform(t7_t1, t10_t1),
            ReplacementTransform(t7_t2, t10_t2),
            ReplacementTransform(t8_win, t10_t3),
            ReplacementTransform(t9_over, t10_t4),
            *[FadeOut(i) for i in [t8_mask, t9_r1, t9_r2]],
            *[Uncreate(i) for i in [t7_l1, t7_l2, t5_equa]],
            Uncreate(s_text, lag_ratio=0.1),
            run_time=2,
        )
        s_text.become(t_text[9])
        self.play(
            Write(t10_t5),
            Write(t10_t6),
            Write(s_text[:8]),
            t10_t2.animate.next_to(t10_t5, DOWN, aligned_edge=LEFT),
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t5_num],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t5_var],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t9_text],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in game.box],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in game.m_num],
            RouteCreation(game.rect, lambda i: (1-0.2*rush_into(i), 2-1.2*rush_into(i)), remover=True),
            run_time=2,
        )
        self.play(
            Write(s_text[8:]),
            t4_true_l.animate.set_opacity(1),
            t4_true_t.animate.set_opacity(1),
            t4_false_l.animate.set_opacity(0.1),
            t4_false_t.animate.set_opacity(0.1),
            t4_code[26:].animate.set_opacity(0.1),
            t4_code[10:25].animate.set_opacity(1),
        )
        self.wait(1)
        t11_code = t4_code[13:25]
        t4_code.remove(*t11_code)
        self.play(
            t11_code.animate.align_to(t4_code, LEFT),
            *[Uncreate(i) for i in [t4_line, t4_true_l, t4_false_l]],
            *[Uncreate(i) for i in [t4_if_t, t4_true_t, t4_false_t]],
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t4_code],
            run_time=2,
        )
        t11_ascii = VGroup(
            *[Ftext('%03d'%i).scale(0.5).shift(0.5*i*DOWN+LEFT*1.5) for i in range(32, 127)],
            *[Ftext("'%c'"%str(chr(i))).scale(0.5).shift(0.5*i*DOWN) for i in range(32, 127)],
            *[Line(RIGHT*0.5).move_to(0.5*i*DOWN+LEFT*0.75) for i in range(32, 127)],
        ).move_to(BOTTOM, UP).to_edge(RIGHT)
        def DtB(i):
            out = ''
            for count in range(8):
                out = str(i & 1)+out
                i = i >> 1
            return out
        t11_asc_b = VGroup(
            *[Ftext(DtB(i)).shift(0.5*i*DOWN).scale(0.8) for i in range(97, 97+15)]
        ).center().to_edge(RIGHT, buff=3)
        t11_num = Ftext('01011111').next_to(t11_code[11:], DOWN)
        t_text[10:].scale(0.8).to_edge(LEFT)
        self.play(
            Transform(s_text, t_text[10], path_arc=PI/3),
            ApplyMethod(t11_ascii.shift, DOWN*t11_ascii[72].get_y(), rate_func=slow_into),
            run_time=2,
        )
        self.play(
            TransformFromCopy(t11_code[11:], t11_num),
            TransformFromCopy(t11_ascii[65:80], t11_asc_b),
            run_time=1.5,
        )
        self.play(
            t11_num[2].animate.set_color(RED),
            *[i[2].animate.set_color(RED) for i in t11_asc_b],
            run_time=1.5,
        )
        copy_tmp = t11_asc_b[0][0].copy()
        self.play(*[i[2].animate.become(copy_tmp.copy().move_to(i[2])) for i in t11_asc_b])
        self.play(
            *[Transform(i, Ftext("'%c'->%d" % (j+32, j)).scale(0.8).align_to(i, RIGHT).match_y(i)) for i, j in zip(t11_asc_b, range(65, 65+15))],
            t11_ascii.shift, DOWN*t11_ascii[40].get_y(),
            run_time=2,
        )
        t12_target = [31, 32, 33, 34, 63, 64, 65, 66, 95, 96, 97, 98, 99]
        t12_more = Ftext('...').scale(0.5)
        t12_list = VGroup(
            *[Ftext('%03d'%i).scale(0.5).shift(0.5*j*DOWN+1.5*LEFT) if j % 4 else t12_more.copy().shift(0.5*j*DOWN+1.5*LEFT) for j, i in enumerate(t12_target)],
            *[Ftext("'%c'"%str(chr(i))).scale(0.5).shift(0.5*j*DOWN) if j % 4 else t12_more.copy().shift(0.5*j*DOWN) for j, i in enumerate(t12_target)],
            *[Line(0.5*i*DOWN+LEFT, 0.5*LEFT, color=RED) for i in range(5)],
            *[Line(0.5*i*DOWN+LEFT, 0.5*i*DOWN+0.5*LEFT) for i in range(5, 9)],
            *[Line(0.5*i*DOWN+LEFT, 0.5*(i-4)*DOWN+0.5*LEFT, color=RED) for i in range(9, 13)],
        ).center().to_edge(RIGHT)
        self.play(
            FadeOut(t11_num),
            FadeOut(t11_asc_b, LEFT),
            Transform(s_text, t_text[11]),
            run_time=1.5,
        )
        self.play(
            FadeIn(t12_list[0], t11_ascii[0].get_y()*DOWN),
            FadeIn(t12_list[13], t11_ascii[0].get_y()*DOWN),
            ReplacementTransform(t11_ascii[3:32], t12_list[4]),
            ReplacementTransform(t11_ascii[35:64], t12_list[8]),
            ReplacementTransform(t11_ascii[67:95], t12_list[12]),
            ReplacementTransform(t11_ascii[190:], t12_list[26:]),
            ReplacementTransform(t11_ascii[98:127], t12_list[17]),
            ReplacementTransform(t11_ascii[130:159], t12_list[21]),
            ReplacementTransform(t11_ascii[162:190], t12_list[25]),
            *[TransformMatchingShapes(t11_ascii[i-32], t12_list[j]) for j, i in enumerate(t12_target) if j % 4],
            *[TransformMatchingShapes(t11_ascii[i+63], t12_list[j+13]) for j, i in enumerate(t12_target) if j % 4],
            run_time=3,
        )
        t13_text = Text('input|out').move_to(t11_num, UP)
        t13_key = VGroup(*[Square(0.6) for i in range(10)], Rectangle(1.3, 0.6)).arrange(DOWN, buff=0.1)
        t13_key[5:10].next_to(t13_key[2], buff=0.1)
        t13_key[10].next_to(t13_key[4], DOWN, buff=0.1, aligned_edge=LEFT)
        t13_num = VGroup(*[MyNum(i).next_to(j) for i, j in zip([65, 68, 83, 87, 27, 0], t13_key[5:].set_stroke(color=RED))])
        t13_list = VGroup(t13_key.set_fill(BLACK, 1), t13_num).next_to(t13_text, DOWN).add(t13_text)
        for m, t in zip(t13_key, ['A', 'D', 'S', 'W', 'Esc', 'a', 'd', 's', 'w', ';', 'Space']):
            m.add(Ftext(t).move_to(m).scale(0.5))
        t13_key[4][0].set_width(0.4)
        t13_key[10][0].set_width(1)
        t13_r1 = Rectangle(color=GREEN).surround(t13_num[4], buff=0.2)
        t13_t2 = Ftext('getch() = 27').scale(0.5).next_to(t10_t6, DOWN, aligned_edge=LEFT)
        t13_t3 = Text('退出', font='思源黑体', color=GREEN).scale(0.8).next_to(t13_t2)
        t13_t4 = Text('Esc键退出(常规配置)', font='思源黑体').scale(0.5).next_to(t13_t2, DOWN, aligned_edge=LEFT)
        t13_t5 = Text(':被与运算影响的输入', font='思源黑体').scale(0.5)
        t13_t5.add(Square(0.2, color=RED).next_to(t13_t5, LEFT)).next_to(t13_key, DOWN, aligned_edge=LEFT)
        self.play(
            Write(t13_t5),
            Write(t13_text),
            *[Write(i) for i in t13_num],
            *[RouteCreation(i, lambda i:(0, i)) for i in t13_key],
            run_time=2,
        )
        self.play(
            ReplacementTransform(t11_code, t13_t2),
            Transform(s_text, t_text[12]),
            RouteCreation(t13_r1),
            Reveal(t13_t3),
            Write(t13_t4),
            run_time=1.5,
        )
        self.wait(2.5)
        self.play(
            Uncreate(t13_r1),
            t13_list.animate.to_edge(UR),
            Uncreate(s_text, lag_ratio=0.1),
            frame.animate.shift(cli.unit_h*4*UP),
            *[AnimationGroup(*[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in j])
              for j in [t10_t1, t10_t2, t10_t3, t10_t4, t10_t5, t10_t6, t13_t2,
                        t13_t3, t13_t4, t13_t5, t1_t_c1, t1_t_c2, t12_list]],
            run_time=2,
        )
        self.wait()


class Scene_9(Scene):
    def construct(self):
        text_list = [
            '输入后, 自然就需要处理输入数据了',
            '"adsw"移动坐标, 直接枚举就完了',
            '此外, 还需要一点限制防止光标越界',
            '空格对应左键操作, 解析坐标后调用函数指针',
            '这时, 才会有触雷判定, 很简单的判定',
            '最后, 就剩可视化界面了...',
            '输出部分老套路了, 懂得都懂',
            '还有个表达式, 完善失败效果',
            '接下来就是游戏最关键的算法了',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).scale(0.9).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        frame = self.camera.frame
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        )
        old_text = Text('input|out')
        old_key = VGroup(*[Square(0.6) for i in range(10)], Rectangle(1.3, 0.6)).arrange(DOWN, buff=0.1)
        old_key[5:10].next_to(old_key[2], buff=0.1)
        old_key[10].next_to(old_key[4], DOWN, buff=0.1, aligned_edge=LEFT)
        old_num = VGroup(*[MyNum(i).next_to(j) for i, j in zip([65, 68, 83, 87, 27, 0], old_key[5:].set_stroke(color=RED))])
        old_list = VGroup(old_key.set_fill(BLACK, 1), old_num).next_to(old_text, DOWN).add(old_text).to_edge(UR)
        for m, t in zip(old_key, ['A', 'D', 'S', 'W', 'Esc', 'a', 'd', 's', 'w', ';', 'Space']):
            m.add(Ftext(t).move_to(m).scale(0.5))
        old_key[4][0].set_width(0.4)
        old_key[10][0].set_width(1)
        cli.set_y(cli.unit_h*20.3)
        frame.set_y(cli.unit_h*4)
        t1_grid = MyGrid(9, 9, 5).to_edge(UP)
        t1_rect = Square(5/9, color=RED, fill_opacity=0.5).move_to(t1_grid)
        t1_code = VGroup(
            cli.m_text[17][8:21].copy(),
            cli.m_text[17][23:36].copy(),
            cli.m_text[17][38:56].copy(),
            cli.m_text[17][58:76].copy(),
            Ftext('c - 0 || (...)', t2c={'c': '#9cdcfe', '0': '#b5cea8'}).match_height(cli.m_text[-1]),
        ).arrange(DOWN, aligned_edge=LEFT).scale(1.2).next_to(old_list, LEFT)
        t1_var = VGroup(
            VGroup(*[Ftext(i, color='#9cdcfe') for i in ['W', 'H', 'p', 'c']]).arrange(DOWN),
            VGroup(*[MyNum(i) for i in [9, 9, 40, 1]]).arrange(DOWN, aligned_edge=LEFT),
            VGroup(*[Text(i, font='思源黑体').scale(0.8) for i in ['长度', '宽度', '坐标', '临时']]).arrange(DOWN),
        ).arrange(RIGHT).next_to(t1_grid, RIGHT, aligned_edge=DOWN)
        t1_r1 = VGroup(*[Square(color=RED).surround(i[4:6], buff=0.2) for i in t1_code])
        t1_r2 = VGroup(*[Rectangle(color=BLUE).surround(i, buff=0.5) for i in old_num[:4]])
        t1_r2.add(Square(color=BLUE).surround(old_num[5], buff=0.2))
        t1_l1 = VGroup(*[Arrow(j.get_left(), i.get_right(), fill_color=GREEN) for i, j in zip(t1_r1, t1_r2)])
        self.play(
            Write(s_text),
            TransformFromCopy(cli.m_text[17][8:21], t1_code[0]),
            TransformFromCopy(cli.m_text[17][23:36], t1_code[1]),
            TransformFromCopy(cli.m_text[17][38:56], t1_code[2]),
            TransformFromCopy(cli.m_text[17][58:76], t1_code[3]),
            TransformMatchingShapes(cli.m_text[18][25:].copy(), t1_code[4]),
            run_time=2,
        )
        self.play(
            ShowCreation(t1_r1, lag_ratio=0.1),
            ShowCreation(t1_r2, lag_ratio=0.2),
            FadeIn(t1_l1, lag_ratio=0.3),
            old_key.animate.set_opacity(0.1),
            *[i[7:].animate.set_opacity(0.1) for i in t1_code],
            run_time=2,
        )
        self.wait(0.5)
        self.play(
            *[i[7:].animate.set_opacity(1) for i in t1_code],
            *[FadeOut(i) for i in [t1_r1, t1_r2, t1_l1]],
            old_key.animate.set_opacity(1),
            run_time=0.5,
        )
        t2_num = t1_grid.add_number(DR)
        t2_move = VGroup(*[Square(5/9, color=BLUE, fill_opacity=0.5).move_to(t1_grid.box[i]) for i in [39, 41, 49, 31]])
        for m, i in zip(t2_move, ['-1', '+1', '+9', '-9']):
            m.add(Ftext(i, color=GREEN).scale(0.5).move_to(m))
        t2_key = old_key[:5].copy().add(old_key[10].copy().set_width(2, True))
        t2_key[5].set_stroke(color=WHITE)[0].set_width(1, True)
        for i, j in zip(t2_key, [LEFT, RIGHT, ORIGIN, UP, UL, DOWN]):
            i.move_to(j*0.7)
        t2_key.next_to(t1_grid, RIGHT, aligned_edge=UP)
        t2_rect = Rectangle(3.3, 0.5, fill_opacity=0.5, color=RED)
        self.play(
            FadeOut(old_num, LEFT),
            FadeOut(old_key[5:10]),
            Transform(s_text, t_text[1]),
            Uncreate(old_text, lag_ratio=0.1),
            ShowCreation(t1_grid.line, lag_ratio=0.1),
            ReplacementTransform(old_key[10], t2_key[5]),
            t1_code.animate.scale(5/6).next_to(t1_grid, LEFT, aligned_edge=UP),
            *[ReplacementTransform(i, j) for i, j in zip(old_key[:5], t2_key[:5])],
            run_time=2,
        )
        self.play(
            RouteCreation(t1_rect),
            Reveal(t1_var[0], LEFT),
            FadeIn(t1_var[2], LEFT*0.5),
            *[Write(i) for i in t1_var[1]],
            run_time=2,
        )
        self.play(
            Write(t2_num, lag_ratio=0.1),
            FadeOut(t1_var[2], LEFT*0.5, rate_func=rush_into),
            run_time=1.5,
        )
        self.play(AnimationGroup(
            *[TransformFromCopy(i[10:], j) for i, j in zip(t1_code[:4], t2_move)],
            lag_ratio=0.5, run_time=2.5,
        ))
        grid_w = 5/9
        t2_move.add_to_back(t1_rect)
        t1_var[1][3].re_num(68)
        def r_func(i): return smooth(1-i)
        t1_var[1][2].re_num(41)
        self.play(
            FadeOut(s_text, DOWN),
            t2_move.animate.shift(grid_w*RIGHT),
            FadeOut(t2_rect.match_x(t1_code).match_y(t1_code[1])),
            ApplyMethod(t2_key[1].set_color, RED, rate_func=r_func),
            run_time=1,
        )
        t1_var[1][2].re_num(42)
        self.play(
            FadeOut(t2_rect),
            t2_move.animate.shift(grid_w*RIGHT),
            ApplyMethod(t2_key[1].set_color, RED, rate_func=r_func),
            run_time=0.8,
        )
        t1_var[1][3].re_num(83)
        self.play(
            NumberToValue(t1_var[1][2], 51),
            t2_move.animate.shift(grid_w*DOWN),
            FadeOut(t2_rect.match_y(t1_code[2])),
            ApplyMethod(t2_key[2].set_color, RED, rate_func=r_func),
            run_time=0.6,
        )
        self.play(
            FadeOut(t2_rect),
            NumberToValue(t1_var[1][2], 60),
            t2_move.animate.shift(grid_w*DOWN),
            ApplyMethod(t2_key[2].set_color, RED, rate_func=r_func),
            run_time=0.4,
        )
        t1_var[1][2].re_num(61)
        t1_var[1][3].re_num(68)
        self.play(
            t2_move.animate.shift(grid_w*RIGHT),
            FadeOut(t2_rect.match_y(t1_code[1])),
            ApplyMethod(t2_key[1].set_color, RED, rate_func=r_func),
            run_time=0.4,
        )
        t1_var[1][2].re_num(62)
        self.play(
            FadeOut(t2_rect),
            t2_move[2].animate.shift(grid_w*8*LEFT+grid_w*DOWN),
            ApplyMethod(t2_key[1].set_color, RED, rate_func=r_func),
            *[i.animate.shift(grid_w*RIGHT) for i in [*t2_move[:2], *t2_move[3:]]],
            run_time=0.6,
        )
        t1_var[1][3].re_num(83)
        self.play(
            Write(s_text.become(t_text[2])[:4]),
            NumberToValue(t1_var[1][2], 71),
            t2_move.animate.shift(grid_w*DOWN),
            FadeOut(t2_rect.match_y(t1_code[2])),
            ApplyMethod(t2_key[2].set_color, RED, rate_func=r_func),
            run_time=0.8,
        )
        self.play(
            FadeOut(t2_rect),
            Write(s_text[4:]),
            frame.animate.center(),
            NumberToValue(t1_var[1][2], 80),
            t2_move.animate.shift(grid_w*DOWN),
            ApplyMethod(t2_key[2].set_color, RED, rate_func=r_func),
            run_time=2,
        )
        t3_code1 = Ftext(
            'p = p % S', t2c={'p': '#9cdcfe', 'S': '#9cdcfe'}
        ).scale(0.5).align_to(t1_code, LEFT).align_to(t1_grid, DOWN)
        t3_code2 = Ftext(
            'p = (p + S) % S', t2c={'p': '#9cdcfe', 'S': '#9cdcfe'}
        ).scale(0.5).match_y(t3_code1).align_to(t3_code1, LEFT)
        t3_text = Text('最简单的方法: 模除', font='思源黑体').scale(0.5).next_to(t3_code1, UP, aligned_edge=LEFT)
        self.play(t2_move[2:4].animate.set_stroke(color=RED, width=5), rate_func=there_and_back)
        self.play(Write(t3_text), Write(t3_code1), run_time=1)
        self.play(t2_move[2:4].animate.shift(grid_w*9*UP))
        t1_var[1][3].re_num(68)
        self.play(
            NumberToValue(t1_var[1][2], 0),
            t2_move[0].animate.shift(grid_w*8*UL),
            t2_move[1].animate.shift(grid_w*9*UP+grid_w*RIGHT),
            t2_move[2].animate.shift(grid_w*RIGHT),
            t2_move[3].animate.shift(grid_w*8*LEFT+grid_w*DOWN),
            t2_move[4].animate.shift(grid_w*8*UL),
            FadeOut(t2_rect.match_y(t1_code[1])),
            ApplyMethod(t2_key[1].set_color, RED, rate_func=r_func),
            run_time=1.5,
        )
        self.play(TransformMatchingShapes(t3_code1, t3_code2))
        self.play(
            t2_move[1].animate.shift(grid_w*9*DOWN),
            t2_move[4].animate.shift(grid_w*9*DOWN),
            FadeOut(s_text, DOWN),
        )
        self.wait(0.5)
        t4_game = Game_MS(9, 9, 10).to_edge(UP)
        t4_game.set_bomb_custom([33, 38, 49, 51, 53, 61, 62, 65, 74, 79])
        t4_game.run_func(0)
        t4_code = Ftext(
            'c || (\n    f(p % W, p / W),\n    m[p] < 9 || (B = 0)\n)',
        ).scale(0.5).next_to(t1_grid, LEFT, aligned_edge=UP)
        c_style_set_color(t4_code, t4_code.text, {'f': '#9cdcfe'})
        t4_rect = Rectangle(
            fill_opacity=0.5, color=RED,
        ).surround(cli.m_text[18][25:68], stretch=True, buff=0.5)
        t4_t1 = Ftext(
            'f(47 % 9, 47 / 9)', t2c={'f': '#9cdcfe', '9': '#b5cea8'}
        ).scale(0.5).next_to(t4_code, DOWN)
        t4_n1 = MyNum(47, color='#b5cea8').scale(0.5).move_to(t4_t1[2:4])
        t4_n2 = MyNum(47, color='#b5cea8').scale(0.5).move_to(t4_t1[10:12])
        t4_t1.remove(*t4_t1[2:4], *t4_t1[10:12]).add(t4_n1, t4_n2)
        t4_t2 = Ftext(
            'f(2, 5)', t2c={'f': '#9cdcfe', '2': '#b5cea8', '5': '#b5cea8'}
        ).scale(0.5).next_to(t4_t1, DOWN, aligned_edge=LEFT)
        t4_t3 = t4_t2[1:].copy().add_updater(lambda i: i.next_to(t1_rect, LEFT))

        shift = [UP, DOWN, LEFT, RIGHT]
        self.play(
            Uncreate(t3_text, lag_ratio=0.1),
            frame.animate.shift(4*cli.unit_h*UP),
            Transform(t3_code2, cli.m_text[18][8:23], remover=True),
            run_time=1.5,
        )
        t2_move.remove(t1_rect)
        self.add(*t4_game, t1_rect)
        self.play(
            *[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in t1_grid.line],
            FadeIn(t4_game.m_num, lag_ratio=0.05),
            t1_rect.animate.set_fill(opacity=0),
            FadeIn(t4_game.box, lag_ratio=0.1),
            Write(s_text.become(t_text[3])[:10]),
            FadeOut(t2_move),
            FadeOut(t2_num),
            run_time=2,
        )
        t1_var[1][3].re_num(0)
        self.play(
            FadeOut(t2_rect.match_y(t1_code[4])),
            ApplyMethod(t2_key[5].set_color, RED, rate_func=r_func),
            FadeOut(VGroup(*[t4_game.box[j] for j in t4_game.dig_buf]), lag_ratio=0.2),
            run_time=2,
        )
        self.play(
            *[AnimationGroup(*[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in j]) for j in t1_code[:4]],
            TransformMatchingShapes(t1_code[4], t4_code),
            Reveal(t4_rect, LEFT),
            Write(s_text[10:]),
            run_time=2,
        )
        self.play(
            t1_rect.animate.shift(2*grid_w*RIGHT+5*grid_w*DOWN),
            NumberToValue(t1_var[1][2], 47),
            FadeIn(t4_t3, RIGHT),
            run_time=2,
        )
        self.play(TransformFromCopy(t4_code[11:26], t4_t1.copy(), remover=True))
        self.add(t4_t1)
        self.play(TransformFromCopy(t4_t1, t4_t2.copy(), remover=True))
        self.add(t4_t2)
        self.play(
            ApplyMethod(t4_t3.set_color, RED, rate_func=r_func),
            ApplyMethod(t4_t2.set_color, RED, rate_func=r_func),
            t4_game.run_func(47, anim=True),
            FadeOut(s_text, DOWN),
            run_time=1,
        )
        t4_t3_x = t4_t3.clear_updaters().get_x()
        for pos, move in [[57, DR], [77, 2*DR]]:
            self.play(
                t1_rect.animate.shift(grid_w*move),
                *[NumberToValue(i, pos) for i in [t1_var[1][2], t4_n1, t4_n2]],
                UpdateFromFunc(t4_t2, lambda i: i.re_text(
                    'f(%d, %d)'%(t4_n1.number % 9, t4_n1.number // 9),
                ).set_color_by_t2c({'f': '#9cdcfe', '[2:3]': '#b5cea8', '[5:6]': '#b5cea8'})),
                UpdateFromFunc(t4_t3, lambda i: i.become(t4_t2[1:]).set_x(t4_t3_x).match_y(t1_rect)),
                run_time=1.5,
            )
            self.play(t4_game.run_func(pos, anim=True),run_time=0.5)
        t5_text = VGroup(*[Text(i, font='思源黑体') for i in ['当前位置', '雷', '失败条件']]).scale(0.5)
        t5_text[0].next_to(t4_code[32:36], DOWN)
        t5_text[1].next_to(t4_code[37:42], DOWN)
        t5_text[2].next_to(t4_code[44:51], DOWN)
        self.play(
            *[AnimationGroup(*[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in j])
              for j in [t4_t1, t4_t2, t4_t3]],
            RouteCreation(Underline(t4_code[32:51], stroke_width=10, color=RED), lambda i: (rush_from(i), max(0, 5*i-4)), remover=True),
            t1_rect.animate.shift(grid_w*3*LEFT+grid_w*2*UP),
            Reveal(t5_text, UP, rate_func=smooth),
            Write(s_text.become(t_text[4])[:14]),
            NumberToValue(t1_var[1][2], 56),
            t4_game.run_func(56, anim=True),
            run_time=2,
        )
        self.play(
            Write(s_text.become(t_text[4])[14:]),
            t1_rect.animate.shift(grid_w*UR),
            NumberToValue(t1_var[1][2], 48),
            t4_game.run_func(48, anim=True),
            run_time=1.2,
        )
        self.play(
            t1_rect.animate.shift(grid_w*2*RIGHT),
            NumberToValue(t1_var[1][2], 50),
            t4_game.run_func(50, anim=True),
            run_time=0.8,
        )
        self.play(
            t1_rect.animate.shift(grid_w*UR),
            NumberToValue(t1_var[1][2], 42),
            t4_game.run_func(42, anim=True),
            FadeOut(t5_text, DOWN*0.5),
            FadeOut(s_text, DOWN),
            run_time=1,
        )
        for i, j in zip(t4_game.box, t4_game.v_mask):
            i.set_opacity(0) if j == 0 else 0
        self.play(
            *[AnimationGroup(*[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in j])
              for j in [*t1_var[:2], t2_key, t4_code]],
            cli.m_text.animate.shift(cli.unit_h*2*UP),
            FadeOut(t4_rect, cli.unit_h*2*UP),
            Write(s_text.become(t_text[5])),
            run_time=2,
        )
        def map_to_str(mob, pos):
            out = ''
            for i, m in enumerate(zip(mob.v_mask, mob.v_map)):
                out += ' >'[i == pos]+('*' if m[0] else ' 12345678@'[m[1]])
                if (i+1) % mob.n_cols == 0:
                    out += '\n'
            return out.split('\n')
        t6_con = Console(lines=10, width=5).next_to(ORIGIN, RIGHT).to_edge(UP)
        for i, m in enumerate(map_to_str(t4_game, 42)):
            t6_con.new_text(m, i)
        t6_con.cursor.shift(9*t6_con.unit_h*DOWN)
        t6_line = VGroup(
            VGroup(
                Underline(cli.m_text[19][8:18], stroke_width=10),
                Underline(cli.m_text[19][45:55], stroke_width=10),
            ).set_color(YELLOW),
            Underline(cli.m_text[19][56:75], stroke_width=10, color=GREEN),
            Underline(cli.m_text[20][29:41], stroke_width=10, color=BLUE),
            Underline(cli.m_text[20][43:74], stroke_width=10, color=RED),
        )
        t6_text = VGroup(*[
            Text(i, font='思源黑体').scale(0.8).next_to(j, DOWN, 0.05, RIGHT).match_color(j)
            for i, j in zip(['遍历', '换行', '光标', '元素'], t6_line)
        ])
        t6_t2 = VGroup(*[
            Text(i, font='思源黑体').scale(0.4).to_edge(UR).shift(j*DOWN)
            for i, j in zip(['循环的标准配置', '打印每一行后执行', '下标相同时打印光标', '根据m和M打印元素'], [0.6, 2.1, 3.6, 5.1])
        ])
        t6_r1 = Rectangle(
            t6_con.unit_w*18, t6_con.unit_h*9, stroke_width=2, color=YELLOW
        ).move_to(t6_con.get_grid_place(8.5, 4))
        t6_r2 = Rectangle(
            t6_con.unit_w, t6_con.unit_h*9, color=GREEN, stroke_width=0.5, fill_opacity=0.5,
        ).move_to(t6_con.get_grid_place(18, 4))
        t6_r3 = VGroup(*[
            t6_r2.copy().set_color(BLUE).shift(t6_con.unit_w*i*2*LEFT)
            for i in range(1, 10)
        ])
        t6_r4 = t6_r3.copy().set_color(RED).shift(t6_con.unit_w*RIGHT)
        self.play(
            ShowCreation(t6_con, lag_ratio=0.01),
            t4_game.add(t1_rect).animate.next_to(ORIGIN, LEFT).to_edge(UP),
            run_time=2,
        )
        self.play(
            ShowCreation(t6_line[:2], lag_ratio=0),
            Reveal(t6_text[:2], UP),
            run_time=2,
        )
        self.play(
            Transform(s_text, t_text[6]),
            ShowCreation(t6_r1),
            Reveal(t6_r2, LEFT),
            run_time=1,
        )
        Group(Dot(t4_game.rect.get_corner(UL), radius=0), t6_r3, t6_r4).to_edge(UL)
        self.play(
            Group(t4_game, t6_con, t6_r1, t6_r2).animate.to_edge(LEFT),
            t6_text[1].animate.to_edge(UR).shift(DOWN*1.5),
            t6_text[0].animate.to_edge(UR),
            *[Write(i) for i in t6_t2[:2]],
            run_time=2,
        )
        self.play(
            t6_line[:2].animate.set_opacity(0.4),
            ShowCreation(t6_line[2:], lag_ratio=0),
            Reveal(t6_text[2:], UP),
            run_time=2,
        )
        self.play(
            t6_text[3].animate.to_edge(UR).shift(DOWN*4.5),
            t6_text[2].animate.to_edge(UR).shift(DOWN*3),
            t6_line[2:].animate.set_opacity(0.4),
            *[Write(i) for i in t6_t2[2:]],
            FadeOut(s_text, DOWN),
            Reveal(t6_r4, DOWN),
            Reveal(t6_r3, UP),
            run_time=1,
        )
        self.wait()
        t8_rect = Rectangle(fill_opacity=0.5, color=RED).surround(cli.m_text[19][20:43], stretch=True, buff=0.5)
        t8_text = Text('踩雷后翻开所有炸弹格子', font='思源黑体', color=RED).scale(0.7).next_to(t8_rect, DOWN, 0.2)
        t8_bomb = []
        def anim_func(pos):
            anim = t4_game.run_func(pos, anim=True)
            for i, m in enumerate(map_to_str(t4_game, pos)):
                t6_con.new_text(m, i)
            return anim
        self.play(
            anim_func(52),
            t1_rect.animate.shift(grid_w*DR),
            Write(s_text.become(t_text[7])[:10]),
            *[FadeOut(i) for i in [t6_r1, t6_r2, t6_r3, t6_r4]],
            run_time=1,
        )
        self.play(
            anim_func(79),
            Write(s_text[10:]),
            t1_rect.animate.shift(grid_w*3*DOWN),
            run_time=1,
        )
        for i, m in enumerate(t4_game.v_map[:78]):
            if m == 9:
                t4_game.v_mask[i] = 0
                t8_bomb.append(t4_game.box[i])
        self.play(
            FadeIn(t8_rect, scale=0.2),
            Write(t8_text),
            run_time=2,
        )
        for i, m in enumerate(map_to_str(t4_game, 79)):
            t6_con.new_text(m, i)
        self.play(
            *[i.animate.set_opacity(0) for i in t8_bomb],
            run_time=2,
        )
        self.wait()
        t4_game.box[52].set_opacity(0)
        t4_game.box[79].set_opacity(0)
        self.play(
            frame.animate.shift(4*cli.unit_h*UP),
            bg_rect.animate.shift(4*cli.unit_h*DOWN),
            cli.m_text.animate.shift(12*cli.unit_h*DOWN),
            *[AnimationGroup(*[FadeOut(i, random.random()*0.5*shift[random.randint(0, 3)]) for i in j])
              for j in [[t4_game.rect, t1_rect, *t6_con[:2], t6_con.cursor, t8_rect], t4_game.m_num, t4_game.box, *t6_con[2:4], *t6_con.m_text, *t6_text, *t6_t2, t6_line, t8_text]],
            RouteCreation(s_text, lambda i: (i, 1), lag_ratio=0.5),
            Write(t_text[8]),
            run_time=2,
        )
        self.wait()


class Scene_10(Scene):
    def construct(self):
        around_d = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        text_list = [
            '接下来就是游戏最关键的算法了',
            'edge(), 判断输入坐标是否越界',
            '功能明显, 比较简单, 属于常规操作',
            'tmp(), 对输入坐标对应的m值进行累加',
            '未越界且非雷才会累加, 只会围绕地雷执行',
            '逐个执行的效果可能有点晕? 整合一下再执行',
            '再稍微整合一下...就是set()了, 生成地雷',
            '生成一定数量的雷, 且输入坐标附近不生成雷',
            'dig(), 翻开输入坐标, 是空地则翻开周围',
            '正常流程只有第一次执行set(), 后续仅执行dig()',
            '如何调用这些函数呢, 先写点简单的代码',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).scale(0.9).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        frame = self.camera.frame
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        )
        cli.set_y(cli.unit_h*14.3)
        frame.set_y(cli.unit_h*12)
        t1_title = Text('函数和函数指针', font='思源黑体').to_edge(UP, buff=0.2)
        t1_rect = VGroup(
            Rectangle().surround(cli.m_text[2][57:], stretch=True),
            Rectangle().surround(cli.m_text[3][:22], stretch=True),
            Rectangle().surround(cli.m_text[4][:22], stretch=True),
            Rectangle().surround(cli.m_text[5][:22], stretch=True),
            Rectangle().surround(cli.m_text[9][:22], stretch=True),
        ).set_fill(RED, 0.5).set_stroke(RED)
        t1_text = VGroup(
            VGroup(*[Ftext(i+'()', color='#dcdcaa', t2c={'[-2:]': WHITE}) for i in ['edge', 'tmp', 'dig', 'set']]).arrange(DOWN, aligned_edge=LEFT, buff=1.2),
            VGroup(*[Text(i, font='思源黑体') for i in ['越界判定', '数值累加', '挖掘动作', '生成地雷']]).scale(0.8),
        ).scale(0.8).to_edge(LEFT).to_edge(UP, buff=1)
        t1_grid = MyGrid(9, 9)
        t1_axis = VGroup(
            *[MyNum(i).move_to(t1_grid.box[i*9].get_center()+5/9*LEFT).scale(0.8) for i in range(9)],
            *[MyNum(i).next_to(t1_grid.box[i], UP, buff=0.1).scale(0.8) for i in range(9)],
        )
        Group(t1_grid, t1_axis).to_edge(UP, buff=0.75)
        t1_r_pos = []
        for i, j in zip(t1_text[0], t1_text[1]):
            j.next_to(i, DOWN, aligned_edge=LEFT, buff=0.2)
            t1_r_pos.append(Group(i, j).get_center())
        t1_text_r = Rectangle(2.5, 1.5, color=RED).move_to(t1_r_pos[0])

        self.add(t1_rect, cli.m_text, bg_rect, s_text)
        self.play(
            *[FadeIn(i, scale=0.5) for i in t1_rect],
            Write(t1_title),
            run_time=1.5,
        )
        self.wait(0.5)
        t2_code = cli.m_text[3].copy().scale(0.9).to_edge(UL, buff=0.3)
        t2_func = VGroup(Ftext('input|out')).to_edge(UR, buff=0.3).scale(0.8, about_edge=UL)
        t2_rect = VGroup(Dot(radius=0))
        t1g_b_dot = t1_grid.box[0].get_center()
        def t2_edge_func():
            x = random.randint(0, 8) if random.random() < 0.85 else (-1  if random.random() < 0.5 else 9)
            y = random.randint(0, 8) if random.random() < 0.85 else (-1  if random.random() < 0.5 else 9)
            outside = (x+1) % 10 and (y+1) % 10
            m_color = GREEN if outside else RED
            rect = Square(5/9, color=m_color).shift(5/9*(np.array([x, -y, 0]))+t1g_b_dot)
            text = Ftext(
                'edge(%d, %d)=>%d'%(x, y, not outside),
                t2c={'[:4]': '#dcdcaa', '[-1:]': m_color}
            ).set_height(0.35).move_to(t2_func[-1].get_left()+0.4*DOWN, LEFT)
            anim = [FadeIn(rect, scale=0.5)]
            if 15 < len(t2_func):
                text.shift(0.4*UP)
                anim += [FadeOut(t2_func[0], 0.5*LEFT)]
                anim += [FadeOut(t2_rect[0])]
                t2_rect.remove(t2_rect[0])
                t2_func.remove(t2_func[0])
                anim += [i.animate.shift(0.4*UP) for i in t2_func]
            anim += [FadeIn(text, 0.4*RIGHT)]
            t2_rect.add(rect)
            t2_func.add(text)
            return AnimationGroup(*anim)
        self.play(
            frame.animate.center(),
            Write(t1_axis, lag_ratio=0.5),
            Uncreate(t1_title, lag_ratio=0),
            Uncreate(s_text, lag_ratio=0),
            *[Write(i) for i in t1_text[1]],
            TransformFromCopy(cli.m_text[3], t2_code),
            RouteCreation(t1_grid.line, lag_ratio=0.1),
            TransformFromCopy(cli.m_text[3][4:8], t1_text[0][0]),
            TransformFromCopy(cli.m_text[4][5:8], t1_text[0][1]),
            TransformFromCopy(cli.m_text[5][5:8], t1_text[0][2]),
            TransformFromCopy(cli.m_text[9][5:8], t1_text[0][3]),
            run_time=2,
        )
        self.wait()
        self.play(
            ShowCreation(t1_text_r),
            Write(s_text.become(t_text[1])),
            Write(t2_func[0]),
            run_time=1,
        )
        for i in range(10):
            self.play(t2_edge_func(), run_time=1-i*0.1)
        turn_animation_into_updater(Transform(s_text, t_text[2]))
        for i in range(40):
            self.play(t2_edge_func(), run_time=max(0.1, (i-20)/30))
        t3_code = cli.m_text[4].copy().scale(0.8).to_edge(UL, buff=0.3)
        t3_num = t1_grid.add_number(size=1, init=lambda i: 0)
        t3_func = VGroup(Ftext('input|out')).to_edge(UR, buff=0.3).scale(0.8, about_edge=UL)
        t3_rect = VGroup(Dot(radius=0))
        t3_rect_s = Square(5/9, color=BLUE)
        t3_num_color = [WHITE, GREEN, BLUE, RED, YELLOW, GREY_BROWN, PURPLE, PINK, LIGHT_BROWN, BLACK]
        def t3_tmp_func(x_i, y_i, **var):
            t3_num[x_i+y_i*9].text = '10'
            for x, y in [[0, 0], [-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]:
                x, y = x+x_i, y+y_i
                target = t3_num[x+y*9]
                target_i = int(target.text)
                auOK = (x+1) % 10 and (y+1) % 10 and target_i != 9
                rect = Square(
                    5/9, color=GREEN if auOK else RED, fill_opacity=0.2, stroke_width=0,
                ).shift(5/9*(np.array([x, -y, 0]))+t1g_b_dot)
                if target_i == 10:
                    target_i = 8
                    rect.set_color(BLUE)
                    text = Ftext(
                        'm[%d] = 9;'%(x_i+y_i*9),
                        t2c={'[0:1]': '#9cdcfe', '[2:4]': '#b5cea8', '[-2:-1]': '#b5cea8', '[-6:-5]': WHITE},
                    ).set_height(0.35).move_to(t3_func[-1].get_left()+0.4*DOWN, LEFT)
                else:
                    text = Ftext(
                        'tmp(%d, %d)=>%s'%(x, y, 'OK' if auOK else 'Null'),
                        t2c={'[:3]': '#dcdcaa', 'OK': GREEN, 'Null': RED}
                    ).set_height(0.35).move_to(t3_func[-1].get_left()+0.4*DOWN, LEFT)
                t3_rect_s.move_to(rect)
                anim = [FadeIn(rect)]
                if auOK:
                    target.text = '%d'%(target_i+1)
                    anim += [target.animate.re_text(target.text).set_color(t3_num_color[target_i+1])]
                if 15 < len(t3_func):
                    text.shift(0.4*UP)
                    anim += [FadeOut(t3_func[0], 0.5*LEFT)]
                    anim += [FadeOut(t3_rect[0])]
                    t3_rect.remove(t3_rect[0])
                    t3_func.remove(t3_func[0])
                    anim += [i.animate.shift(0.4*UP) for i in t3_func]
                anim += [FadeIn(text, 0.4*RIGHT)]
                t3_rect.add(rect)
                t3_func.add(text)
                self.play(*anim, **var)
        self.play(
            AnimationGroup(*[
                AnimationGroup(FadeOut(i, 0.5*LEFT), FadeOut(j))
                for i, j in zip(t2_func, t2_rect)
            ], lag_ratio=1),
            Write(t3_func[0]),
            TransformMatchingShapes(t2_code, t3_code),
            t1_text_r.animate.move_to(t1_r_pos[1]),
            ShowCreation(t3_num, lag_ratio=0.1),
            Transform(s_text, t_text[3]),
            run_time=2,
        )
        self.wait(1)
        self.add(t3_rect_s)
        t3_tmp_func(0, 4, run_time=0.5)
        turn_animation_into_updater(Transform(s_text, t_text[4]))
        t3_tmp_func(1, 4, run_time=0.3)
        t3_tmp_func(3, 5, run_time=0.1)
        t3_tmp_func(0, 3, run_time=0.1)
        t3_tmp_func(2, 5, run_time=0.2)
        t3_tmp_func(2, 4, run_time=0.3)
        self.play(
            Write(t_text[5][:14], remover=True),
            FadeOut(s_text),
        )
        self.add(s_text.become(t_text[5])[:14])
        self.wait()
        t4_func = VGroup(
            Text(
                '地雷坐标=>行|无效', font='思源黑体',
                t2c={'地雷坐标': '#b5cea8', '行': GREEN, '无效': RED},
            ).set_height(0.35).move_to(t3_func[0].get_left(), LEFT),
            Ftext(
                'temp(4, 2)=>5|3',
                t2c={'[:4]': '#dcdcaa', '4': '#b5cea8', '2': '#b5cea8', '5': GREEN, '3': RED}
            ).set_height(0.35).move_to(t3_func[1].get_left(), LEFT)
        )
        t4_rect = VGroup(Dot(radius=0), t3_rect_s.copy().scale(3, about_edge=DR).set_opacity(0.2))
        t4_r2 = Rectangle(color=RED, stroke_width=5).surround(t3_func[7:], stretch=True, buff=0.2)
        def t4_tmp_func(x_i, y_i):
            a_out = [
                (x+1) % 10 and (y+1) % 10 and int(t3_num[x+y*9].text) != 9
                for x, y in [[i+x_i, j+y_i] for i, j in around_d]
            ]
            rect = Square(
                5/3, color=BLUE, fill_opacity=0.2,
            ).shift(5/9*(np.array([x_i, -y_i, 0]))+t1g_b_dot).rotate(PI/2)
            text = Ftext(
                'temp(%d, %d)=>%d|%d' % (x_i, y_i, a_out.count(True), a_out.count(False)),
                t2c={'[:4]': '#dcdcaa', '[5:6]': '#b5cea8', '[8:9]': '#b5cea8', '[12:13]': GREEN, '[14:]': RED},
            ).set_height(0.35).move_to(t4_func[-1].get_left()+0.4*DOWN, LEFT)
            t3_num[x_i+y_i*9].text = '9'
            anim = [FadeIn(rect), t3_num[x_i+y_i*9].animate.re_text('9').set_color(BLACK)]
            for x, y in [[i+x_i, j+y_i] for i, j in around_d]:
                if (x+1) % 10 and (y+1) % 10:
                    target = t3_num[x+y*9]
                    target_i = int(target.text)
                    if target_i != 9:
                        target.text = '%d'%(target_i+1)
                        anim += [target.animate.re_text(target.text).set_color(t3_num_color[target_i+1])]
            if 15 < len(t4_func):
                text.shift(0.4*UP)
                anim += [FadeOut(t4_func[0], 0.5*LEFT)]
                anim += [FadeOut(t4_rect[0])]
                t4_rect.remove(t4_rect[0])
                t4_func.remove(t4_func[0])
                anim += [i.animate.shift(0.4*UP) for i in t4_func]
            anim += [FadeIn(text, 0.4*RIGHT)]
            t4_rect.add(rect)
            t4_func.add(text)
            return AnimationGroup(*anim)
        self.play(
            ReplacementTransform(t3_rect_s, t4_rect[1]),
            ShowCreation(t4_r2),
            Write(s_text[14:]),
            run_time=1,
        )
        self.play(
            AnimationGroup(*[
                AnimationGroup(FadeOut(i, 0.5*LEFT), FadeOut(j))
                for i, j in zip(t3_func[:7], t3_rect[:7])
            ]),
            run_time=1, lag_ratio=1,
        )
        self.play(
            Write(t4_func[0]),
            TransformMatchingShapes(t3_func[7:].copy(), t4_func[1]),
            run_time=2,
        )
        self.play(
            AnimationGroup(*[
                AnimationGroup(FadeOut(i, 0.5*LEFT), FadeOut(j))
                for i, j in zip(t3_func[7:], t3_rect[7:])
            ], lag_ratio=1),
            t4_tmp_func(4, 4),
            Uncreate(t4_r2),
            run_time=2,
        )
        t4_total = 15
        t4_t_target = t_text[6][:10]
        while t4_total:
            i = random.randint(0, 80)
            if int(t3_num[i].text) != 9:
                if t4_total == 2:
                    self.add(s_text, t4_t_target)
                    turn_animation_into_updater(s_text.animate.fade(1).build())
                    turn_animation_into_updater(Write(t4_t_target))
                self.play(
                    t4_tmp_func(i % 9, i//9),
                    run_time=max(0.2, 1-there_and_back(t4_total/15)),
                )
                t4_total -= 1
        self.add(s_text.become(t_text[6])[:10]).remove(t4_t_target)
        t6_code = Ftext('void set(int x, int y) {...}').scale(0.5).to_edge(UL, buff=0.3)
        c_style_set_color(t6_code, t6_code.text)
        t8_func = VGroup(Ftext(
            'set(-2,-2)', t2c={'set': '#dcdcaa', '2': '#b5cea8'}
        ).set_height(0.35).move_to(t4_func[0].get_left(), LEFT))
        t6_rect = Square(5/3, color=RED, fill_opacity=0.2).shift(5/9*(np.array([-2, 2, 0]))+t1g_b_dot)
        t6_num = [t3_num]
        t6_n_v = []
        def t6_set_func(x, y):
            vm = 81*[0]
            for c in range(15):
                i = random.randint(0, 80)
                while vm[i] == 9 or (abs(x-i % 9) < 2 and abs(y-i//9) < 2):
                    i = random.randint(0, 80)
                vm[i] = 9
                for m, n in [[i % 9+j, i//9+k] for j, k in around_d]:
                    if 0 <= m and m < 9 and 0 <= n and n < 9:
                        k = m+n*9
                        vm[k] += 1 if vm[k] < 9 else 0
            num = t1_grid.add_number(size=1, init=vm)
            t6_n_v.clear()
            t6_n_v.extend(vm)
            for i, j in zip(num, vm):
                i.set_color(t3_num_color[j])
            text = Ftext(
                'set(%d,%d)' % (x, y), t2c={'set': '#dcdcaa', '[4:5]': '#b5cea8', '[6:7]': '#b5cea8'}
            ).set_height(0.35).move_to(t8_func[-1].get_left()+0.4*DOWN, LEFT)
            anim = [
                t6_rect.animate.move_to(5/9*(np.array([x, -y, 0]))+t1g_b_dot),
                ApplyMethod(t8_func[-1].set_color, GREY),
                TransformMatchingShapes(t6_num[0], num),
            ]
            if 15 < len(t8_func):
                text.shift(0.4*UP)
                anim += [FadeOut(t8_func[0], 0.5*LEFT)]
                t8_func.remove(t8_func[0])
                anim += [i.animate.shift(0.4*UP) for i in t8_func]
                anim[-1].set_color(GREY)
            else:
                anim += [t8_func[-1].animate.set_color(GREY)]
            anim += [FadeIn(text, 0.4*RIGHT)]
            t6_num[0] = num
            t8_func.add(text)
            return AnimationGroup(*anim)
        self.play(
            TransformMatchingShapes(t4_func, t8_func[0]),
            TransformMatchingShapes(t3_code, t6_code),
            t1_text_r.animate.move_to(t1_r_pos[3]),
            FadeOut(t4_rect, lag_ratio=0.2),
            Write(s_text[10:]),
            FadeIn(t6_rect),
            run_time=2,
        )
        self.add(s_text)
        self.wait()
        for i in range(7):
            if i == 2:
                turn_animation_into_updater(Transform(s_text, t_text[7]))
            i = int(random.random()*81)
            self.play(t6_set_func(i % 9, i//9), run_time=1)
            self.wait(0.5)
            last_pos = i
        t8_code = Ftext('void dig(int v, int l) {...}').scale(0.5).to_edge(UL, buff=0.3)
        c_style_set_color(t8_code, t8_code.text)
        t8_rect = t6_rect.copy().scale(1/3)
        t8_func = t8_func
        t8_dig_buf = []
        t8_m_v =[1]*81
        def t8_dig_r(x, y):
            ol = []
            t8_dig_buf.clear()
            def dig(x, y):
                i = x+y*9
                if t8_m_v[i]:
                    t8_dig_buf.append(i)
                    t8_m_v[i] = 0
                    if t6_n_v[i] == 0:
                        ol.append(i)
                        for m, n in [[x+i, y+j] for i, j in around_d]:
                            if 0 <= m and m < 9 and 0 <= n and n < 9:
                                dig(m, n)
            dig(x, y)
            rect = VGroup(*[
                t1_grid.box[i].copy().set_fill(GREEN, 0.2).set_stroke(GREEN).scale(3)
                for i in ol
            ])
            return rect
        t8_rect_o = t8_dig_r(last_pos % 9, last_pos//9)
        def t8_dig_func_t(x, y):
            text = Ftext(
                'dig(%d,%d)' % (x, y), t2c={'dig': '#dcdcaa', '[4:5]': '#b5cea8', '[6:7]': '#b5cea8'}
            ).set_height(0.35).move_to(t8_func[-1].get_left()+0.4*DOWN, LEFT)
            anim = []
            if 15 < len(t8_func):
                text.shift(0.4*UP)
                anim += [FadeOut(t8_func[0], 0.5*LEFT)]
                t8_func.remove(t8_func[0])
                anim += [i.animate.shift(0.4*UP) for i in t8_func]
            anim += [FadeIn(text, 0.4*RIGHT)]
            t8_func.add(text)
            return AnimationGroup(*anim)
        t8_mask_box = VGroup(*[
            t1_grid.box[i].copy().set_fill(WHITE, 1).set_stroke(width=0).scale(0.8)
            if m == 1 else Dot(radius=0)
            for i, m in enumerate(t8_m_v)
        ])
        self.play(
            TransformMatchingShapes(t6_code, t8_code),
            t1_text_r.animate.move_to(t1_r_pos[2]),
            Transform(s_text, t_text[8]),
            run_time=2,
        )
        self.play(
            t8_dig_func_t(last_pos % 9, last_pos//9),
            ReplacementTransform(t6_rect, t8_rect),
            FadeIn(t8_rect_o, lag_ratio=0.9),
            run_time=2,
        )
        self.play(
            FadeIn(t8_mask_box, lag_ratio=0.1),
            FadeOut(t8_rect_o, lag_ratio=0.9),
            run_time=2,
        )
        t8_rect_o.remove(*t8_rect_o)
        not_dig = [i for i in range(81) if t8_m_v[i] == 1 and t6_n_v[i] < 9]
        not_dig.sort(key=lambda i: t6_n_v[i])
        self.add(t8_mask_box, t8_rect)
        anim_c = 0
        while len(not_dig):
            i = not_dig[0]
            x, y = i % 9, i//9
            l_anim = FadeOut(t8_rect_o)
            t8_rect_o = t8_dig_r(x, y)
            for i in t8_dig_buf:
                not_dig.remove(i)
            self.play(
                l_anim,
                FadeIn(t8_rect_o),
                t8_dig_func_t(x, y),
                *[t8_mask_box[i].animate.fade(1) for i in t8_dig_buf],
                t8_rect.animate.move_to(5/9*(np.array([x, -y, 0]))+t1g_b_dot),
            )
            anim_c += 1
            if anim_c == 6:
                turn_animation_into_updater(Transform(s_text, t_text[9]))
        self.play(
            Transform(s_text, t_text[10]),
            *[FadeOutRandom(i) for i in [t1_text, t1_axis, t1_grid.line, t6_num[0], t8_mask_box, t8_func, t8_code, t1_text_r, t8_rect]],
            run_time=2,
        )
        self.wait()


class Scene_11(Scene):
    def construct(self):
        around_d = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        text_list = [
            '如何调用这些函数呢, 先写点简单的代码',
            '按正常思路, 用flag来判断是否第一次执行',
            '确实挺直观, 但每轮循环都会进行判断',
            '虽然开销不大, 但的确不是理想的方案',
            '前面提到的函数指针, 在这里就发挥了用处',
            '不仅解决了问题, 还明显节省了一点代码量',
            '然后, 再演示一遍set()和dig()的效果...',
            'set()以地雷为中心, 遍历周围格子执行tmp()',
            'dig()以输入坐标为中心, 如果对应位置是空地',
            '则遍历周围执行dig(), 还存在空地则以此类推',
            '有没有发现两个函数之间的共同点?',
            '"以某点为中心, 遍历周围执行特定操作"',
            '简单的复现一下这种算法(遍历周围点)',
            '看起来貌似很简单? 不过这个代码量令人头疼',
            '简写一下, 带到伪代码中, 要想办法压缩这部分',
            '结合函数指针, 复用重复部分, 再叠一层指针',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).scale(0.9).fix_in_frame()
        s_text = t_text[0].copy()
        t1_code = [
            'void dig(x, y) {...}',
            'void set(x, y) {...}',
            'int main() {',
            '    ...',
            '}',
        ]
        cli = CommandLine()
        for i in t1_code:
            cli.add_text(i)
        cli.set_color_c_style()
        mt = cli.m_text
        uw = cli.unit_w
        uh = cli.unit_h
        frame = self.camera.frame
        frame.scale(0.5, about_edge=UL)
        bg_rect = Rectangle(
            FRAME_WIDTH+0.5, FRAME_HEIGHT, stroke_width=0,
            fill_opacity=1, fill_color='#1e1e1e',
        )

        self.add(bg_rect, s_text)
        self.play(Write(cli.m_text), FadeOut(bg_rect), run_time=2)
        self.wait()
        t1_u1 = mt[1][16:].copy()
        cli.new_text('void set(x, y) {flag = 0; ...}', 1)
        cli.new_text('    flag = 1;', 3, insert=True)
        cli.set_color_c_style(1)
        cli.set_color_c_style(3)
        self.remove(*mt[1][16:])
        self.play(
            Write(mt[3]),
            Write(mt[1][16:25]),
            Transform(s_text, t_text[1]),
            ReplacementTransform(t1_u1, mt[1][26:]),
            mt[4:].shift(uh*UP).animate.shift(uh*DOWN),
            run_time=1.5,
        )
        for i in ['    ...', '    dig(x, y);', '    if (flag) set(x, y);']:
            cli.new_text(i, 5, insert=True)
        cli.set_color_c_style(5)
        cli.set_color_c_style(6)
        self.play(
            Write(mt[5:7]),
            TransformFromCopy(mt[4], mt[7]),
            mt[8].shift(3*uh*UP).animate.shift(3*uh*DOWN),
            run_time=1.5,
        )
        rect = Rectangle(9*uw, uh, color=RED).move_to(cli.get_grid_place(8, 5)).scale(1.1)
        self.play(RouteCreation(rect, lambda i: (rush_from(i)*1.2, max(rush_into(i)*0.2, 5.2*i-4))))
        rect = Rectangle(
            30*uw, uh, color=GREEN, fill_opacity=1, stroke_width=1,
        ).move_to(cli.get_grid_place(14.5, 4))
        self.add(rect, mt)
        self.play(Transform(s_text, t_text[2]), FadeIn(rect))
        turn_animation_into_updater(
            UpdateFromAlphaFunc(rect, lambda i, t: i.fade(t), run_time=7)
        )
        for i in range(20):
            if i == 9:
                turn_animation_into_updater(Transform(s_text, t_text[3]), run_time=0.9)
            for i in range(3):
                rect.shift(uh*DOWN)
                self.wait(0.1)
            rect.shift(3*uh*UP)
            self.wait(0.1)
        t5_u1 = mt[1][16:].copy()
        t5_u2 = mt[3].copy()
        t5_u3 = mt[5].copy()
        cli.code_dict['f'] = '#9cdcfe'
        cli.new_text('void set(x, y) {f = dig; ...; f(x, y);}', 1)
        cli.new_text('    f = set;', 3)
        cli.new_text('    f(x, y);', 5)
        for i in [1, 3, 5]:
            cli.set_color_c_style(i)
        self.remove(*mt[1][16:], *mt[3], *mt[5])
        self.play(
            t5_u1[8:].animate.shift(uw*LEFT),
            ReplacementTransform(t5_u1[:8], mt[1][16:23:]),
            ReplacementTransform(t5_u2, mt[3]),
            FadeOut(t5_u3[:14], LEFT),
            ReplacementTransform(t5_u3[14:17], mt[5][4]),
            ReplacementTransform(t5_u3[17:], mt[5][5:]),
            Transform(s_text, t_text[4]),
            run_time=1.5,
        )
        self.play(
            t5_u1[13].animate.shift(10*uw*RIGHT),
            mt[7:].animate.shift(uh*UP),
            TransformMatchingShapes(mt[6], mt[1][28:38]),
            run_time=1.5,
        )
        mt.remove(mt[6])
        cli.text.remove(cli.text[6])
        turn_animation_into_updater(
            UpdateFromAlphaFunc(rect, lambda i, t: i.fade(t), run_time=5)
        )
        for i in range(20):
            if i == 9:
                turn_animation_into_updater(Transform(s_text, t_text[5]), run_time=0.9)
            for i in range(2):
                rect.shift(uh*DOWN)
                self.wait(0.1)
            rect.shift(2*uh*UP)
            self.wait(0.1)
        self.remove(rect, *t5_u1)
        t7_grid1 = MyGrid(9, 9).next_to(ORIGIN, LEFT, 0.2)
        t7_grid2 = MyGrid(9, 9).next_to(ORIGIN, RIGHT, 0.2)
        t7_v_map = 81*[0]
        t7_v_mask = 81*[1]
        t7_dig_buf = []
        t7_buf_bomb = []
        t7_dig_buf_air = []
        for c in range(15):
            i = random.randint(0, 80)
            while t7_v_map[i] == 9 or (abs(4-i % 9) < 2 and abs(4-i//9) < 2):
                i = random.randint(0, 80)
            t7_v_map[i] = 9
            t7_buf_bomb.append(i)
            for m, n in [[i % 9+j, i//9+k] for j, k in around_d]:
                if 0 <= m and m < 9 and 0 <= n and n < 9:
                    k = m+n*9
                    t7_v_map[k] += 1 if t7_v_map[k] < 9 else 0
        def t7_dig(x, y):
            i = x+y*9
            if t7_v_mask[i]:
                t7_dig_buf.append(i)
                t7_v_mask[i] = 0
                if t7_v_map[i] == 0:
                    t7_dig_buf_air.append(i)
                    for m, n in [[x+i, y+j] for i, j in around_d]:
                        if 0 <= m and m < 9 and 0 <= n and n < 9:
                            t7_dig(m, n)
        t7_dig(4, 4)
        t7_num_color = [WHITE, GREEN, BLUE, RED, YELLOW, GREY_BROWN, PURPLE, PINK, LIGHT_BROWN, BLACK]
        t7_num_s = t7_grid1.add_number(size=1, init=lambda i: 0)
        t7_num1 = t7_grid1.add_number(size=1, init=t7_v_map)
        for i, j in zip(t7_num1, t7_v_map):
            i.set_color(t7_num_color[j])
        t7_num2 = t7_num1.copy().move_to(t7_grid2)
        t7_box2 = t7_grid2.box.copy().set_color(WHITE)
        for i in t7_box2:
            i.scale(0.8)
        t7_r1 = VGroup(*[
            t7_grid1.box[i].copy().set_fill(BLUE, 0.2).set_stroke(BLUE).scale(3)
            for i in t7_buf_bomb
        ])
        t7_r2 = VGroup(*[
            t7_grid2.box[i].copy().set_fill(GREEN, 0.2).set_stroke(GREEN).scale(3)
            for i in t7_dig_buf_air
        ])
        
        self.play(
            Transform(s_text, t_text[6]),
            FadeIn(t7_num_s, lag_ratio=0.1),
            FadeIn(t7_box2, lag_ratio=0.1),
            mt.animate.scale(0.8, about_edge=UL),
            frame.animate.scale(2, about_edge=UL),
            ShowCreation(t7_grid1.line, lag_ratio=0.1),
            ShowCreation(t7_grid2.line, lag_ratio=0.1),
            run_time=2,
        )
        self.play(
            FadeIn(t7_r1, lag_ratio=0.1),
            FadeIn(t7_r2, lag_ratio=0.1),
            run_time=1.5,
        )
        self.add(t7_num2, *t7_box2, *t7_r2, *t7_num_s, t7_r1, s_text)
        self.play(
            *[Transform(i, j) for i, j in zip(t7_num_s, t7_num1)],
            *[t7_box2[i].animate.fade(1) for i in t7_dig_buf],
            run_time=1.5,
        )
        self.play(
            frame.animate.scale(0.9).move_to(t7_grid1),
            Transform(s_text, t_text[7]),
            run_time=1.5,
        )
        self.play(*[i.animate.scale(1/3).set_stroke(width=2) for i in t7_r1], run_time=1.5)
        self.play(*[FadeOut(i.copy(), scale=3, remover=True) for i in t7_r1], run_time=1.5)
        self.wait()
        t9_rect = Square(5/9, color=RED).move_to(t7_grid2)
        self.play(
            *[t7_box2[i].animate.fade(0) for i in t7_dig_buf],
            frame.animate.move_to(t7_grid2),
            Transform(s_text, t_text[8]),
            ShowCreation(t9_rect),
            FadeOut(t7_r2),
            run_time=1.5,
        )
        self.play(t7_box2[40].animate.fade(1), run_time=0.5)
        self.play(Rotate(t9_rect, PI/2), rate_func=slow_into, run_time=1)
        t10_text = t_text[9].copy().scale(0.8).to_edge(DOWN, buff=0.1)
        t7_r2.set_fill(opacity=0.2).set_stroke(GREEN, 2, 1)
        turn_animation_into_updater(s_text.animate.scale(0.8).to_edge(DOWN, buff=0.7).build())
        self.wait(0.5)
        self.add(t10_text)
        turn_animation_into_updater(Write(t10_text, run_time=1))
        turn_animation_into_updater(t9_rect.animate.fade(1).build(), run_time=4)
        for i, j in zip(t7_r2, t7_dig_buf_air):
            x_i, y_i = j % 9, j // 9
            a_anim = [
                t7_box2[x+y*9].animate.fade(1) for x, y in
                [[x_i+m, y_i+n] for m, n in around_d]
                if 0 <= x and x < 9 and 0 <= y and y < 9
            ]
            self.play(*a_anim, FadeIn(i, scale=3), run_time=4/len(t7_r2))
            self.wait(2/len(t7_r2))
        self.wait(0.5)
        self.play(
            *[i.animate.scale(3).set_stroke(width=0) for i in t7_r1],
            *[i.animate.set_stroke(width=0) for i in t7_r2],
            frame.animate.scale(10/9).center(),
            Uncreate(t10_text, lag_ratio=0.2),
            Transform(s_text, t_text[10]),
            run_time=2,
        )
        self.wait(2)
        t12_text = t_text[11].copy().scale(0.8).to_edge(DOWN, buff=0.1).unfix_from_frame()
        self.play(
            s_text.animate.scale(0.8).to_edge(DOWN, buff=0.7),
            Write(t12_text),
            run_time=1.5,
        )
        self.play(
            *[i.animate.scale(1/3).set_stroke(width=2) for i in t7_r1],
            *[i.animate.scale(1/3).set_stroke(width=2) for i in t7_r2],
            run_time=1.5,
        )
        self.play(
            *[FadeOut(i.copy(), scale=3, remover=True) for i in t7_r1],
            *[FadeOut(i.copy(), scale=3, remover=True) for i in t7_r2],
            run_time=2, rate_func=slow_into,
        )
        self.wait(0.5)
        t11_code_t = 'int i, pos[8][2] = [-1, 0, 1, 0, 0, 1...];\nfor (i = 0; i < 8; ++i) {\n    func(x+pos[i][0], y+pos[i][1]);\n}\n'
        t11_code = cli.new_word(t11_code_t).scale(1.2).next_to(t10_text, DOWN, buff=1).to_edge(LEFT, buff=0.1)
        t11_text = VGroup(
            *[Text(i, font='思源黑体') for i in ['//八个点的坐标偏移量', '//遍历', '//执行操作']]
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).match_height(t11_code[:-2]).next_to(t11_code[:-2], buff=0.1).set_color('#6a993e')
        c_style_set_color(t11_code, t11_code_t)
        self.play(
            Write(t11_code),
            frame.animate.shift(BOTTOM),
            Transform(s_text, t_text[12]),
            run_time=2,
        )
        self.play(*[Write(i, run_time=1) for i in t11_text])
        self.wait(2)
        turn_animation_into_updater(Transform(s_text, t_text[13]))
        self.play(AnimationGroup(
            *[i.animate.set_color(RED).scale(1.2).build().set_rate_func(there_and_back) for i in t11_code],
            lag_ratio=0.05, run_time=4,
        ))
        self.wait(0.5)
        t15_cli_new = [
            'void dig(x, y) {',
            '    ...',
            '    for(around()) dig(x, y);',
            '}',
            'void set(x, y) {',
            '    f = dig;',
            '    ...',
            '    for(around()) tmp(x, y);',
            '    ...',
            '}',
            'int main() {...}'
        ]
        for i, m in enumerate(t15_cli_new):
            cli.new_text(m, i)
        cli.set_color_c_style()
        t15_code = cli.new_word(
            'for(around()) func();'
        ).scale(1.5).align_to(t11_code, UL)
        c_style_set_color(t15_code, 'for(around()) func();')
        turn_animation_into_updater(Transform(s_text, t_text[14]))
        turn_animation_into_updater(Uncreate(t11_text, lag_ratio=0, run_time=2))
        self.play(TransformMatchingShapes(t11_code, t15_code, run_time=2))
        self.wait(0.5)
        self.remove(mt[2], mt[7])
        self.play(
            *[FadeOutRandom(i) for i in [t7_r1, t7_r2, t7_grid1.line, t7_grid2.line, t7_num_s, t7_num2, t7_box2]],
            frame.animate.center().scale(0.75, about_edge=UL),
            TransformFromCopy(t15_code[:13], mt[2][4:17]),
            ReplacementTransform(t15_code[:13], mt[7][4:17]),
            TransformFromCopy(t15_code[13:], mt[2][18:]),
            ReplacementTransform(t15_code[13:], mt[7][18:]),
            run_time=2.5,
        )
        self.wait(0.5)
        self.remove(mt[7])
        t16_u1 = mt[7].copy()
        cli.new_text('    dig(x, y);', 7)
        cli.set_color_c_style(7)
        turn_animation_into_updater(Transform(s_text, t_text[15]))
        self.play(
            FadeOut(t16_u1, 5*uh*UP),
            Write(mt[7]),
            run_time=2,
        )
        self.remove(mt[2], mt[9])
        t16_u2 = mt[9].copy()
        t16_u3 = mt[5].copy()
        t16_u4 = mt[2].copy()
        cli.new_text('    f = dig;', 9)
        cli.set_color_c_style(9)
        cli.new_text('    f = tmp;', 5)
        cli.set_color_c_style(5)
        cli.new_text('    for(around()) f(x, y);', 2)
        cli.set_color_c_style(2)
        self.add(t16_u4)
        self.play(
            Write(mt[5]),
            t16_u2.animate.shift(uh*DOWN),
            mt[10].animate.shift(uh*DOWN),
            ReplacementTransform(t16_u3, mt[9]),
            ReplacementTransform(t16_u4[18:], mt[2][18:]),
            run_time=2,
        )
        self.add(*mt)
        print(self.num_plays)
        self.wait()


class Scene_12(Scene):
    def construct(self):
        around_d = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        num_color = [WHITE, GREEN, BLUE, RED, YELLOW, GREY_BROWN, PURPLE, PINK, LIGHT_BROWN, BLACK]
        def rect_flash(i): return (rush_from(i)*1.2, max(rush_into(i)*0.2, 5.2*i-4))
        text_list = [
            '结合函数指针, 复用重复部分, 再叠一层指针',
            '接下来就是处理剩下的细节了...',
            '模拟运行一遍...是不是感觉有点不对劲?',
            '不对劲就对了, 某些地雷周围的数字明显不符合规则',
            '回到bug发生前的状态...',
            '在dig()里, 对应位置是空地才会执行后面的操作',
            '如果在数字上执行, 就会提前返回导致生成错误',
            '防止提前返回, 调用dig()前将对应值归零就行了',
            '除此之外, dig()还更改了M和s, 同样需要复位',
            '整理好思路后, 压缩代码也就轻而易举了',
        ]
        t1_code1 = [
            'void dig(x, y) {',
            '    ...',
            '    for(around()) f(x, y);',
            '}',
            'void set(x, y) {',
            '    f = tmp;',
            '    ...',
            '    dig(x, y);',
            '    ...',
            '    f = dig;',
            '}',
            'int main() {...}',
        ]
        t1_code2 = [
            'void dig(x, y) {',
            '    if (!edge(x, y) and !M[x + y * W]) {',
            '        M[x + y * W] = 1, --s;',
            '        if (m[x + y * W] == 0) {',
            '            int i, P[8][2] = [-1, 0...];',
            '            for (i = 0; i < 8; ++i)',
            '                f(x+P[i][0], y+P[i][1]);',
            '        }',
            '    }',
            '}',
            'void set(x, y) {',
            '    for (f = tmp, c = 0; c < B;) {',
            '        i = rand() % S;',
            '        if (m[i] < 9 and temp()) {',
            '            dig(i % W, i / W);',
            '            m[i] = 9, ++c;',
            '        }',
            '    }',
            '    f = dig, f(x, y);',
            '}',
        ]
        t1_code3 = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        cli.code_dict['and'] = '#569cd6'
        cli.code_dict['tmp'] = '#dcdcaa'
        cli.code_dict['f'] = '#9cdcfe'
        for i in t1_code1:
            cli.add_text(i)
        cli.set_color_c_style()
        old_mt = cli.m_text.copy()
        cli.text.clear()
        cli.m_text.remove(*cli.m_text)
        for i in t1_code2:
            cli.add_text(i)
        cli.set_color_c_style()
        mt = cli.m_text
        uw = cli.unit_w
        uh = cli.unit_h
        frame = self.camera.frame
        frame.scale(0.75, about_edge=UL)
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).scale(0.9).fix_in_frame()
        s_text = t_text[0].copy()
        t1_text_t = [
            'dig() 挖掘动作',
            '是否越界, 是否挖过',
            '更新指定的变量',
            '如果挖开的是空地...',
            '遍历周围执行特定操作',
            'set() 生成地雷',
            '生成指定数量的地雷',
            '地雷生成位置随机',
            '输入坐标周围不生成雷',
            '地雷附近数值累加',
            '输入位置执行dig()',
        ]
        t1_text_ct = [
            'if (!edge(x, y) and !M[x + y * W]) {}',
            'M[x + y * W] = 1, --s;',
            'if (m[x + y * W] == 0) {}',
            'int i, P[8][2] = [-1, 0...];\nfor (i = 0; i < 8; ++i)\n    f(x+P[i][0], y+P[i][1]);',
            'for (c = 0; c < B; ++c) {}',
            'i = rand() % S;',
            'if (m[i] < 9 and temp()) {}',
            'm[i] = 9, dig(i % W, i / W);',
            'f(x, y);',
        ]
        t1_text = VGroup(*[
            i.add(Circle(color=BLUE, fill_opacity=1, stroke_width=0).scale(0.15).next_to(i, LEFT)).scale(0.5)
            for i in [Text(i, font='思源黑体') for i in t1_text_t]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(old_mt, aligned_edge=UP)
        t1_text_c = VGroup(*[Ftext(i) for i in t1_text_ct])
        for i, j, k in zip([*t1_text[1:5], *t1_text[6:]], t1_text_c, t1_text_ct):
            c_style_set_color(j, k, cli.code_dict)
            i.scale(0.8, about_edge=LEFT).shift(RIGHT*0.2)
            j.scale(0.4).move_to(np.array([FRAME_WIDTH*0.1, i.get_top()[1], 0]), UL)
        self.add(old_mt, s_text)
        self.wait()
        self.play(
            Transform(s_text, t_text[1]),
        )
        self.play(*[Write(i) for i in t1_text], run_time=2)
        self.wait()
        self.play(
            *[Write(i) for i in t1_text_c],
            frame.animate.set_x(2),
            run_time=2,
        )
        self.wait()
        turn_animation_into_updater(frame.animate.center().scale(4/3).build().set_run_time(2))
        for i in t1_text:
            turn_animation_into_updater(Uncreate(i, lag_ratio=0.1, run_time=1))
        self.wait(0.5)
        turn_animation_into_updater(s_text.animate.fade(1).shift(DOWN).build())
        self.play(
            *[FadeOut(old_mt[i], 4*uw*LEFT) for i in [1, 2, 6, 7, 8, 11]],
            *[ReplacementTransform(old_mt[i], mt[j]) for i, j in [[0, 0], [3, 9], [4, 10], [10, 19]]],
            Transform(old_mt[5][4:], mt[11][9:17]),
            t1_text_c[0][:36].animate.replace(mt[1][4:]),
            t1_text_c[0][36:].animate.replace(mt[8][4:]),
            t1_text_c[1].animate.replace(mt[2][8:]),
            t1_text_c[2][:24].animate.replace(mt[3][8:]),
            t1_text_c[2][24:].animate.replace(mt[7][8:]),
            t1_text_c[3][:28].animate.replace(mt[4][12:]),
            t1_text_c[3][29:52].animate.replace(mt[5][12:]),
            t1_text_c[3][57:].animate.replace(mt[6][16:]),
            t1_text_c[4][:5].animate.replace(mt[11][4:9]),
            t1_text_c[4][5:18].animate.replace(mt[11][18:31]),
            t1_text_c[4][22:25].animate.replace(mt[11][31:]),
            t1_text_c[4][25:].animate.replace(mt[17][4:]),
            t1_text_c[5].animate.replace(mt[12][8:]),
            t1_text_c[6][:26].animate.replace(mt[13][8:]),
            t1_text_c[6][26:].animate.replace(mt[16][8:]),
            t1_text_c[7][10:].animate.replace(mt[14][12:]),
            t1_text_c[7][:10].animate.replace(mt[15][12:22]),
            TransformMatchingShapes(t1_text_c[4][18:22].add(t1_text_c[4][17].copy()), mt[15][21:]),
            TransformMatchingShapes(VGroup(t1_text_c[8], old_mt[9]), mt[18][4:]),
            run_time=4,
        )
        self.remove(*t1_text_c, *old_mt).add(mt)
        t3_grid = MyGrid(9, 9, 5).to_edge(UR)
        t3_num = t3_grid.add_number(size=1, init=lambda i: 0)
        t3_v_map = [0]*81
        grid_b_dot = t3_grid.box[0].get_center()
        t3_rect_bug = VGroup()
        def set_func_bug(x_i, y_i):
            rect = Square(5/9, color=BLUE).shift(5/9*(np.array([x_i, -y_i, 0]))+grid_b_dot)
            anim = [
                RouteCreation(rect, rect_flash, remover=True),
                t3_num[x_i+y_i*9].animate.re_text('9').set_color(BLACK),
            ]
            if t3_v_map[x_i+y_i*9] == 0:
                for x, y in [[i+x_i, j+y_i] for i, j in around_d]:
                    if (x+1) % 10 and (y+1) % 10:
                        num = t3_v_map[x+y*9]
                        if num != 9:
                            num += 1
                            t3_v_map[x+y*9] += 1
                            anim += [t3_num[x+y*9].animate.re_text(str(num)).set_color(num_color[num])]
            else:
                t3_rect_bug.add(rect.copy().set_color(RED))
                for x, y in [[i+x_i, j+y_i] for i, j in around_d]:
                    if (x+1) % 10 and (y+1) % 10:
                        if t3_v_map[x+y*9] != 9:
                            t3_rect_bug.add(
                                Square(5/9, color=RED, fill_opacity=0.2).shift(5/9*(np.array([x, -y, 0]))+grid_b_dot)
                            )
            t3_v_map[x_i+y_i*9] = 9
            return AnimationGroup(*anim)
        self.remove(s_text)
        self.play(
            ShowCreation(t3_grid.line, lag_ratio=0.1),
            ShowCreation(t3_num, lag_ratio=0.1),
            Write(s_text.become(t_text[2])[:10]),
            run_time=1.5,
        )
        anim_c = 0
        turn_animation_into_updater(frame.animate.scale(0.8, about_edge=UR).build())
        while anim_c < 10:
            pos = int(random.random()*81)
            if t3_v_map[pos] < 9:
                if anim_c == 4:
                    while t3_v_map[pos] % 9 == 0:
                        pos = int(random.random()*81)
                    t3_num_save = t3_num.copy()
                    t3_n_s_pos = pos
                    t3_n_s_vm = t3_v_map.copy()
                if anim_c == 5:
                    tmp = s_text[10:]
                    self.add(tmp)
                    turn_animation_into_updater(Write(tmp))
                self.play(set_func_bug(pos % 9, pos // 9), run_time=1)
                anim_c += 1
        self.add(s_text)
        turn_animation_into_updater(Transform(s_text, t_text[3]))
        self.play(
            frame.animate.scale(1.25, about_edge=UR),
            run_time=2,
        )
        self.play(AnimationGroup(
            *[RouteCreation(i, rect_flash) for i in t3_rect_bug], run_time=2, lag_ratio=0.05
        ))
        t4_r1 = Square(5/9, color=RED).move_to(t3_grid.box[t3_n_s_pos])
        t4_r2 = Rectangle(
            mt[4][12:].get_width(), mt[4:7].get_height(),
            color=RED, stroke_width=5, fill_opacity=0.2,
        ).align_to(mt[4], UR)
        t4_l1 = Underline(mt[3][8:], color=RED, stroke_width=10)
        turn_animation_into_updater(Transform(s_text, t_text[4]))
        self.play(
            *[FadeOut(i) for i in t3_rect_bug],
            *[ReplacementTransform(i, j, path_arc=PI/2) for i, j in zip(t3_num[::-1], t3_num_save)],
            run_time=2.5,
        )
        turn_animation_into_updater(Transform(s_text, t_text[5]))
        self.play(ShowCreationThenFadeOut(t4_l1), run_time=2)
        self.play(RouteCreation(t4_r2, rect_flash, False), run_time=2)
        t7_rect = Rectangle(
            41*uw, uh, color=GREEN, fill_opacity=0.5, stroke_width=1,
        ).move_to(cli.get_grid_place(20, 0))
        self.add(t7_rect, mt)
        turn_animation_into_updater(Transform(s_text, t_text[6]))
        self.play(
            FadeIn(t7_rect),
            FadeIn(t4_r1, scale=0.5, rate_func=slow_into),
            *[i.animate.set_opacity(0.2) for i in mt[4:7]],
            run_time=2,
        )
        for i in [1, 1, 1, 4, 1, 1, 0]:
            self.wait(2/7)
            t7_rect.shift(i*uh*DOWN)
        self.play(
            FadeOut(t7_rect),
            Transform(s_text, t_text[7]),
        )
        cli.new_text('            m[i] = 0;', 14, insert=True)
        cli.set_color_c_style(14)
        self.play(
            Write(mt[14]),
            mt[15:].shift(uh*UP).animate.shift(uh*DOWN),
            run_time=2,
        )
        self.play(
            FadeIn(t7_rect.shift(9*uh*UP)),
            *[i.animate.set_opacity(1) for i in mt[4:7]],
        )
        t7_wt = 3/24
        for i in range(8):
            if i == 4:
                t7_rect.shift(2*uh*DOWN)
                for x, y in [[i+t3_n_s_pos%9, j+t3_n_s_pos//9] for i, j in around_d]:
                    t7_rect.shift(uh*UP)
                    self.wait(t7_wt)
                    if (x+1) % 10 and (y+1) % 10:
                        num = t3_n_s_vm[x+y*9]
                        if num != 9:
                            num += 1
                            t3_num_save[x+y*9].re_text(str(num)).set_color(num_color[num])
                    t7_rect.shift(uh*DOWN)
                    self.wait(t7_wt)
            else:
                t7_rect.shift(uh*DOWN)
                self.wait(t7_wt)
        turn_animation_into_updater(Transform(s_text, t_text[8]))
        self.play(
            t7_rect.animate.surround(mt[2][8:], stretch=True, buff=0.3).set_color(RED),
            run_time=2,
        )
        cli.new_text('            M[i] = 0;', 17, insert=True)
        cli.set_color_c_style(14)
        cli.new_text('    s += B;', 21, insert=True)
        cli.set_color_c_style(21)
        self.play(
            Write(mt[17]),
            Write(mt[21]),
            mt[18:21].shift(uh*UP).animate.shift(uh*DOWN),
            mt[22:].shift(2*uh*UP).animate.shift(2*uh*DOWN),
            s_text[:5].animate.fade(0.9),
            run_time=2,
        )
        self.wait(0.5)
        self.play(
            FadeOut(t4_r1),
            FadeOut(t7_rect),
            FadeOutRandom(t3_num_save),
            FadeOutRandom(t3_grid.line),
            Transform(s_text, t_text[9]),
        )
        mt_tmp = mt.copy()
        self.remove(mt).add(mt_tmp)
        cli.text.clear()
        cli.m_text.remove(*cli.m_text)
        for i in t1_code3:
            cli.add_text(i)
        cli.set_color_c_style()
        mt.shift(5*uh*UP)
        self.play(TransformMatchingShapes(mt_tmp, mt[5:14]), run_time=3)
        turn_animation_into_updater(s_text.animate.fade(1).shift(DOWN).build().set_run_time(2))
        self.play(
            frame.animate.shift(5*uh*UP),
            UpdateFromAlphaFunc(mt[:5], lambda i, t: i.fade(1-t)),
            UpdateFromAlphaFunc(mt[14:], lambda i, t: i.fade(1-t)),
            run_time=3,
        )
        self.wait()


class Scene_13(Scene):
    def construct(self):
        text_list = [
            '24行已经实现了基本功能',
            '但控制台界面过于简洁, 单调',
            '所以还准备了另一个版本, 只需修改一点细节',
            '仅多3行, 使用winAPI实现, 就不细讲了',
        ]
        t_text = VGroup(*[Text(i, font='思源黑体') for i in text_list]).to_edge(DOWN).scale(0.9).fix_in_frame()
        s_text = t_text[0].copy()
        code = open(os.getcwd()+'\\code_1.c', 'r').readlines()
        cli = CommandLine()
        cli.code_dict['COORD'] = '#4ec9b0'
        cli.code_dict['DWORD'] = '#4ec9b0'
        cli.code_dict['CONSOLE_CURSOR_INFO'] = '#4ec9b0'
        for i in code:
            cli.add_text(i)
        cli.set_color_c_style()
        code_old = cli.m_text.copy()
        code = open(os.getcwd()+'\\code_5.c', 'r').readlines()
        for i, m in enumerate(code):
            cli.new_text(m, i)
        cli.set_color_c_style()
        code_new = cli.m_text
        mob = Game_MS(16, 16, 40)
        mob.set_bomb(0, 0, 4)
        mob.auto_read = True
        def map_to_str(mob, pos):
            out = ''
            for i, m in enumerate(zip(mob.v_mask, mob.v_map)):
                out += ' >'[i == pos]+('*' if m[0] else ' 12345678@'[m[1]])
                if (i+1) % mob.n_cols == 0:
                    out += '\n'
            return out.split('\n')
        con = Console(32, 17, 5).next_to(code_new, buff=0.5)
        for i in map_to_str(mob, 0):
            con.add_text(i)
        frame = self.camera.frame
        self.add(code_old, s_text)
        self.play(Write(s_text), run_time=1)
        self.wait(2)
        self.add(con)
        turn_animation_into_updater(Transform(s_text, t_text[1]))
        def con_func(m):
            if m.total % 15 == 0:
                pos = mob.get_auto()
                if pos is None:
                    print('end')
                    m.clear_updaters(con_func)
                    return
                mob.run_func(pos)
                for i, j in enumerate(map_to_str(mob, pos)):
                    con.new_text(j, i)
            m.total += 1
        con.total = 1
        con.add_updater(con_func)
        self.play(frame.animate.scale(1.4, about_edge=UL), run_time=2)
        self.wait(2)
        self.play(
            Transform(s_text, t_text[2]),
            frame.animate.scale(0.9, about_edge=UL),
        )
        write_value = {'run_time':1, 'rate_func':lambda i: max(0, 2*i-1), 'lag_ratio':0.01}
        self.play(AnimationGroup(
            Transform(code_old[1][9:], code_new[1][9:]),
            Shift(code_old[2][:71], cli.unit_h*DOWN),
            Shift(code_old[2][71], 4*cli.unit_w*RIGHT+cli.unit_h*DOWN),
            Write(code_new[2], **write_value),
            Write(code_new[3][71:], **write_value),
            *[Shift(i, cli.unit_h*DOWN) for i in code_old[3:10]],
            Shift(code_old[10][:16], cli.unit_h*DOWN),
            Shift(code_old[10][16:], 10*cli.unit_w*RIGHT+cli.unit_h*DOWN),
            *[Shift(i, cli.unit_h*DOWN) for i in code_old[11:15]],
            Shift(code_old[15][:12], 5*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            Shift(code_old[15][13:27], 2*cli.unit_w*RIGHT+cli.unit_h*DOWN),
            Shift(code_old[15][28:49], 24*cli.unit_w*LEFT+2*cli.unit_h*DOWN),
            Shift(code_old[15][50:59], 32*cli.unit_w*LEFT+3*cli.unit_h*DOWN),
            Shift(code_old[15][59:], 43*cli.unit_w*LEFT+4*cli.unit_h*UP),
            FadeOut(code_old[16][18:35], 2*cli.unit_h*DOWN),
            Shift(code_old[16][:9], 2*cli.unit_h*DOWN),
            Shift(code_old[16][9:17], 18*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            Shift(code_old[16][35:], 2*cli.unit_h*DOWN),
            Write(code_new[16][4:15], **write_value),
            Write(code_new[16][30:], **write_value),
            *[Shift(i, 2*cli.unit_h*DOWN) for i in code_old[17:19]],
            Write(code_new[17][26:], **write_value),
            Shift(code_old[19][8:13], 2*cli.unit_h*DOWN),
            Shift(code_old[19][13:19], 32*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            TransformMatchingShapes(code_old[19][20:44], code_new[23][23:45]),
            Shift(code_old[19][45:62], 7*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            FadeOut(code_old[19][63:65], 6*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            Shift(code_old[19][65:71], 5*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            FadeOut(code_old[19][71:73], 4*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            Shift(code_old[19][73:], 3*cli.unit_w*RIGHT+2*cli.unit_h*DOWN),
            FadeOut(code_old[20][12:14], 2*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            Shift(code_old[20][14:21], 2*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            TransformMatchingShapes(code_old[20][21:27], code_new[23][46:51]),
            Transform(code_old[20][27], code_new[23][52]),
            Transform(code_old[20][29:33], code_new[23][54:57]),
            TransformMatchingShapes(
                VGroup(code_old[20][33:41], code_old[20][56:74]),
                code_new[22][39:77],
            ),
            Shift(code_old[20][41:54], 16*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            FadeOut(code_old[20][54], 15*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            Shift(code_old[20][55], 15*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            Shift(code_old[20][74:], 3*cli.unit_w*RIGHT+3*cli.unit_h*DOWN),
            TransformMatchingShapes(code_old[20][64:68].copy(), code_new[23][71:77]),
            Shift(code_old[21], 3*cli.unit_h*DOWN),
            Write(code_new[21][13:44], **write_value),
            Shift(code_old[22][6:], 2*cli.unit_w*LEFT+3*cli.unit_h*DOWN),
            FadeOut(code_old[22][4:6], 2*cli.unit_w*LEFT+3*cli.unit_h*DOWN),
            Write(code_new[22][12:38], **write_value),
            Write(code_new[22][77:], **write_value),
            Shift(code_old[23], 3*cli.unit_h*DOWN),
            run_time=6, lag_ratio=0.01,
        ))
        self.play(Transform(s_text, t_text[3]))
        self.wait(4)


class Scene_14(Scene):
    def construct(self):
        self.camera.background_rgba = [*Color(BLACK).get_rgb(), 1]
        text = VGroup(*[Text(i, font='Source Han Sans K') for i in [
            'bgm:',
            'Xeuphoria - Lifeline',
            'koma\'n - Just be Friends-piano.ver-',
            'koma\'n - ダブルラリアット-piano.ver-',
            'Xeuphoria 金世正 - 꽃길 (Flower Way)\n(Xeuphoria Ver.)',
            '视频也接近尾声了',
            '如果感觉视频有趣的话',
            '就多多三连支持一下吧',
        ]]).scale(0.5)
        text[4][32:].shift(0.05*DOWN)
        text.arrange(DOWN, aligned_edge=LEFT).to_edge(UR, buff=0.2)
        text_2 = VGroup(*[Text(i, font='思源黑体') for i in [
            '按以往视频来算, 能坚持到最后的比例基本在0.1%',
            '如果你能看到这...那就发点弹幕吧',
            '比如说说自己曾经多少行实现扫雷;',
            '代码是否看懂了, 有没有学到奇怪的技巧;',
            '是否学过编程, 看完对编程有没有兴趣;',
            '视频过程中暂停了几次, 视频刷了几遍;',
            '如果还有下期视频的话...应该有很多选择了',
            '实现自动扫雷, 或是使用图形界面, 亦或是2048',
            '也许还会出manim教程?(不断挖坑)',
        ]]).scale(0.5)
        text_2[0].scale(0.9)
        for i, j in zip(text_2[2:6], [RED, GREEN, BLUE, WHITE]):
            i.add(Circle(fill_opacity=1, stroke_width=0).scale(0.05).next_to(i, LEFT)).scale(0.9).set_color(j)
        text_2.arrange(DOWN, aligned_edge=LEFT).to_edge(UR, buff=0.2)
        self.play(*[Write(i) for i in text[:5]])
        self.play(Write(text[5]))
        self.wait()
        self.play(Write(text[6]))
        self.play(Write(text[7]))
        self.wait(2)
        self.play(
            FadeOutRandom(text),
            Write(text_2[0]),
            run_time=2,
        )
        self.wait()
        self.play(Write(text_2[1]))
        self.play(AnimationGroup(
            *[Write(i, run_time=1, lag_ratio=0.2) for i in text_2[2:6]],
            lag_ratio=0.4, run_time=4,
        ))
        self.wait(3)
        for i in text_2[6:]:
            self.play(Write(i))
        self.wait(2)
        self.play(FadeOutRandom(text_2, run_time=2))


class Turn_start_code(Scene):
    def construct(self):
        code = open(os.getcwd()+'\\code_2.c', 'r').readlines()
        cli = CommandLine()
        m_text = cli.m_text
        for i in code:
            cli.add_text(i)
        cli.code_dict.update({
            'LIST': '#4ec9b0', 'List': '#4ec9b0',
            'COORD': '#4ec9b0', 'Coord': '#4ec9b0',
            'SNAKE': '#4ec9b0', 'Snake': '#4ec9b0', 
            'clock_t': '#4ec9b0', 'HANDLE': '#4ec9b0', 
            'STUDENT': '#4ec9b0', 'Student': '#4ec9b0',
            'CONSOLE_CURSOR_INFO': '#4ec9b0',
        })
        cli.set_color_c_style()
        index = 0
        run_t = 1
        move = np.array([0, 0, 0])
        frame = self.camera.frame
        for count in [1, 2, 4, 18]:
            for i in range(count):
                self.play(
                    anim_CLI_types(cli, index),
                    ApplyMethod(frame.shift, move, rate_func=linear),
                    run_time=run_t,
                )
                if index == 12:
                    move = np.array([0, -0.02, 0])
                index += 1
            run_t = max(run_t*0.5, 0.1)
        cli.cursor_move_to(18, 154)
        self.add(cli)
        self.play(frame.animate.shift(DOWN*cli.unit_h*78), run_time=3, rate_func=rush_into)
        self.play(frame.animate.scale(1.5, about_edge=LEFT).shift(DOWN*cli.unit_h*78), run_time=3, rate_func=rush_from)
        for i in range(10):
            cli.cursor_del()
            self.wait(0.1)
        self.remove(cli).add(*m_text)
        
        code = open(os.getcwd()+'\\code_6.txt', 'r').readlines()
        rect = Rectangle(
            width=cli.unit_w*51, height=cli.unit_h*8,
            fill_opacity=0.3, fill_color=BLUE,
        ).move_to(cli.get_grid_place(33, 157.5))
        cli.cursor_move_to(0, 154)
        flag = False
        for i, m in enumerate(code[:8]):
            cli.new_text(m, 154+i, insert=flag)
            cli.set_color_c_style(154+i)
            self.play(anim_CLI_types(cli, 154+i), run_time=0.01*(len(m)-8))
            flag = True
        self.play(Reveal(rect,LEFT))
        for i, m in enumerate(code[8:16]):
            i += 162
            cli.new_text(m, i, insert=True)
            cli.set_color_c_style(i)
        m_text[170:].shift(8*cli.unit_h*UP)
        self.play(
            ApplyMethod(m_text[170:].shift, 8*cli.unit_h*DOWN),
            TransformFromCopy(m_text[154:162], m_text[162:170]),
            ApplyMethod(rect.shift, 8*cli.unit_h*DOWN),
        )
        for i, m in enumerate(code[16:32]):
            i += 170
            cli.new_text(m, i, insert=True)
            cli.set_color_c_style(i)
        m_text[186:].shift(16*cli.unit_h*UP)
        rect_copy = rect.copy()
        self.play(
            ApplyMethod(m_text[186:].shift, 16*cli.unit_h*DOWN),
            TransformFromCopy(m_text[162:170], m_text[170:178]),
            TransformFromCopy(m_text[162:170], m_text[178:186]),
            ApplyMethod(rect.shift, 8*cli.unit_h*DOWN),
            ApplyMethod(rect_copy.shift, 16*cli.unit_h*DOWN),
        )
        self.play(Uncreate(rect), Uncreate(rect_copy))
        
        text = VGroup()
        for i, m in enumerate(code[:32]):
            t = cli.new_word(m, 0, i+154)
            c_style_set_color(t, m, cli.code_dict)
            text.add(t)
        u_w, u_h = cli.unit_w, cli.unit_h
        t_1 = VGroup(
            cli.new_word('--', 16, 157), cli.new_word('++', 16, 165),
            cli.new_word('++', 16, 173), cli.new_word('--', 16, 181),
        )
        self.remove(*m_text[154:186])
        self.add(text)
        self.play(
            *[ReplacementTransform(text[2+i*8][:43], t_1[i]) for i in range(4)], # --
            *[ApplyMethod(text[3+i*8][16:].shift, u_w*2*RIGHT) for i in range(4)], # -- right
            *[ApplyMethod(text[2+i*8][43].shift, u_w*31*LEFT) for i in range(4)], # ;
            *[ApplyMethod(text[5+i*8][29].shift, u_w*RIGHT) for i in range(4)], # =
            *[ApplyMethod(text[5+i*8][44].shift, u_w*15*LEFT) for i in range(4)], # +
            *[ApplyMethod(text[5+i*8][46:].shift, u_w*14*LEFT) for i in range(4)], # grid
            *[FadeOut(text[5+i*8][31:43], RIGHT) for i in range(4)], # del
            run_time=2,
        )
        text_update_list = [
            ['if (--snake_head_x < 0)', 3],
            ['snake_head_x += grid_column;', 5],
            ['if (++snake_head_x >= grid_column)', 11],
            ['snake_head_x -= grid_column;', 13],
            ['if (++snake_head_y >= grid_row)', 19],
            ['snake_head_y -= grid_row;', 21],
            ['if (--snake_head_y < 0)', 27],
            ['snake_head_y += grid_row;', 29],
        ]
        self.remove(*t_1)
        for t, index in text_update_list:
            self.remove(*text[index])
            text[index].set_submobjects(cli.new_word(t, 16 if index % 4 == 1 else 12, index+154))
            c_style_set_color(text[index], t, cli.code_dict)
            self.add(text[index])
        self.play(
            *[ApplyMethod(text[i*8].shift, u_h*(i*5)*UP) for i in range(4)],
            *[ApplyMethod(text[3+i*8].shift, u_h*(2+i*5)*UP) for i in range(4)],
            *[ApplyMethod(text[5+i*8].shift, u_h*(3+i*5)*UP) for i in range(4)],
            *[FadeOut(text[1+i*8], LEFT) for i in range(4)],
            *[FadeOut(text[2+i*8][43], LEFT) for i in range(4)],
            *[FadeOut(text[4+i*8], LEFT) for i in range(4)],
            *[FadeOut(text[6+i*8], LEFT) for i in range(4)],
            *[FadeOut(text[7+i*8], LEFT) for i in range(4)],
            ApplyMethod(m_text[186:].shift, u_h*20*UP),
            run_time=2,
        )
        text.remove(*[text[i] for i in range(32) if i % 8 not in [0, 3, 5]])
        text_2 = VGroup()
        for i, m in enumerate(code[32:44]):
            t = cli.new_word(m, 0, i+154)
            c_style_set_color(t, m, cli.code_dict)
            text_2.add(t)
        self.play(
            Transform(text[0][12:20], text_2[0][12]),
            ApplyMethod(text[0][21:].shift, u_w*7*LEFT),
            *[Transform(text[3+i*3][17:25], text_2[3+i*3][17]) for i in range(3)],
            *[ApplyMethod(text[3+i*3][26:].shift, u_w*7*LEFT) for i in range(3)],
            *[Transform(text[1+i*3][6:18], text_2[1+i*3][18:20]) for i in range(4)],
            *[ApplyMethod(text[1+i*3][19:].shift, u_w*10*LEFT) for i in range(4)],
            *[Transform(text[2+i*3][:12], text_2[2+i*3][16:18]) for i in range(4)],
            *[Transform(text[2+i*3][16:-1], text_2[2+i*3][22:25]) for i in range(4)],
            *[ApplyMethod(text[2+i*3][13:15].shift, u_w*10*LEFT) for i in range(4)],
            *[ApplyMethod(text[2+i*3][-1].shift, u_w*(18 if i < 2 else 15)*LEFT) for i in range(4)],
            run_time=2,
        )
        text_2.remove(*text_2)
        for i, m in enumerate(code[44:48]):
            t = cli.new_word(m, 0, i+154)
            c_style_set_color(t, m, cli.code_dict)
            text_2.add(t)
        self.play(
            *[TransformMatchingShapes(text[i*3:i*3+3], text_2[i]) for i in range(4)],
            ApplyMethod(m_text[186:].shift, u_h*8*UP),
            run_time=2,
        )
        text.remove(*text)
        for i, m in enumerate(code[48:]):
            t = cli.new_word(m, 0, i+154)
            c_style_set_color(t, m, cli.code_dict)
            text.add(t)
        self.play(
            TransformMatchingShapes(text_2, text), 
            ApplyMethod(m_text[186:].shift, u_h*2*UP),
            run_time=2,
        )
        old_mt = VGroup(*text, *m_text[186:], *m_text[104:154])
        self.clear()
        end_code = open(os.getcwd()+'\\code_3.c', 'r').readlines()
        for i, j in zip(end_code, range(104, 104+len(end_code))):
            cli.new_text(i, j)
            cli.set_color_c_style(j)
        cli.m_text.remove(*cli.m_text[104+len(end_code):])
        self.remove(*cli.m_text).add(old_mt)
        self.play(
            frame.animate.shift(46.5*cli.unit_h*UP),
            TransformMatchingShapes(old_mt, cli.m_text[104:]),
            run_time=4,
        )
        self.wait()
        self.play(
            FadeOutRandom(cli.m_text[104:]),
            run_time=2,
        )
        self.wait()


class Turn_start_text(Scene):
    def construct(self):
        self.camera.background_rgba = [*Color(BLACK).get_rgb(), 1]
        text_list = [
            '随着代码不断堆积, 重复着',
            '不知何时, 感到厌倦, 无趣',
            '不经意间, 脑海中浮现一丝崭新的想法, 却又稍纵即逝',
            '即使如此, 也要重塑这虚无缥缈的构想',
            '打破常规, 重新定义, 不断突破',
            '代码随即扭曲, 瓦解, 混淆',
            '这就是混沌的开始...',
        ]
        t1, t2, t3, t4, t5, t6, t7 = [Text(i, font='思源黑体') for i in text_list]
        frame = self.camera.frame
        shift = [UP, DOWN, LEFT, RIGHT]
        t1_copy = [t1[9:].copy().set_fill(opacity=0.1*i) for i in range(10)]
        t1_low = t1[:9].copy().set_fill(opacity=0).set_stroke(width=2)
        self.wait()
        self.play(
            RouteCreation(t1_low, lambda i: (rush_from(0.1*i), rush_into(1.1*i))),
            rate_func=slow_into, run_time=2, lag_ratio=0.02,
        )
        self.play(
            FadeIn(t1[:9]),
            FadeOut(t1_low),
            AnimationGroup(*[FadeIn(i, DOWN) for i in t1_copy], lag_ratio=0.2, run_time=2),
        )
        t2_t1 = Text('漫无目的', font='思源黑体').align_to(t2[8], LEFT)
        t2_t2 = Text('毫无意义', font='思源黑体').align_to(t2[8], LEFT)
        self.wait()
        self.remove(*t1_copy)
        self.play(
            FadeOut(t1, RIGHT, rate_func=rush_from),
            AnimationGroup(Write(t2[:5]), Write(t2[6:10]), Write(t2[10:]), lag_ratio=1),
            run_time=2.4,
        )
        self.wait(0.2)
        self.play(ReplacementTransform(t2[8:], t2_t1), run_time=0.8)
        self.wait(0.2)
        self.play(ReplacementTransform(t2_t1, t2_t2), run_time=0.8)
        self.wait(0.2)
        self.play(
            RouteCreation(t2[:8], lambda i: (1-0.2*rush_into(i), 2-1.2*rush_into(i))),
            RouteCreation(t2_t2, lambda i: (1-0.2*rush_into(i), 2-1.2*rush_into(i))),
            remover=True, run_time=2,
        )
        t3.set_width(FRAME_WIDTH*0.9)
        frame.set_x(t3[9].get_left()[0])
        self.play(
            *[FadeIn(i, shift[random.randint(0, 3)]) for i in t3[:5]],
            rate_func=slow_into, run_time=0.8,
        )
        self.play(
            AnimationGroup(Write(t3[6:9]), Write(t3[11:18]), lag_ratio=1),
            Reveal(t3[9:11]),
            run_time=2,
        )
        self.play(
            frame.animate.center(),
            FadeIn(t3[18:], RIGHT),
        )
        self.play(FadeOutRandom(t3, lag_ratio=0.05, run_time=2, rate_func=rush_into))
        t4_l = t4[:5]
        t4_r = text_split_submob(t4[5:])
        t4_r.sort(lambda i:random.random())
        self.play(Write(t4_l), run_time=0.5)
        self.play(FadeInRandom(t4_r, lag_ratio=0.1, run_time=2))
        self.wait(1.5)
        self.remove(t4_l, t4_r)
        self.play(
            FadeOutRandom(t4),
            Write(t5[:5]),
            run_time=1,
        )
        self.play(*[TrimCreation(i, shift[int(random.random()*4)]) for i in t5[6:11]], run_time=1)
        self.play(*[Reveal(i, shift[int(random.random()*4)]) for i in t5[12:]], run_time=1)
        self.wait()
        t6 = text_split_submob(t6)
        t6.sort(lambda i: random.random())
        self.play(ReplacementTransform(t5, t6, path_arc=PI/4, run_time=2))
        self.wait(1)
        self.play(*[i.animate.shift(random.random()*shift[random.randint(0, 3)]) for i in t6], rate_func=slow_into, run_time=1)
        self.play(*[i.animate.shift(random.random()*shift[random.randint(0, 3)]) for i in t6], rate_func=slow_into, run_time=1)
        t7_chaos = Ftext('Chaos').move_to(t7[3:5])
        t7_cgame = Ftext('Cgame').match_x(t7_chaos).align_to(t7_chaos, UP)
        t7_l, t7_r = t7[:3], t7[5:]
        self.play(FadeOutRandom(t6), FadeIn(t7), run_time=1)
        self.add(t7_l, t7_r)
        turn_animation_into_updater(ApplyMethod(t7_l.next_to, t7_chaos, LEFT, {'buff': 0.1}))
        turn_animation_into_updater(ApplyMethod(t7_r.next_to, t7_chaos, RIGHT, {'buff': 0.1}))
        self.wait(0.7)
        self.play(Glitch(t7[3:5]), run_time=0.15, remover=True)
        self.play(Glitch(t7_chaos), run_time=0.15)
        self.wait(0.5)
        self.play(Transform(t7_chaos, t7_cgame))
        self.wait()


class Turn_stop(Scene):
    def construct(self):
        self.camera.background_rgba = [*Color(BLACK).get_rgb(), 1]
        rect = Rectangle(FRAME_WIDTH-1, FRAME_HEIGHT/2, stroke_width=10, color=BLUE)
        fill = rect.copy().set_stroke(width=0).set_fill(opacity=0.6)
        text = Text('战术暂停', font='思源黑体').scale(2)
        self.play(ShowCreation(rect), Write(text), run_time=1)
        self.play(Reveal(fill, LEFT, rate_func=linear, run_time=2))
        self.play(*[Uncreate(i, lag_ratio=0.1) for i in self.mobjects])


class Progress_bar(Scene):
    def construct(self):
        text_list = [
            '开头',
            '简单的介绍视频内容以及风格',
            '讲解',
            '进一步了解扫雷, 以及代码剖析',
            '演示',
            '展示最终效果, 简单地优化界面',
            '结尾',
            '部分内容的补充',
        ]
        text = VGroup(*[Text(i, font='思源黑体') for i in text_list])
        text.arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(UL, buff=0.2)
        for i in text[1::2]:
            i.scale(0.5, about_edge=UL)
        video_rect = Console(32, 9, FRAME_WIDTH*0.6).to_edge(UR, buff=0.2)
        video_rect.remove(video_rect.cursor)
        video_rect.background.set_height(0.6*FRAME_HEIGHT, True).next_to(video_rect.bg_title, DOWN, 0)
        text_2 = VGroup(*[Text(i, font='思源黑体') for i in [
            '"战术暂停"',
            '信息量略大的部分, 右下角会有"战术暂停"的标志',
            '大多为了节省时间, 内容不是很主要, 可以直接略过',
        ]])
        text_2.scale(0.5)[0].scale(1.6)
        text_2.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(video_rect, DOWN, aligned_edge=LEFT)
        progress_rate = [0.1196, 0.6346, 0.1987, 0.0471]
        pro_rect = VGroup(
            *[Rectangle(i*FRAME_WIDTH, 0.6, stroke_width=2, fill_opacity=1) for i in progress_rate]
        ).arrange(buff=0).to_edge(DOWN, buff=-0.1)
        for i, j in zip(pro_rect, [RED, GREEN, BLUE, ORANGE]):
            i.set_fill(j)
        for i, j in zip(pro_rect[:3], text[::2]):
            i.add(j.copy().set_height(0.4).match_x(i).to_edge(DOWN, buff=0.05))
        tbg_rect = Rectangle(
            0.33*FRAME_WIDTH, 0.2*FRAME_HEIGHT,
            stroke_width=0, color=RED,
        ).move_to(text[:2])
        self.add(tbg_rect, text, video_rect, pro_rect)
        self.play(
            *[Write(i) for i in text],
            RouteCreation(video_rect, lambda i: (0, i), lag_ratio=0),
            FadeIn(pro_rect, UP*0.5),
            run_time=2,
        )
        self.play(
            tbg_rect.animate.set_fill(opacity=1),
            pro_rect[0].animate.shift(0.1*UP),
        )
        for i, j, k in zip([text[i:i+2] for i in range(2, 8, 2)], pro_rect, pro_rect[1:]):
            self.wait(4)
            self.play(
                tbg_rect.animate.match_y(i),
                j.animate.shift(0.1*DOWN),
                k.animate.shift(0.1*UP),
            )
        self.wait(4)
        self.play(
            tbg_rect.animate.set_fill(opacity=0),
            pro_rect[3].animate.shift(0.1*DOWN),
            *[Write(i) for i in text_2],
        )
        self.wait()


class Subtitle(Scene):
    text = ['test-1', 'test-2']
    def construct(self):
        self.camera.background_rgba = [*Color(BLACK).get_rgb(), 1]
        text = [Dot(fill_opacity=0, radius=0)]+[
            Text(i, font='思源黑体') for i in self.text
        ]+[Dot(fill_opacity=0, radius=0)]
        for i, j in zip(text, text[1:]):
            self.play(FadeOut(i), FadeIn(j))
            self.wait()


class Subtitle_1(Subtitle):
    text = [
        '扫雷, 一款简单又有趣的小游戏',
        '游戏规则很简单, 找出所有没有地雷的方格',
        '如果踩到有地雷的方格, 则全盘皆输',
        '相信大家都见过早期Windows系统附带的版本',
        '但在win7版本之后, Windows系统不再附带扫雷',
        '但能通过应用商店(Microsoft store)来获取',
        '这种版本的扫雷, 虽然多了几分特色',
        '却不再是我们熟悉的那个扫雷了...',
        '所以, 不如自己做一个扫雷?',
    ]


class Subtitle_2(Subtitle):
    text = [
        '到此, 游戏的流程和细节就差不多讲完了',
        '这里使用的是Dev-c++ 5.11进行演示',
        '......',
        '(深思熟虑)',
        '(放弃思考)',
        '可能是打开方式不对, 修改一点细节',
        '简  单',
        '雷区大小和雷数可以随意设置',
        '(只要不是太离谱)',
    ]


class Subtitle_3(Subtitle):
    text = [
        '效果显著, 观感有明显提升',
        '(简单)',
        '(血压拉满)',
        '可能是觉得太简单然后大意了(雾)',
        '提高下难度吧, 不能显得我太菜()',
        '......',
        '什么嘛, 我还是挺厉害的',
    ]


class Subtitle_4(Subtitle):
    text = [
        '关于代码风格(规范)',
        '代码使用Clang-Format格式化',
    ]
