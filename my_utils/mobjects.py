import random
from manimlib.mobject.svg.text_mobject import Text
from manimlib.utils.config_ops import digest_config
from manimlib.mobject.geometry import *
from my_utils.functions import *
MYNUM_MOBJECT_DATA = {}
FTEXT_MOBJECT_DATA = {}
FTEXT_WORD_SPACE = 0.3688304874239998
FTEXT_ROW_SPACE = 0.670654537728


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
            def func(i): return i
        elif type(init) == list:
            def func(i): return init[i]
        else:
            func = init
        for i, m in enumerate(self.box):
            num.add(Ftext('%d' % func(i)).scale(size).move_to(m.get_corner(to)-to*buff, to))
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
