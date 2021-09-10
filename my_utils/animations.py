from manimlib.animation.animation import Animation
from manimlib.animation.composition import AnimationGroup
from manimlib.utils.rate_functions import *
from manimlib.animation.fading import *
from manimlib.utils.bezier import *
from manimlib.constants import *
from my_utils.mobjects import *


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


class NumberToValue(Animation):
    def __init__(self, mob, value, **kwargs):
        self.v_new = value
        self.v_old = mob.number
        self.n_round = mob.n_round
        super().__init__(mob, **kwargs)

    def interpolate_mobject(self, alpha):
        new = round(interpolate(self.v_old, self.v_new, alpha), self.n_round)
        self.mobject.re_num(new)


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
