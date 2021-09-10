from my_utils.animations import *


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
