from manimlib import *
sys.path.append(os.getcwd())
from my_utils import *


class test_RouteCreation(Scene):
    def construct(self):
        mobjects_1 = VGroup(
            Circle(),
            Circle(fill_opacity=1),
            Ftext("Text").scale(2)
        )
        mobjects_1.arrange(RIGHT, buff=2).shift(TOP/2)
        mobjects_2 = mobjects_1.copy().shift(BOTTOM)
        self.play(
            *[ShowCreation(mob) for mob in mobjects_1],
            *[RouteCreation(mob) for mob in mobjects_2],
            run_time=2
        )
        self.wait()


class test_TrimCreation(Scene):
    def construct(self):
        mobjects = VGroup(
            Circle(),
            Circle(fill_opacity=1),
            Ftext("Text").scale(2)
        )
        mobjects.arrange(RIGHT, buff=2)
        self.play(
            *[TrimCreation(mob, DOWN) for mob in mobjects],
            run_time=2
        )
        self.wait()


class test_Reveal(Scene):
    def construct(self):
        mobjects = VGroup(
            Circle(),
            Circle(fill_opacity=1),
            Ftext("Text").scale(2)
        )
        mobjects.arrange(RIGHT, buff=2)
        self.play(
            *[Reveal(mob, LEFT) for mob in mobjects],
            run_time=2
        )
        self.wait()


class test_Glitch(Scene):
    def construct(self):
        mobject = Ftext("Hello, world").scale(2)
        self.play(Reveal(mobject))
        self.play(Glitch(mobject), run_time=0.2)
        self.wait()


class test_FadeRandom(Scene):
    def construct(self):
        mobjects_1 = Ftext("Hello, world!\nHello, manim!")
        mobjects_1.scale(1.5).shift(TOP/2)
        mobjects_2 = mobjects_1.copy().shift(BOTTOM)
        self.play(
            *[FadeIn(mob) for mob in mobjects_1],
            *[FadeInRandom(mob) for mob in mobjects_2],
        )
        self.wait()
        self.play(
            *[FadeOut(mob) for mob in mobjects_1],
            *[FadeOutRandom(mob) for mob in mobjects_2],
        )
        self.wait()
