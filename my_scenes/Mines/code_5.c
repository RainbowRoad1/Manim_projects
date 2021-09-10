#include <conio.h>
#include <stdio.h>
#include <windows.h>
int S, W = 9, H = 9, B = 10, s, p = 0, c = 1, i, *m, *M, (*f)(int, int), *O;
int edge(int x, int y) { return !(-~x % -~W && -~y % -~H); }
void tmp(int x, int y) { edge(x, y) || m[x += W * y] - 9 && ++m[x]; }
void dig(int v, int l) {
    for (l = edge(v, l) || M[v += W * l] || (++M[v], --s, m[v]) ? 0 : 9; l;)
        --l - 4 || --l, f(v % W + l % 3 - 1, v / W + l / 3 - 1);
}
void set(int x, int y) {
    for (f = tmp, srand(m); c < B || (f = dig, f(x, y), s += B, 0);)
        if (m[i = rand() % S] < 9 && 1 < (abs(x - i % W) | abs(y - i / W)))
            m[i] = 0, dig(i % W, i / W), m[i] = 9, --M[i], ++c;
}
int main() {
    COORD C = (s = S = W * H, O = GetStdHandle((DWORD)-11), (COORD){0});
    m = calloc(S * 2, 4), SetConsoleCursorInfo(O, &(CONSOLE_CURSOR_INFO){1});
    for (f = set, M = m + S; c - 27; c = B % s ? _getch() & 95 : 27) {
        c - 68 || ++p, c - 65 || --p, c - 83 || (p += W), c - 87 || (p -= W);
        p = (p + S) % S, c || (f(p % W, p / W), m[p] < 9 || (B = 0));
        for (SetConsoleCursorPosition(O, C), i = 0; i < S; ++i % W || puts(""))
            SetConsoleTextAttribute(O, (M[i] ? m[i] : 9) | (p - i ? 240 : 64)),
                printf(M[i] || B < m[i] - 8 ? " %c" : "?", " 12345678@"[m[i]]);
    }
    puts(B - s ? "Game over!" : "You win!"), _getch();
}
