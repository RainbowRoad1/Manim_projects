#include <conio.h>
#include <stdlib.h>
int S, W = 9, H = 9, B = 10, s, p = 0, c = 1, i, *m, *M, (*f)(int, int);
int edge(int x, int y) { return !(-~x % -~W && -~y % -~H); }
void tmp(int x, int y) { edge(x, y) || m[x += W * y] - 9 && ++m[x]; }
void dig(int v, int l) {
    for (l = edge(v, l) || M[v += W * l] || (++M[v], --s, m[v]) ? 0 : 9; l;)
        --l - 4 || --l, f(v % W + l % 3 - 1, v / W + l / 3 - 1);
}
void set(int x, int y) {
    for (f = tmp; c < B || (f = dig, f(x, y), s += B, 0);)
        if (m[i = rand() % S] < 9 && 1 < (abs(x - i % W) | abs(y - i / W)))
            m[i] = 0, dig(i % W, i / W), m[i] = 9, --M[i], ++c;
}
int main() {
    f = set, s = S = W * H, m = calloc(S * 2, 4), M = m + S, srand(m);
    for (; c - 27 && !system("cls"); c = B % s ? _getch() & 95 : 27) {
        c - 65 || --p, c - 68 || ++p, c - 83 || (p += W), c - 87 || (p -= W);
        p = (p + S) % S, c || (f(p % W, p / W), m[p] < 9 || (B = 0));
        for (i = 0; B || m[i] - 9 || ++M[i], i < S; ++i % W || _cputs("\n"))
            _cprintf("%c%c", " >"[p == i], " 12345678@*"[M[i] ? m[i] : 10]);
    }
    _cputs(B - s ? "Game over!" : "You win!"), _getch();
}