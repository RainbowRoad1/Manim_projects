#include <conio.h> // Code from https://github.com/RainbowRoad1/Cgame
#include <stdlib.h>
// 14 lines made snake game.
int main() {
    int W = 20, S = W * W, *m, z[2] = {0}, l = 3, i, c = 'D', C, *p, f;
    for (srand(m = calloc(S, 4)), C = m[1] = -1; C - 27; _sleep(100)) {
        if (_kbhit())C = _getch() & 95, C - 65 && C - 68 &&
            C - 83 && C - 87 || (C ^ c) & 20 ^ 4 && (c = C);
        p = z + !!(c & 2), *p += c / 3 & 2, *p = (--*p + W) % W;
        f = !system("cls"), *(p = m + *z + z[1] * W) > 0 && (C = 27);
        for (; *p && (m[i = rand() % S] || (--m[i], ++l, --f)););
        for (i = 0, *p = l; i < S; ++i % W || _cputs("|\n"))
            _cputs(m[i] > 0 ? m[i] -= f, "()" : m[i] ? "00" : "  ");
    }
}
// 22 lines made tetris game.
#define Get(C) for (C, i = n[T]; j = X + i % 4, k = Y + i / 4 % 4, i; i >>= 4)
int W = 10, H = 25, S, i, j, k, c, d = 0, X = 0, Y = 0, T = 0, *m,
    n[] = {25921, 38481, 38484, 38209, 25922, 43345, 34388, 38160, 25920, 38177,
           42580, 38993, 51264, 12816, 25872, 34113, 21537, 38208, 21520};
int move(int *v, int l) {
    Get (*v += l)(j < 0 || j >= W || k >= H || m[k * W + j]) && (c = 0);
    return c ? 1 : (*v -= l, v == &Y && (c = -1));
}
int main_2() {
    for (srand(m = calloc(S = W * H, 4)); c - 27; _sleep(50), system("cls")) {
        Get(c = _kbhit() ? _getch() & 95 : 1) m[k * W + j] = 0;
        c ^ 65 || move(&X, -1), c ^ 68 || move(&X, 1), c ^ 83 || move(&Y, 1);
        c ^ 87 || (i = T < 12 ? 3 : T != 18, move(&T, T & i ^ i ? 1 : -i));
        Get(++d - 10 || (d = 0, c = 1, move(&Y, 1))) m[k * W + j] = 1;
        if (c == -1 && !(Y || (c = 27), T = rand() % 20, Y = X = 0))
            for (j = W, i = S - 1; j -= m[i], i; i-- % W || (j = W))
                for (j || (k = i += W); !j && (--k >= W); m[k] = m[k - W]) {}
        for (; i < S; ++i % W || _cputs("|\n")) _cputs(m[i] ? "[]" : "  ");
    }
}
// More games in https://github.com/RainbowRoad1/Cgame