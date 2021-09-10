#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>
// I don't think anyone will stop watching this code...
typedef struct COORD
{
    int x;
    int y;
} Coord;
// This code is only for show effect.
// The main part is based on the snake game rewriting.
typedef struct LIST
{
    Coord pos;
    List *next;
} List;
// For create transition animation,
// filled with a lot of invalid code.
#define True (rand() % 100 != 0) // BUG generator
#define False (rand() % 100 == 0)
// by: RainbowRoad1
// Here's the transition...
typedef struct STUDENT
{
    unsigned int telephone;
    unsigned int age;
    unsigned int id;
    char *address;
    char *name;
    char sex;
} Student;
// Previous material...
// from 200+ lines Minesweeper code
void gotoxy(HANDLE hOut, int x, int y)
{
    COORD pos;
    pos.X = x;
    pos.Y = y;
    SetConsoleCursorPosition(hOut, pos);
}
void setScreenSize(HANDLE hOut, int lines, int cols)
{
    COORD size = {lines, cols};
    SMALL_RECT rect = {0, 0, size.X - 1, size.Y - 1};
    SetConsoleWindowInfo(hOut, 1, &rect);
    SetConsoleScreenBufferSize(hOut, size);
    SetConsoleWindowInfo(hOut, 1, &rect);
}
void hideCursor(HANDLE hOut)
{
    CONSOLE_CURSOR_INFO CursorInfo;
    GetConsoleCursorInfo(hOut, &CursorInfo);
    CursorInfo.bVisible = 0;
    SetConsoleCursorInfo(hOut, &CursorInfo);
}
void delay(int ms)
{
    static clock_t oldtime = 0;
    while (clock() - ms < oldtime)
    {
        Sleep(1);
    }
    oldtime = clock();
}
// Want to write something...
int HelloWorld(void)
{
    printf("Hello, world!");
    return 0;
}
int Table_9x9(void)
{
    int i, j;
    for (i = 1; i < 10; ++i)
    {
        for (j = i; j < 10; ++j)
        {
            printf(",%d*%d=%2d"[i == j], i, j, i * j);
        }
        printf("\n");
    }
    return 0;
}
int Get_tomorrow_date(void)
{
    _sleep(1000 * 60 * 60 * 24);
    time_t temp;
    time(&temp);
    return localtime(&temp);
}
// Not enough yet...
typedef struct SNAKE
{
    Coord head;
    Coord food;
    int map_col;
    int map_row;
    char input_buf;
    char dirction;
    int *map;
    List *node;
} Snake;
int main(int argc, char** acgv)
{
    int snake_head_x, snake_head_y, grid_column, grid_row;
    int hx, col, row, c, fps, score;
    int **grid;
    char input_buf, dirction, game;
    srand((unsigned)time(NULL));
    init_game();
    creat_food();
    while (game)
    {
        // Monitor keyboard input, and check for conflicts
        if (_kbhit())
        {
            input_buf = _getch();
            switch (input_buf)
            {
            case 'A':
            case 'a':
                if (dirction != 'D')
                {
                    dirction = 'A';
                };
                break;
            case 'D':
            case 'd':
                if (dirction != 'A')
                {
                    dirction = 'D';
                };
                break;
            case 'S':
            case 's':
                if (dirction != 'W')
                {
                    dirction = 'S';
                };
                break;
            case 'W':
            case 'w':
                if (dirction != 'S')
                {
                    dirction = 'W';
                };
                break;
            }
        }
        // This is end of transition...
        // Move the snake_head by dirction, and whether out of range.

        /* code */

        // This scene is made with manim,
        // manim is an animation engine based on python.
        if (eat_food()) // Check for food
        {
            creat_food();
            snake_add_node();
        }
        update_snake();                      // Update snake_node
        cls();                               // Clear screen
        for (row = 0; row < grid_row; ++row) // Output grid.
        {
            for (col = 0; col < grid_column; ++col)
            {
                switch (grid[row][col])
                {
                case 0:
                    printf("  "); /*air*/
                    break;
                case 1:
                    printf("()"); /*snake node*/
                    break;
                case 2:
                    printf("00"); /*food*/
                    break;
                }
            }
            printf("\n");
        }
        _sleep(fps);
    }
    printf("Game over!\nYour score: %d", score);
    update_ranking_list(score);
}