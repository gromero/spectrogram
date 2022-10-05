#include <stdio.h>

int main()
{
  int v = (int) 0x11111100;
  float f = ((float) v) * (1.0 / ((float) 0x80000000));
  printf("%#.17f\n", f);
}
