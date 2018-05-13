#include <stdio.h>
#include <math.h>
#define M_PI acos(-1.0)

int main(){
  double n[1000];
  int k;
  double whole;
  double alpha;

  n[0]=0.2;
  printf ("%d, %f\n",0,n[0]);
  alpha=M_PI;
  for (k=1; k<=999; k++){
  whole=(n[k-1]+M_PI)*100;
  n[k]=whole - ((long)whole);
  printf ("%d, %f\n",k,n[k]);
}
}
