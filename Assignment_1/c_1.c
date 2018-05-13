#include <stdio.h>



int main(){
  int n,nold,new,k;
  n=1;
  nold=1;
  printf ("%d, %d\n",1,nold);
  printf ("%d, %d\n",2,n);
  for (k=3; k<=10; k++) {
  new=n+nold;
  nold=n;
  n=new;
  printf("%d, %d\n", k,new);
}
}
