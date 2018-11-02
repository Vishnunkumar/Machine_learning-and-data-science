function y = f(t)
  T_1 = 100*(1-(exp(-.2*t)));
  T_2 = 40*(exp(-.1*t));
  y = T_1-T_2; 
endfunction