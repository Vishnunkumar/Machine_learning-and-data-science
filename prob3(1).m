n = 500;
m = rand(n,1);
k = zeros(n,1);
z = 0;
fract = zeros(n,1);
for i = 1:n
  if m(i)>.1667
     k(i) = k(i)+1; 
endif
endfor
k