n = 700;
m = rand(n,1);
k = zeros(n,1);
fract = zeros(n,1);
for i = 1:n
  if m(i)<.1667
    k(i) = 1;
   endif 
  if m(i)>.1667
    k(i) =0;
    endif
endfor
##Assigning outcomes to 1 if m<.1667 and 0 if m>.1667
fract(1)=k(1);
for j = 1:n-1
  fract(j+1)=fract(j)+k(j+1);
endfor
##Assigning the prob to 0
prob = zeros(n,1);
for l = 1:n
  prob(l)=fract(l)/l;
endfor
##Depicting the probability
prob;
y = 1:700;
##plotting the simulated experiment
plot(y,prob)
xlabel('No of throws');
ylabel('Probability achieved');
title('Monte-Carlo Simulation of rolling dice')