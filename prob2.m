M=[1,2,3,4,5,6,7,8,9,10];
SH_A = [38,44,41,40,41,39,44,46,38,42];
SH_B = [43,41,40,39,36,35,45,42,41,41];
SH_C = [48,42,43,41,39,40,43,38,42,40];
SH_D = [45,43,41,42,42,39,40,39,40,41];
SH_E = [36,39,41,40,46,42,40,39,37,42];
ST = ['SH_A';'SH_B';'SH_C';'SH_D';'SH_E']
SI = [SH_A;SH_B;SH_C;SH_D;SH_E]
MeanSH_A = mean(SH_A);
MeanSH_B = mean(SH_B);
MeanSH_C = mean(SH_C);
MeanSH_D = mean(SH_D);
MeanSH_E = mean(SH_E);
Mean = [MeanSH_A,MeanSH_B,MeanSH_C,MeanSH_D,MeanSH_E]
Range_A = max(SH_A)-min(SH_A);
Range_B = max(SH_B)-min(SH_B);
Range_C = max(SH_C)-min(SH_C);
Range_D = max(SH_D)-min(SH_D);
Range_E = max(SH_E)-min(SH_E);
Range = [Range_A,Range_B,Range_C,Range_D,Range_E]
Std_A = std(SH_A);
Std_B = std(SH_B);
Std_C = std(SH_C);
Std_D = std(SH_D);
Std_E = std(SH_E);
Sta_Dev = [Std_A,Std_B,Std_C,Std_D,Std_E]
Med_A = median(SH_A);
Med_B = median(SH_B);
Med_C = median(SH_C);
Med_D = median(SH_D);
Med_E = median(SH_E);
Med = [Med_A,Med_B,Med_C,Med_D,Med_E]
IQR_A = iqr(SH_A);
IQR_B = iqr(SH_B);
IQR_C = iqr(SH_C);
IQR_D = iqr(SH_D);
IQR_E = iqr(SH_E);
IQR = [IQR_A,IQR_B,IQR_C,IQR_D,IQR_E]
pkg load dataframe
Table = {ST;Mean;Range;Sta_Dev;Med;IQR};
pkg load statistics
SI = SI'
boxplot(SI)
xlabel('Hardness')
ylabel('No of values')
title('Rockwell Hardness Measurement')