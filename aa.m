clc
clear all
close all

% CRIM: Per capita crime rate by town
% ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
% INDUS: Proportion of non-retail business acres per town
% CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
% NOX: Nitric oxide concentration (parts per 10 million)
% RM: Average number of rooms per dwelling
% AGE: Proportion of owner-occupied units built prior to 1940
% DIS: Weighted distances to five Boston employment centers
% RAD: Index of accessibility to radial highways
% TAX: Full-value property tax rate per $10,000
% PTRATIO: Pupil-teacher ratio by town
% B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
% LSTAT: Percentage of lower status of the population
% MEDV: Median value of owner-occupied homes in $1000s





%% Information regrading the dataset
M = readtable('Boston.csv');

%% checking for the missing value

Y = table2array(M);
TF = sum(ismissing(Y),1);
TF1=sum(isnan(Y),1);
T=Y(:,2:end);                                                              %% removing the index coloumn
Average=mean(T,1);                                                         %% average of each column in the datset
sta=std(T,[],1,"omitnan");                                                 %% standard devation in the matlab
ma=max(T,1);                                                               %% maximum in the dataset
mi=min(T,1);                                                               %% minimum in the dataset
V=var(T,[],1,"omitnan");                                                   %% variance in the data set
p_1=prctile(T,25,1);                                                       %% percentile within 25%
p_2=prctile(T,50,1);                                                       %% percentile within 50%
p_3=prctile(T,75,1);                                                       %% percentile within 75%
k = kurtosis(T,1);                                                         %% kurtosis of the data                                      
y = skewness(T,1);                                                         %% skewness of the data
def_disc=[Average;p_1;p_2;p_3;ma;mi;y;V;sta;k;];
%%
% %%Visualization
cdata= abs(corr(T));                                                       
xvalues = {'Crim','Zn','Indus','Chas','Nox','Rm','Age','Dis','Rad','Tax','Pt-Ratio','B','L-Stat','Medv'};
yvalues = {'Crim','Zn','Indus','Chas','Nox','Rm','Age','Dis','Rad','Tax','Pt-Ratio','B','L-Stat','Medv'};
figure(1)
h = heatmap(xvalues,yvalues,cdata);
%%
%%% Boxplot for detection of outliers
figure(2)
subplot(2,7,1)
boxplot(T(:,1))
ylabel('CRIM')
hold on
subplot(2,7,2)
boxplot(T(:,2))
ylabel('ZN')
hold on
subplot(2,7,3)
boxplot(T(:,3))
ylabel('INDUS')
hold on
subplot(2,7,4)
boxplot(T(:,4))
ylabel('CHAS')
hold on
subplot(2,7,5)
boxplot(T(:,5))
ylabel('NOX')
hold on
subplot(2,7,6)
boxplot(T(:,6))
ylabel('RM')
hold on
subplot(2,7,7)
boxplot(T(:,7))
ylabel('AGE')
hold on
subplot(2,7,8)
boxplot(T(:,8))
ylabel('DIS')
hold on
subplot(2,7,9)
boxplot(T(:,9))
ylabel('RAD')
hold on
subplot(2,7,10)
boxplot(T(:,10))
ylabel('TAX')
hold on
subplot(2,7,11)
boxplot(T(:,11))
ylabel('PTRATIO')
hold on
subplot(2,7,12)
boxplot(T(:,12))
ylabel('BLACK')
hold on
subplot(2,7,13)
boxplot(T(:,13))
ylabel('LSTAT')
hold on
subplot(2,7,14)
boxplot(T(:,14))
ylabel('MDEV')
%%
%R=[T(:,3),T(:,5:7),T(:,8:2:10),T(:,11:2:13),T(:,14)];
%R=[T(:,3:11),T(:,13),T(:,14)];
%R=[T(:,6),T(:,13),T(:,14)];
R=[T(:,3:6),T(:,7:11),T(:,13:14)];

%% LSTAT,INDUS NOX PTRATIO RM TAX DIS AGE HAS GOOD CORRELATION WITH MEDEV
%% CHAS DOESNT HAVE ANY EFFECT
%% CRIM,ZN,BLACK
     
%%
figure(3)
subplot(3,4,1)
s1 = plot(R(:,1), R(:,11),'g+');
set(s1, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Indus', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')
%%
subplot(3,4,2)
s2 = plot(R(:,2), R(:,11),'g+');
set(s2, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('chas', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')

%%
subplot(3,4,3)
s3 = plot(R(:,3), R(:,11),'g+');
set(s3, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Nox', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')
%%
subplot(3,4,4)
s4 = plot(R(:,4), R(:,11),'g+');
set(s4, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Rm', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')
%%
subplot(3,4,5)
s5 = plot(R(:,5), R(:,11),'g+');
set(s5, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Age', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')
%%
subplot(3,4,6)
s6 = plot(R(:,6), R(:,11),'g+');
set(s6, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Dis', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')


%%
subplot(3,4,7)
s7 = plot(R(:,7), R(:,11),'g+');
set(s3, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Rad', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')
%%
subplot(3,4,8)
s8 = plot(R(:,8), R(:,11),'g+');
set(s8, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Tax', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')

%%
%%
subplot(3,4,9)
s8 = plot(R(:,9), R(:,11),'g+');
set(s8, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Ptratio', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')

%%
subplot(3,4,10)
s9 = plot(R(:,10), R(:,11),'g+');
set(s9, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('Lstat', 'FontSize', 20)
ylabel('Medv', 'FontSize', 20)
set(gca, 'FontSize', 20, 'YMinorTick','on','XMinorTick','on')

%%
C = [1,10,100,1000,10000];
nc = size(C,2);
sigma =[0.0001 0.001 0.01 0.1 0.2 0.5 0.6 0.9];
nsigma = size(sigma,2);
%%
Traning_independent=R(1:405,1:10);                                         %% Traning independent features
Traning_dependent=R(1:405,11);                                             %% Tranining responses 
%%
Test_indpendent=R(406:506,1:10);                                           %% Test independent features
Test_dependent=R(406:506,11);                                              %% Test responses
%%
no_of_independent_points=size(Traning_independent,1);                      %% Number of sample per fold independent feature 
no_of_dependent_points=size(Traning_dependent,1);                          %%  Number of sample per fold dependent feature
%%
no_CV_folds = 5;                                                           % k-fold Cross-Validation                                               
Independent_features_perfold = round(no_of_independent_points/no_CV_folds);
dependent_features_perfold = round(no_of_dependent_points/no_CV_folds);
%%
SVM_Models = cell(1);
count = 1;
val_results=zeros;
%%
for i = 1:nc                                                                % Loop for Hyperparameter: C
    for j = 1:nsigma                                                        % Loop for Hyperparameter: Sigma
        for k = 1:no_CV_folds                                               % Loop for k-fold Cross-Validation
        val_index_independent = (k-1)*Independent_features_perfold+1:Independent_features_perfold*k;
        val_index_dependent = (k-1)*dependent_features_perfold +1:Independent_features_perfold*k;
    
        train_independent=Traning_independent(setdiff((1:no_of_independent_points),val_index_independent),:);
        train_dependent= Traning_dependent(setdiff((1:no_of_dependent_points),val_index_dependent),:);
        %%
        SVM = fitrsvm(train_independent,train_dependent,'BoxConstraint',C(i),'Kernelfunction','rbf','KernelScale',sigma(j));
        SVM_Models{count,1} = SVM;
        %%
        Train_independent = Traning_independent(val_index_independent,:);
        Train_dependent = Traning_dependent(val_index_dependent,:);
        %%
        y_predict = predict(SVM,Train_independent);
        %%
        v=size(Train_dependent,1);
        RMSE=sqrt((sum((Train_dependent-y_predict).^2))/(v));
        MSE=sum((Train_dependent-y_predict).^2)/(v);
        AMSE=sum(abs(Train_dependent-y_predict))/(v);
       
        val_results(count,1) = C(i);
        val_results(count,2) = sigma(j);
        val_results(count,3) = RMSE;
        val_results(count,4) = MSE;
        val_results(count,5)=AMSE;
     
      
        
        
        count=count+1;
        end
         
    end
   

end

 [N,U]=min(val_results(:,3),[],1);
 SVM1 = SVM_Models{U,1};
 y_predict1=predict(SVM1,Test_indpendent);
 v1=size(Test_indpendent,1);
 RMSE1=sqrt((sum((Test_dependent-y_predict1).^2))/(v1));
 MSE1=sum((Test_dependent-y_predict1).^2)/(v1);
 AMSE1=sum(abs(Test_dependent-y_predict1))/(v1);

