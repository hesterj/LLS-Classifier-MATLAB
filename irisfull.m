%LLS classifier for the simplified iris dataset

B = fopen('iris.txt','r');

%%%
% This computes the number of rows
 c = 0;
fgetl(B);
while ~feof(B)
fgetl(B);
c = c+1;
end
numrows = c;

%%%
B = fopen('iris.txt','r');
C = textscan(B,'%f ,%f ,%f ,%f ,%s');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5}; %tags

training = 11;  %  specify first n rows to be used as training.  
%For training purposes I mixed up the original file a bit so that all the
%different species of iris would be represented in the training set. 

col6=zeros(3,1,training);

irisvirginica = 'Iris-virginica';
irissetosa = 'Iris-setosa';

for n=1:numrows  % this loop assigns numerical tags
    str1 = col5(n);
    tv = strcmp(str1,irisvirginica);
    ts = strcmp(str1,irissetosa);
    if tv==1
        col6(1,1,n)=1;
    elseif ts==1
        col6(2,1,n)=1;
    else
        col6(3,1,n)=1;     
    end
end

x = [col1 col2 col3 col4];  %data row vector

lhs = zeros(4,4,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(4,3,training);
rhstemp = [1:4];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*col6(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(4,3);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(3,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));
end

col7 = zeros(3,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col7(:,:,i)=[1,0,0]';
    elseif results(i)==2
        col7(:,:,i)=[0,1,0]';
    else 
        col7(:,:,i)=[0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = col6(:,:,i)-col7(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct
