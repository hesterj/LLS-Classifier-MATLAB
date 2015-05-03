%LLS classifier for the simplified iris dataset

B = fopen('survey.txt','r');

%%%
% This computes the number of rows
 c = 0;
fgetl(B);
while ~feof(B)
fgetl(B);
c = c+1;
end
numrows = c;

constraint = zeros(numrows,1);
constraint = constraint + 1;

%%%
B = fopen('survey.txt','r');
C = textscan(B,'%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%s');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5}; 
col6 = C{6};
col7 = C{7};
col8 = C{8};
col9 = C{9};

training = 0.1*numrows;
training = round(training); % 10% training set

labels=zeros(5,1,training);

insta = 'Instagram';
none = 'I use none of these';
pin = 'Pinstagram';
twitter = 'Twitter';
tumblr = 'Tumblr'

for n=1:numrows  % this loop assigns numerical tags
    str1 = col9(n);
    ti = strcmp(str1,insta);
    tn = strcmp(str1,none);
	tp = strcmp(str1,pin);
	tt = strcmp(str1,twitter);
	tr = strcmp(str1,tumblr);
    if ti==1
        labels(1,1,n)=1;
    elseif tn==1
        labels(2,1,n)=1;
	elseif tp==1
        labels(3,1,n)=1;
	elseif tt==1
        labels(4,1,n)=1;
    else
        labels(5,1,n)=1;     
    end
end

x = [constraint col1 col2 col3 col4 col5 col6 col7 col8];  %data row vector

lhs = zeros(9,9,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(9,5,training);
rhstemp = [1:4];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*labels(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(9,5);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(5,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));  %sets results(i) to index containing largest element
end

col10 = zeros(5,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col10(:,:,i)=[1,0,0,0,0]';
    elseif results(i)==2
        col10(:,:,i)=[0,1,0,0,0]';
	elseif results(i)==3
        col10(:,:,i)=[0,0,1,0,0]';
	elseif results(i)==4
        col10(:,:,i)=[0,0,0,1,0]';
    else 
        col10(:,:,i)=[0,0,0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = labels(:,:,i)-col10(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct





%LLS classifier for the simplified iris dataset

B = fopen('survey.txt','r');

%%%
% This computes the number of rows
 c = 0;
fgetl(B);
while ~feof(B)
fgetl(B);
c = c+1;
end
numrows = c;

constraint = zeros(numrows,1);
constraint = constraint + 1;

%%%
B = fopen('survey.txt','r');
C = textscan(B,'%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%s');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5}; 
col6 = C{6};
col7 = C{7};
col8 = C{8};
col9 = C{9};

training = 0.5*numrows;
training = round(training); % 50% training set

labels=zeros(5,1,training);

insta = 'Instagram';
none = 'I use none of these';
pin = 'Pinstagram';
twitter = 'Twitter';
tumblr = 'Tumblr'

for n=1:numrows  % this loop assigns numerical tags
    str1 = col9(n);
    ti = strcmp(str1,insta);
    tn = strcmp(str1,none);
	tp = strcmp(str1,pin);
	tt = strcmp(str1,twitter);
	tr = strcmp(str1,tumblr);
    if ti==1
        labels(1,1,n)=1;
    elseif tn==1
        labels(2,1,n)=1;
	elseif tp==1
        labels(3,1,n)=1;
	elseif tt==1
        labels(4,1,n)=1;
    else
        labels(5,1,n)=1;     
    end
end

x = [constraint col1 col2 col3 col4 col5 col6 col7 col8];  %data row vector

lhs = zeros(9,9,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(9,5,training);
rhstemp = [1:4];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*labels(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(9,5);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(5,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));  %sets results(i) to index containing largest element
end

col10 = zeros(5,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col10(:,:,i)=[1,0,0,0,0]';
    elseif results(i)==2
        col10(:,:,i)=[0,1,0,0,0]';
	elseif results(i)==3
        col10(:,:,i)=[0,0,1,0,0]';
	elseif results(i)==4
        col10(:,:,i)=[0,0,0,1,0]';
    else 
        col10(:,:,i)=[0,0,0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = labels(:,:,i)-col10(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct
