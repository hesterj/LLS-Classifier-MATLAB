%LLS classifier for the simplified zoo dataset

B = fopen('zoo.txt','r');

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
B = fopen('zoo.txt','r');
C = textscan(B,'%s ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5};
col6 = C{6};
col7 = C{7};
col8 = C{8};
col9 = C{9};
col10 = C{10};
col11 = C{11};
col12 = C{12};
col13 = C{13};
col14 = C{14};
col15 = C{15};
col16 = C{16};
col17 = C{17};
col18 = C{18}; %tags

training = 0.1*numrows;
training = round(training); % 10% training set 

col19=zeros(7,1,training);

for n=1:numrows  % this loop assigns numerical tags
    if col18(n)==1
	col19(1,1,n)=1;
    elseif col18(n)==2
	col19(2,1,n)=1;
    elseif col18(n)==3
	col19(3,1,n)=1;
    elseif col18(n)==4
	col19(4,1,n)=1;
    elseif col18(n)==5
	col19(5,1,n)=1;
    elseif col18(n)==6
	col19(6,1,n)=1;
    elseif col18(n)==7
	col19(7,1,n)=1;
    end
end

constraint = zeros(numrows+1,1);
constraint = constraint + 1;

x = [constraint col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 col13 col14 col15 col16 col17];  %data row vector

lhs = zeros(18,18,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(18,7,training);
rhstemp = [1:18];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*col19(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(18,7);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(7,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));
end

col20 = zeros(7,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col20(:,:,i)=[1,0,0,0,0,0,0]';
    elseif results(i)==2
        col20(:,:,i)=[0,1,0,0,0,0,0]';
    elseif results(i)==3
	col20(:,:,i)=[0,0,1,0,0,0,0]';
    elseif results(i)==4
	col20(:,:,i)=[0,0,0,1,0,0,0]';
    elseif results(i)==5
	col20(:,:,i)=[0,0,0,0,1,0,0]';
    elseif results(i)==6
	col20(:,:,i)=[0,0,0,0,0,1,0]';
    else 
        col20(:,:,i)=[0,0,0,0,0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = col19(:,:,i)-col20(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct




%LLS classifier for the simplified zoo dataset

B = fopen('zoo.txt','r');

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
B = fopen('zoo.txt','r');
C = textscan(B,'%s ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5};
col6 = C{6};
col7 = C{7};
col8 = C{8};
col9 = C{9};
col10 = C{10};
col11 = C{11};
col12 = C{12};
col13 = C{13};
col14 = C{14};
col15 = C{15};
col16 = C{16};
col17 = C{17};
col18 = C{18}; %tags

training = 0.5*numrows;
training = round(training); % 50% training set

col19=zeros(7,1,training);

for n=1:numrows  % this loop assigns numerical tags
    if col18(n)==1
	col19(1,1,n)=1;
    elseif col18(n)==2
	col19(2,1,n)=1;
    elseif col18(n)==3
	col19(3,1,n)=1;
    elseif col18(n)==4
	col19(4,1,n)=1;
    elseif col18(n)==5
	col19(5,1,n)=1;
    elseif col18(n)==6
	col19(6,1,n)=1;
    elseif col18(n)==7
	col19(7,1,n)=1;
    end
end

constraint = zeros(numrows+1,1);
constraint = constraint + 1;

x = [constraint col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 col13 col14 col15 col16 col17];  %data row vector

lhs = zeros(18,18,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(18,7,training);
rhstemp = [1:18];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*col19(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(18,7);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(7,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));
end

col20 = zeros(7,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col20(:,:,i)=[1,0,0,0,0,0,0]';
    elseif results(i)==2
        col20(:,:,i)=[0,1,0,0,0,0,0]';
    elseif results(i)==3
	col20(:,:,i)=[0,0,1,0,0,0,0]';
    elseif results(i)==4
	col20(:,:,i)=[0,0,0,1,0,0,0]';
    elseif results(i)==5
	col20(:,:,i)=[0,0,0,0,1,0,0]';
    elseif results(i)==6
	col20(:,:,i)=[0,0,0,0,0,1,0]';
    else 
        col20(:,:,i)=[0,0,0,0,0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = col19(:,:,i)-col20(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct
