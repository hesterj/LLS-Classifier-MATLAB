%LLS classifier for the simplified iris dataset

B = fopen('basiciris.txt','r');

%%%
% This computes the number of rows
 c = 0;
fgetl(B);
while ~feof(B)
fgetl(B);
c = c+1;
end
numrows = c+1;

%%%
B = fopen('basiciris.txt','r');
C = textscan(B,'%f ,%f ,%f ,%f ,%s');
col1 = C{1};
col2 = C{2};
col3 = C{3};
col4 = C{4};
col5 = C{5}; %tags

training = 4;  %  specify first n rows to be used as training.  
%For training purposes I mixed up the original file a bit so that all the
%different species of iris would be represented in the training set. 

col6 = [1:35];
col6=col6';

irisvirginica = 'Iris-virginica';


for n=1:numrows  % this loop assigns 1 to virginica, -1 otherwise
    str1 = col5(n);
    tf = strcmp(str1,irisvirginica);
    if tf==1
        col6(n)=1;
    else
        col6(n)=-1;     
    end
end

x = [col1 col2 col3 col4];  %data row vector

lhs = zeros(4,4,training); %these are some initiializations
lhstemp = 0;
rhs = rand(4,training);
rhstemp = [1:4];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,i)=col6(i)*x(i,:);   %compute rhs of w-hat
end

rhsfinal=sum(rhs,2);  %sum of rhs

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = [1:numrows];
    
for i=1:numrows    %  w'*x
    prediction(i) = w2*x(i,:)'; 
    
    if prediction(i)>0
        prediction(i) = 1;
    else 
        prediction(i) = -1;
    end
    
end
prediction=prediction';

finalcomparison = [col6 prediction] %look at this to see accuracy.  rhs is what we predicted, lhs is the actual tag.
    
