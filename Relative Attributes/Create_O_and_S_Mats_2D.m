% This function takes in a matrix with attribute relations in numeric
% categories, and the features extracted from the available test images,
% class labels of all of the images, and the images that are used for training, 
% and outputs the O and S matrix used with rank_with_sim rank svm implementation for training
% Created by Joe Ellis -- PhD Candidate Columbia University

function [O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,num_classes,unseen,trainpics,att_combos)

% INPUTS
% category_order = the order of the relative attributes of each category.
% used_for_training = A vector the length of the samples, a 1 denotes that
%   this sample should be used for training, and a 0 is a test image 
% class_labels = the class_labels of each sample
% num_classes = the total number of classes
% unseen = A vector containing the class labels that are unseen
% trainpics = The number of pictures used for training
% att_combos = The number of category pairs that will be used for training

% OUTPUTS
% O = The matrix that is used as input to the ranking function, this matrix
%   for each row in the matrix contains one 1 and -1 element used for
%   training.
% S = The similarity matrix is the same as the O matrix, but contains
%   samples that have the same score for a given attribute

% num_categories = 6;
num_categories = 2;

% This matrix holds the index of the training samples for each class
train_by_class = zeros(num_classes,trainpics);

% Set up the O and S Mats
O = zeros((trainpics^2)*att_combos,length(class_labels),num_categories);
S = zeros((trainpics^2)*att_combos,length(class_labels),num_categories);

% Create the train_by_class matrix to create the o and s matrix for ranking
% functions
index = ones(1,num_classes);
for j = 1:length(used_for_training)
    
    % pick out the images that are going to be used_for_training
    if used_for_training(j) == 1;
        switch class_labels(j)
            case 1
                train_by_class(1,index(1)) = j;
                index(1) = index(1) + 1;
            case 2
                train_by_class(2,index(2)) = j;
                index(2) = index(2) + 1;
            case 3
                train_by_class(3,index(3)) = j;
                index(3) = index(3) + 1;
            case 4
                train_by_class(4,index(4)) = j;
                index(4) = index(4) + 1;
            case 5
                train_by_class(5,index(5)) = j;
                index(5) = index(5) + 1;
            case 6
                train_by_class(6,index(6)) = j;
                index(6) = index(6) + 1;
%             case 7
%                 train_by_class(7,index(7)) = j;
%                 index(7) = index(7) + 1;
%             case 8
%                 train_by_class(8,index(8)) = j;
%                 index(8) = index(8) + 1;
        end
    end
end

% Now we have the train_by_class matrix which has the training images for
% each seperate variable.  Now we are going to write the code as to how we
% are going to create the o matrix and s matrix

% create the elements to index the o matrix and s matrix
num_images = length(used_for_training);
s_index = ones(1,num_categories);
o_index = ones(1,num_categories);

% Create the list of seen classes
seen = [];
seen_index = 1;
for z = 1:num_classes
     if (ismember(z,unseen) == 0)
         seen(seen_index) = z;
         seen_index = seen_index + 1;
     end
end

% Now we need to get the mix of the 4 categories that should all be together
% This section randomly assigns two seen categoires as the category pairs
% for training.
combo1 = floor(1+((rand(1,att_combos)).*length(seen)));
combo2 = floor(1+((rand(1,att_combos)).*length(seen)));
for z = 1:att_combos
    test_combos(z,1) = seen(combo1(z));
    test_combos(z,2) = seen(combo2(z));
    
    % We should not compare two categoires to each other, and this section
    % does not allow that to happen
    while test_combos(z,1) == test_combos(z,2)
        test_combos(z,2) = floor(1+((rand(1).*length(seen))));
    end
    
    % We also don't want to choose the same combination twice.
    % This function will prevent that from happening by checking the
    % previous combos and making sure they are not the same as the current.
    r = 1;
    while r < z 
        if ismember(0,(sort(test_combos(r,:)) == sort(test_combos(z,:))))
            r = r + 1;
        else
            test_combos(z,1) = seen(floor(1+((rand(1).*length(seen)))));
            test_combos(z,2) = seen(floor(1+((rand(1).*length(seen)))));
            r = 1;
            
            % Make sure that we are not comparing the two values together.
            while test_combos(z,1) == test_combos(z,2)
                test_combos(z,2) = seen(floor(1+((rand(1).*length(seen)))));
            end
        end
    end
            
            
    
end

% Now loop through each attribute pairing that we have and generate the O
% and S matrices.

% Now display which classes we are using for training
disp('These are the category pairs for RankSVM training')
disp(test_combos)
for z = 1:size(test_combos,1)
    on_class = test_combos(z,1);
    compared_class = test_combos(z,2);
    
    % Do this for every attribute
    for l = 1:2
        % If the two relative comparisons are equal add this pairing to
        % the S matrix
        if category_order(l,on_class) == category_order(l,compared_class)
            % Now perform this for every training picture for each class
            for j = 1:trainpics
                for i = 1:trainpics
                    S_row = zeros(1,num_images);
                    S_row(train_by_class(on_class,j)) = 1;
                    S_row(train_by_class(compared_class,i)) = -1;
                    S(s_index(l),:,l) = S_row;
                    s_index(l) = s_index(l) + 1;
                end
            end
            
            % If the relative comparison of the on_class is greater than
            % that of the compared class
        elseif category_order(l,on_class) > category_order(l,compared_class)
            % Now perform this for every training picture for each class
            for j = 1:trainpics
                for i = 1:trainpics
                    O_row = zeros(1,num_images);
                    O_row(train_by_class(on_class,j)) = 1;
                    O_row(train_by_class(compared_class,i)) = -1;
                    O(o_index(l),:,l) = O_row;
                    o_index(l) = o_index(l) + 1;
                end
            end
            
            % If the relative comparison of the new class is greater than
            % that of the compared class
        elseif category_order(l,on_class) < category_order(l,compared_class)
            % Now perform this for every training picture for each class
            for j = 1:trainpics
                for i = 1:trainpics
                    O_row = zeros(1,num_images);
                    O_row(train_by_class(on_class,j)) = -1;
                    O_row(train_by_class(compared_class,i)) = 1;
                    O(o_index(l),:,l) = O_row;
                    o_index(l) = o_index(l) + 1;
                end
            end
            
        end
    end
end

end