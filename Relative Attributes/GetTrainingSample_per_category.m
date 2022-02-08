% Seperate the examples that are used for training in comparison to the
% seen and unseen variables
% Created by Joe Ellis for the Reproducible Codes Class
function Train_samples = GetTrainingSample_per_category(predicts,class_labels,used_for_training)

% Variables
% predicts = the values for each image that have been predicted using the
%   ranking algorithm devised
% class_labels = the ground truth label of each class
% used_for_training = If this image should be used in the training of the
%   model 

% Train Samples is a 3-D matrix of the training variables that will be used
% to train the gaussian distributions of the material for what we are
% doing.

Train_samples = zeros(30,size(predicts,2),8);
index = ones(1,8);

% Set up the matrices for training
for j = 1:length(predicts);
    if used_for_training(j) == 1;
        switch class_labels(j)
            case 1
                Train_samples(index(1),:,1) = predicts(j,:);
                index(1) = index(1) + 1;
            case 2
                Train_samples(index(2),:,2) = predicts(j,:);
                index(2) = index(2) + 1;
            case 3
                Train_samples(index(3),:,3) = predicts(j,:);
                index(3) = index(3) + 1;
            case 4
                Train_samples(index(4),:,4) = predicts(j,:);
                index(4) = index(4) + 1;
            case 5
                Train_samples(index(5),:,5) = predicts(j,:);
                index(5) = index(5) + 1;
            case 6
                Train_samples(index(6),:,6) = predicts(j,:);
                index(6) = index(6) + 1;
            case 7
                Train_samples(index(7),:,7) = predicts(j,:);
                index(7) = index(7) + 1;
            case 8
                Train_samples(index(8),:,8) = predicts(j,:);
                index(8) = index(8) + 1;
        end
    end
end
end
    



