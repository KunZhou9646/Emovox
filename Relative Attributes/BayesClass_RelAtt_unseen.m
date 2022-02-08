% Bayesian Classification of the Relative Attributes
% Created by Joe Ellis for Reproducible Codes Class
% This function takes in the means and Covariances Matrices of each class
% and then classifies the variables based on their values

function accuracy = BayesClass_RelAtt_unseen(predicts,ground_truth,means,Covariances,used_for_training,unseen)

% Variables
% predicts = the values that need to be predicted and classified these are
%   the relative predictions
% ground_truth = the real class_labels they are a 2668 vector;
% means = 1x6x8 matrix of the covariances and the means
% Covariances = 6x6x8 matrix fo the covariances

% This is for tracking the accuracy of the set up
correct = 0;
total = 0;

% Now do a for loop for each of the predicts variables
for j = 1:length(predicts)
    % We don't want to use the variables that are used for training so
    % let's skip those in test
    if used_for_training(j) == 0 && ismember(ground_truth(j),unseen) == 1
        
        %{
        % This is for debug purposes
        if ismember(ground_truth(j),unseen) == 1
            disp('This is an unseen variable, and is of class');
            disp(ground_truth(j));
        end
        %}
        
        % For each of the categories find the guassian probability of the
        % each variable and each point
        best_prob = 0;
        for k = 1:size(means,3)
            
            % Add a bit of value to the Covariances to insure they are
            % positive definite
            Cov_ex = Covariances(:,:,k) + eye(size(Covariances,1)).*.00001;
            prob = mvnpdf(predicts(j,:),means(:,:,k),Cov_ex);
            
            % Debug Purposes
            % let's calc the distance from the prediction values of the
            % ranking to the predicted means of the values
            %{
            distance = pdist([predicts(j,:);means(:,:,k)],'euclidean');
            disp('This is the class: ');
            disp(k);
            disp('This is the distance: ')
            disp(distance);
            disp('The predicted values');
            disp(predicts(j,:));
            disp('The mean values of this variable');
            disp(means(:,:,k));
            %}
            
            if prob > best_prob
                best_prob = prob;
                app_label = k;
            end
        end
        
        % Now see if the label is the same as the ground truth label;
        if ground_truth(j) == app_label;
            correct = correct + 1;
        end
        
        % Add to the total numbers of predicts that are analyzed
        total = total + 1;
    end
end

accuracy = correct/total;
    