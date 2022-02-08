% Generate mean and covariance matrix for each categories relative scores.
% Created by Joe Ellis for the Reproduction Code Class
% Reproducing Relative Attributes

function [means, Covariances] = meanandvar_forcat(Training_Samples,unseen,category_order,num_classes, looseness_constraint)

% The looseness constraint should be the looseless-1
looseness_constraint = looseness_constraint - 1;

% variables
% means = 2-d matrix each row is a mean of the labels should be 8x6 rows
% Covariances = 3-d matrix.  Should be 6x6x8 to finish this work.

% means of the set ups
% Create the list of seen categories
seen = [];
seen_index = 1;
for z = 1:num_classes
    if ismember(z,unseen) == 0
        seen(seen_index) = z;
        seen_index = seen_index + 1;
    end
end

% now we have the seen categories, and we want to find the mean and
% covariance of each of these values.

% set up the means and covariances that we want to find
means = zeros(1,size(Training_Samples,2),size(Training_Samples,3));
Covariances = zeros(size(Training_Samples,2),size(Training_Samples,2),size(Training_Samples,3));

for k = 1:length(seen)
    
    % Get the seen variable index
    class = seen(k);
    
    % Find the means of the seen a
    means(:,:,class) = mean(Training_Samples(:,:,class));
    
    % for loop to iterate over the sections of the training_samples
    Covariances(:,:,class) = cov(Training_Samples(:,:,class));
end

% Now we need to find the average covariance mat for all the seen samples
AVG_COV = sum(Covariances,3)/length(seen);


% Now we need to set up the mean and covariance for all of the unseen
% variables

% Now we have to find the average distance between the means 

dm = zeros(1,size(category_order,1));

for j = 1:size(category_order,1)
    % This section finds the means and sorts the average distance between
    % the neightbors
    sorted_means = sort(nonzeros(means(1,j,:)));
    diff = 0;
    for z = 1:length(sorted_means)-1
        diff = diff + abs(sorted_means(z)-sorted_means(z+1));
    end
    dm(j) = diff/(length(seen)-1);
end

disp('The differences between the elements for each attribute');
dm

% We need to create a category ordering of only the categories available
% not the unseen categories
for j = 1:length(seen)
    there = seen(j);
    new_category_order(:,j) = category_order(:,there);
end
        

for k = 1:length(unseen)
    % This is the unseen class
    class = unseen(k);
   
    % now we have to go through every attribute within this section of the
    % code to figure out 
    for j = 1:size(new_category_order,1)
        attr_rank = category_order(j,class);
        
        % Now get the max and min of that particular ranking
        [max_rank max_idx] = max(new_category_order(j,:));
        [min_rank min_idx] = min(new_category_order(j,:));
        
        %if ismember(attr_rank,new_category_order(j,:)) == 1;
        %    vect = (attr_rank == new_category_order(j,:));
        %    idx = find(vect);
        %    means(1,j,class) = means(1,j,seen(idx(1)));
            %disp(means(1,j,class));
            
            
        if attr_rank > max_rank;
            % Do some stuff
            max_mean = means(1,j,seen(max_idx(1)));
            means(1,j,class) = max_mean + dm(j);
            %disp(means(1,j,class));
            
        elseif attr_rank == max_rank
            % Do some stuff
            new_rank = attr_rank - 1;
            idx = find(new_category_order(j,:) == new_rank);
            if isempty(idx) == 0    
                one_less_mean = means(1,j,seen(idx(1)));
                means(1,j,class) = one_less_mean + dm(j);
            else
                max_mean = means(1,j,seen(max_idx(1)));
                means(1,j,class) = max_mean;
            end
                
            
        elseif attr_rank < min_rank
            % Do some stuff
            % Now we have to find the average distances between the means
             min_mean = means(1,j,seen(min_idx(1)));
             means(1,j,class) = min_mean - dm(j);
             %disp(means(1,j,class));
             
        elseif attr_rank == min_rank
            % Do some stuff
            % Now we have to find the average distances between the means
             new_rank = attr_rank + 1;
             idx = find(new_category_order(j,:) == new_rank);
             if isempty(idx) == 0    
                one_more_mean = means(1,j,seen(idx(1)));
                means(1,j,class) = one_more_mean - dm(j);
             else
                min_mean = means(1,j,seen(min_idx(1)));
                means(1,j,class) = min_mean;
             end
            
        else
            % Find the index of the elements one above and below those
            % elements
            row_vec = new_category_order(j,:);
            min_cand = row_vec < attr_rank - looseness_constraint;
            value = 0;
            min_use_index = 1;
            for a = 1:length(min_cand)
                if min_cand(a) == 1
                    if row_vec(a) > value;
                        min_use_index = a;
                        value = row_vec(a);
                    end
                end
            end
            lower_u = means(1,j,seen(min_use_index));
            
            % Here we have the values for the max used
            max_cand = row_vec > attr_rank + looseness_constraint;
            value = 9;
            max_use_index = 1;
            for a = 1:length(max_cand)
                if max_cand(a) == 1
                    if row_vec(a) < value;
                        max_use_index = a;
                        value = row_vec(a);
                    end
                end
            end
            higher_u = means(1,j,seen(max_use_index));
            
            % This solves for the mean for this class
            means(1,j,class) = (higher_u + lower_u)/2;
            %disp(means(1,j,class));
        end
        
        % Give it the average covariance of all of the elements
        Covariances(:,:,class) = AVG_COV;
    end
end


% I need to see what is getting messed up here

for k = 1:length(unseen)    
    % Get the seen variable index
    class = unseen(k);
    % Find the means of the seen a
    %means(:,:,class) = mean(Training_Samples(:,:,class));
    truemean = mean(Training_Samples(:,:,class));
    % for loop to iterate over the sections of the training_samples
    %Covariances(:,:,class) = cov(Training_Samples(:,:,class));
    
end
% matlab is returning non positive definite matrices for me.  Therefore, I
% need to add a bit to the diagonal.

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



