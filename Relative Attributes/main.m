% Script created to create the graphs that we want to create for the osr
% dataset with the Relative attributes method

% Train the ranking function should be right here
% This portion of the code needs to have some ground truth data labeled and
% the relative similarities finished

% Clear all the data before running the script
clear all;


% These are the num of unseen classes and training images per class
num_unseen = 0;
trainpics = 300; %need to change to 300 Kun
num_iter = 10;
held_out_attributes = 0;
emotion_categories = 2;
num_attributes = 2;
labeled_pairs = 1;
looseness_constraint = 1;
% This is the number of iterations we want to do
accuracy = zeros(1,num_iter);
 
%spk_list = {'0011','0012','0013','0014','0015','0016','0017','0018','0019','0020'}  %[0011--0020]
emo_list = {'Neutral', 'Happy', 'Angry', 'Sad'}
spk_list = {'0013'}  %[0011--0020]
%emo_list = {'Happy', 'Angry', 'Sad'}
for spk_num = 1:size(spk_list,2)
    for emo_num = 1:size(emo_list,2)
        spk_tag = spk_list(spk_num)
        emo_tag = emo_list(emo_num)
        spk_tag_str = string(spk_tag(1)) %0011
        emo_tag_str = string(emo_tag(1)) %Happy
        % output file (score)
        score_path = strcat(spk_tag_str, '_Surprise_' , emo_tag_str , '_Score.csv') %"0011_Neutral_Happy_Score.csv"
        fopen(score_path,'wt')
        % input file (feature extracted by OpenSmile)
        
        %osr_gist_Mat = csvread(strcat(spk_tag_str, '/', spk_tag_str, '_Neutral_', emo_tag_str, '.csv'),1,2); %"0011/0011_Neutral_Happy.csv"
        
        osr_gist_Mat = csvread(strcat(spk_tag_str, '_Surprise_', emo_tag_str, '.csv'),1,2); %"0011_Neutral_Happy.csv" %700x384
        
        % Debug
        % osr_gist_Mat = csvread('0010_Neutral_Angry.csv',1,2);
              
        
        used_for_training = csvread('used_for_training_kun.csv'); %debug Kun
        % class_names = {'Angry','Neutral'};
        class_labels = csvread('classlabel.csv');
        relative_ordering = [2 1; 3 1];
        category_order = relative_ordering;
 
        osr_gist_Mat_normal = mapminmax(osr_gist_Mat',0,1); %normalization
        osr_gist_Mat = osr_gist_Mat_normal'; %384x700

      

        for iter = 1:num_iter

            % Create a random list of unseen images
            unseen = randperm(emotion_categories,num_unseen);
            [O,S] = Create_O_and_S_Mats_2D(category_order,used_for_training,class_labels,emotion_categories,unseen,trainpics,labeled_pairs);
            % Now we need to train the ranking function, but we have some values in the
            % matrices that will not correspond to the anything becuase some attributes
            % will have more nodes with similarity.

            weights = zeros(384,num_attributes); %384x2
            for l = 1:num_attributes

                % Find where each O and S matrix stops having values for each category
                % matrix section

                % Find when the O matrix for this dimension no longer has real values

                for j = 1:size(O,1)
                    O_length = j;
                    if ismember(1,O(j,:,l)) == 0;
                        break;
                    end
                end

                % Find when the S matrix for this dimension no longer has real values.
                for j = 1:size(S,1)
                    S_length = j;
                    if ismember(1,S(j,:,l)) == 0;
                        break;
                    end
                end

                % Now set up the cost matrices both are initialized to 0.1 in the
                % Relative Attributes paper from 2011;
                Costs_for_O = .1*ones(O_length,1);
                Costs_for_S = .1*ones(S_length,1);

                if O_length > 1
                    w = ranksvm_with_sim(osr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
                    %w = testrank(osr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
                    weights(:,l) = w*2;
                else
        %         exit
                % Re-Do the ranking and start over, because we chose category pairs
                % that did not have the O matrix for a given attribute.

                % This function creates the O and S matrix used in the ranking algorithm
                [O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,3,unseen,trainpics,labeled_pairs);

                % initialize the weights matrix that will be learned for ranking
        %         weights = zeros(384,6);
                weights = zeros(384,num_attributes);

                % re-do the creation of the O and S matrix
                l = 1;
                disp('We had to redo the O and S matrix ranking, Pairs chosen were all similar for an attribute');
                end
            end

            % here we want to choose to take out some of the weights for each
            % attribute and also the category order
            if held_out_attributes ~= 0
                rand_atts = randperm(6,6-held_out_attributes);
                for j = 1:length(rand_atts);
                    new_weights(:,j) = weights(:,rand_atts(j));
                    new_cat_order(j,:) = category_order(rand_atts(j),:);
                    new_relative_att_predictor(:,j) = relative_att_predictor(:,rand_atts(j));
                end
            else
                new_cat_order = category_order;
                new_weights = weights;
        %         new_relative_att_predictor = relative_att_predictor;
            end


            % Get the predictions based on the outputs from rank svm
            % Use there trained data
            % relative_att_predictions = feat*new_relative_att_predictor;
            % Use my trained data
            relative_att_predictions = osr_gist_Mat*new_weights;

            % Seperate the training samples from the other training samples
            Train_samples = GetTrainingSample_per_category(relative_att_predictions,class_labels,used_for_training);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Debug %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Calculate the means and covariances from the samples
            [means, Covariances] = meanandvar_forcat(Train_samples,[],new_cat_order,emotion_categories,looseness_constraint);

            % This is for debug to find the problem with the unseen scategories
            means_unseen = meanandvar_forcat(Train_samples,unseen,new_cat_order,emotion_categories,looseness_constraint);

            % This section will find the difference between the values of the means
            disp('The unseen values are')
            unseen
            disp('Actual Means');
            means
            disp('Difference between the means');
            disp(means_unseen - means);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Classify the predicted features from the system
            accuracy(iter) = BayesClass_RelAtt(relative_att_predictions,class_labels,means_unseen,Covariances,used_for_training,unseen);
            disp('unseen accuracy for means found');
            disp(accuracy(iter))

            other_acc = BayesClass_RelAtt_unseen(relative_att_predictions,class_labels,means_unseen,Covariances,used_for_training,unseen);
            disp('unseen accuracy for derived means')
            disp(other_acc);
            disp('The relative ordering of the attributes for each image');
        %     category_order
        end

            total_acc = mean(accuracy);

        relative_att_predictions_norm = normalize(relative_att_predictions(:,1),'range')
        csvwrite(score_path, relative_att_predictions_norm)   % (350:700) 
        disp('The accuracy of this calculation: ');
        disp(total_acc);
        disp('------------- ok -------------- ');
    end
end
