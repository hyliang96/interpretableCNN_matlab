%% Selections of the dataset, model, and category
% the arguements we tried
    % ilsvrcanimalpart n02118333 false vgg-vd-16 0.8
    % vocpart bird  false vgg-vd-16 0.8
    % cub200 cub200 false vgg-vd-16 0.8

% Choices of the dataset. You may select 'ilsvrcanimalpart', 'vocpart', or 'cub200'.
dataset='vocpart'; %'vocpart'; 'cub200';

% Learn a CNN for multi-class classification or single-class classification
isMultiClassClassification=true; %true;

% choose a class for single class classification: classify this class from other classes
% If you select dataset='ilsvrcanimalpart', then you may choose of the following categories.
    % 'n01443537','n01503061','n01639765','n01662784','n01674464','n01882714','n01982650',
    % 'n02084071','n02118333','n02121808','n02129165','n02129604','n02131653','n02324045',
    % 'n02342885','n02355227','n02374451','n02391049','n02395003','n02398521','n02402425',
    % 'n02411705','n02419796','n02437136','n02444819','n02454379','n02484322','n02503517',
    % 'n02509815','n02510455'
% If you select dataset='vocpart', then you may choose of the following categories.
    % 'bird','cat','cow','dog','horse','sheep'
% If you select dataset='cub200', then you need to choose
    % categoryName='cub200'.
% categoryName='cub200';

% Choices of the networks. You may select 'vgg-vd-16', 'alexnet', 'vgg-m', or 'vgg-s'.
model='vgg-vd-16'; % 'alexnet'; 'vgg-m'; 'vgg-s';

% droprout rate;when using a small number of training samples, avoid over-fitting.
dropoutRate=0.8; %0.5; 0.6; 0.7; 0.8; 0.9;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Network configurations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setup_matconvnet(); % setup matconvnet

switch(dataset) % setup the dataset
    case 'ilsvrcanimalpart'
        setup_ilsvrcanimalpart();
    case 'vocpart'
        setup_vocpart();
    case 'cub200'
        setup_cub200();
    otherwise
        errors('Cannot find the target dataset.');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning an interpretable CNN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(isMultiClassClassification)
    if(strcmp(dataset,'vocpart'))
        categoryName={'bird','cat','cow','dog','horse','sheep'};
        lossType='ourloss_logistic'; % 'ourloss_softmaxlog';
        learn_icnn_multiclass(model,categoryName,lossType,dropoutRate);
    end
else
    learn_icnn(model,categoryName,dropoutRate);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Showing results and evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(isMultiClassClassification)
    if(strcmp(dataset,'vocpart'))
        categoryName={'bird','cat','cow','dog','horse','sheep'};
    end
end
showResults(categoryName,isMultiClassClassification,model,dataset);

