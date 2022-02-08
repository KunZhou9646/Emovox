function w = ranksvm_with_sim(X_,OMat,SMat,Costs_for_O,Costs_for_S,w,opt)
% W = RANKSVM(X,A,C,W,OPT)
% Solves the Ranking SVM optimization problem in the primal (with quatratic
%   penalization of the training errors).  
%
% X contains the training inputs and is an n x d matrix (n = number of points).
% A is a sparse p x n matrix, where p is the number of preference pairs.
%   Each row of A should contain exactly one +1 and one -1
%   reflecting the indices of the points constituing the pairs. 
% C is a vector of training error penalizations (one for each preference pair). 
%
% OPT is a structure containing the options (in brackets default values):
%   lin_cg: Find the Newton step, by linear conjugate gradients [0]
%   iter_max_Newton: Maximum number of Newton steps [20]
%   prec: Stopping criterion
%   cg_prec and cg_it: stopping criteria for the linear CG.
 
% Copyright Olivier Chapelle, olivier.chapelle@tuebingen.mpg.de
% Last modified 25/08/2006  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was modified  by Joe Ellis -- Columbia University DVMM Lab
% https://sites.google.com/site/joeelliscolumbiauniversity/
% Code was modified to reproduce the results from Relative Attributes paper
% by Grauman and Parakh.  The variables that are used here can be seen
% below.
%
% Variables:
% X contains the training inputs and is an n x d matrix (n = number of points).
% OMat is the matrix that holds the pairs.  It is set up in the same way
%   that A is described above.a sparse p x n matrix, where p is the number 
%   of preference pairs. Each row of A should contain exactly one +1 and 
%   one -1 reflecting the indices of the points constituing the pairs.
%   The plus one is the attribute that represents the images that have more
%   of an attribute -1 is the one that does not.
% SMat is the matrix that holds the pairs.  It is set up in the same way
%   that A is described above.a sparse p x n matrix, where p is the number 
%   of preference pairs. Each row of A should contain exactly one +1 and 
%   one -1 reflecting the indices of the points constituing the pairs.
%   We still have to have one attribute that has the +1 and -1 to make this
%   work based on the implementation below.
% Costs_for_O this has the value of the cost for each preference pair
% Costs_for_S this has the value of the cost for each preference pair
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  

    oldO = OMat;
    oldS = SMat;

  if size(OMat,1) == 1
      OMat = [];
  end
  
  if size(SMat,1) == 1
      SMat = [];
  end

  global X A
  X = X_; A = OMat; S = SMat; % To avoid passing theses matrices as arguments to subfunctions

  if nargin < 7       % Assign the options to their default values
    opt = [];
  end;
  if ~isfield(opt,'lin_cg'),            opt.lin_cg = 0;                    end;
  if ~isfield(opt,'iter_max_Newton'),   opt.iter_max_Newton = 10;          end;  
  if ~isfield(opt,'prec'),              opt.prec = 1e-7;                   end;  
  if ~isfield(opt,'cg_prec'),           opt.cg_prec = 1e-3;                end;  
  if ~isfield(opt,'cg_it'),             opt.cg_it = 20;                    end;  
  
  d = size(X,2);
  n = size([A;S],1);
  
  if (d*n>1e9) & (opt.lin_cg==0)
    warning('Large problem: you should consider trying the lin_cg option')
  end;
  
  if nargin<6
    w = zeros(d,1); 
  end; 
  iter = 0;
  %out = 1-A*(X*w);
  
  %%%%%%%%%%
  % This is the line that needs to be altered to change the way that
  % ranksvm works for my purposes
  % out1 is the same exact implementation as what was originally designed
  % out2 needs to take into account the pairs that are highly similar
  
  disp('The size of the X matrix')
  disp(size(X))
  disp('The sixe of the w matrix')
  disp(size(w))
  
  % This is the section of the code that sets up the work
  if size(oldO,1) == 1
    %out1 = 1-A*(X*w);
    out2 = -(S*(X*w));
    out1 = [];
  elseif size(oldS,1) == 1
    out1 = 1-A*(X*w);
    %out2 = (S*(X*w));
    out2 = [];
  else
    out1 = 1-A*(X*w);
    out2 = -(S*(X*w));
  end 
    
  % Now concatenate the vectors together.
  out = [out1; out2];
  % We need to keep track of the pairs that are chosen for the experiments,
  % so thusly concatenate the two vectors together so they will be the same
  % for our experiments.
  A = [A;S];
  % disp('The size of the A Matrix')
  % disp(size(A))
  % C = [Costs_for_O;Costs_for_S];
  
  C = 0.1*ones(size(A,1),1);
  while 1
    iter = iter + 1;
    if iter > opt.iter_max_Newton;
      warning(sprintf(['Maximum number of Newton steps reached.' ...
                       'Try larger lambda']));
      break;
    end;
    
    [obj, grad, sv] = obj_fun_linear(w,C,out);      
    
    % Compute the Newton direction either by linear CG
    % Advantage of linear CG when using sparse input: the Hessian
    % is never computed explicitly.
    if opt.lin_cg
      [step, foo, relres] = minres(@hess_vect_mult, -grad,...
                                   opt.cg_prec,opt.cg_it,[],[],[],sv,C);
    else
      Xsv = A(sv,:)*X;
      hess = eye(d) + Xsv'*(Xsv.*repmat(C(sv),1,d)); % Hessian
      step  = - hess \ grad;   % Newton direction
      relres = 0;
    end;
    
    % Do an exact line search
    [t,out] = line_search_linear(w,step,out,C);
    
    w = w + t*step;
    fprintf(['Iter = %d, Obj = %f, Nb of sv = %d, Newton decr = %.3f, ' ...
             'Line search = %.3f, Lin CG acc = %.4f     \n'],...
            iter,obj,sum(sv),-step'*grad/2,t,relres);
    
    if -step'*grad < opt.prec * obj  
      % Stop when the Newton decrement is small enough
      break;
    end;
  end;   
  
  
function [obj, grad, sv] = obj_fun_linear(w,C,out)
  % Compute the objective function, its gradient and the set of support vectors
  % Out is supposed to contain 1-A*X*w
  global X A
  out = max(0,out);
  obj = sum(C.*out.^2)/2 + w'*w/2; % L2 penalization of the errors
  grad = w - (((C.*out)'*A)*X)'; % Gradient
  sv = out>0;  
  
  
function y = hess_vect_mult(w,sv,C)
  % Compute the Hessian times a given vector x.
  global X A
  y = w;
  z = (C.*sv).*(A*(X*w));  % Computing X(sv,:)*x takes more time in Matlab :-(
  y = y + ((z'*A)*X)';
  
  
function [t,out] = line_search_linear(w,d,out,C) 
  % From the current solution w, do a line search in the direction d by
  % 1D Newton minimization
  global X A
  t = 0;
  % Precompute some dots products
  Xd = A*(X*d);
  wd = w'*d;
  dd = d'*d;
  while 1
    out2 = out - t*Xd; % The new outputs after a step of length t
    sv = find(out2>0);
    g = wd + t*dd - (C(sv).*out2(sv))'*Xd(sv); % The gradient (along the line)
    h = dd + Xd(sv)'*(Xd(sv).*C(sv)); % The second derivative (along the line)
    t = t - g/h; % Take the 1D Newton step. Note that if d was an exact Newton
                 % direction, t is 1 after the first iteration.
    if g^2/h < 1e-10, break; end;
  end;
  out = out2;
