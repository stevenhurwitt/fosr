% Code for implementing bivariate probit extension of LFRM as described in
% Section 3.2 to predict probability of preeclampsia and low birth weight

clear;clc; 
warning off all;
tic;
load Design;
resp = [];
resp(1, :) = preecl;
resp(2, :) = lbw;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % "Design" contains:                                                   % %
% % y : simulated MAP measurements for n = 200 subjects                  % %
% % tij : simulated time points at which the MAP measurements are        % %
% %       collected                                                      % %
% % ni : number of simulated measurements per subject                    % %
% % id : subjects' id                                                    % %
% % n : number of subjects                                               % %
% % O : matrix of simulated covariates                                   % %
% % gest : simulated gestational age at delivery                         % %
% % Weight : simulated birth weigth (in grams)                           % %
% % lbw : indicator of low birth weight                                  % %
% % preecl : indicator variable for preeclampsia                         % %
% %                                                                      % %
% % The code is set to perform out of sample prediction. Specifically,   % %
% % measurements collected after the 30th week are held out for the test % %
% % set and predicted. To perform sequential predictions, change line 91 % %
% %                                                                      % %
% % Output files :                                                       % %
% % problowbweight.txt : contains the estimated probability of l.b.w.    % %
% % probpreeclampsia.txt : contains the estimated prob. of preeclampsia  % %
% % line 740 : plotting estimated trajectories                           % %
% % line 748 : computing predictive errors                               % %
% % line 792 : visualization of impact of covariates (boxplots)          % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- define global constants --- %
 
ktr = 20;
rep = 1;
r = 10;               % number of covariates
q = 10;               % number of fixed basis functions
n = 200;              % number of subjects
N = sum(ni(1:n));     % total number of measurements
m = 290;
tg = [1:290]' / 290;                      % equally spaced measurement times
X = [ones(N,1) zeros(N,q -1)];            % design matrix for data in study
Xb = [ones(290, 1) zeros(290, q -1)];     % design matrix for estimation / prediction
xi = [0 : q-1]' / (q-1);                  % kernel locations 
for h = 1 : q-1
    X(:, h +1 ) = exp(-4*(tij - xi(h)).^2);
    Xb(:, h +1 ) = exp(-4*(tg - xi(h)).^2);
end
idru = unique(id);
nrun = 25000;         % Tot. number of iteration
burn = 5000;          % Burn-in
thin = 5;             % Thin
sp = (nrun - burn)/thin;     % number of posterior samples                                 
trsize = 100;                % number of subjects in the train set (for out of sample prediction)
tstsize =  n - trsize;       % number of subjects in the test set
firstweek = zeros(n, 1);     % week when first measurement is collected
for j = 1:n
    firstweek(j) = min(tij(id == j).*290./7);
end
lastweek = zeros(n, 1);      % week when last measurement is collected
for j = 1:n
    lastweek(j) = max(tij(id == j).*290./7);
end
ni = ni';
population = idru(ni(1:n) > 1 & firstweek < 20 & lastweek > 35);
% test set chosen among subjects with at least 1 measurement in the first
% 20 weeks and at least one measurement after 35th week
tst = sort(randsample(population, tstsize));    % ID in the test sample
train = setdiff(1:n,tst)';                      % ID in the training sample
train = train'; test = tst;
tstsize = length(tst);
% save train and test ID to .txt file
dlmwrite('train.txt', train, ' ')
dlmwrite('test.txt', tst, ' ')
train = train';
tst = test;
nitrain = ni(train);                            % Tot # of observations in the training set
nitst = ni(tst); 
nitest = zeros(1, length(tst));                 % Measurements used for prediction
for j = 1:length(tst)
    h = tst(j);
    obs = tij(id == h).*290./7;
    nitest(j) = length(obs(obs < 30));          % Test set: retained only obs in the first 30 weeks
end
nitestobs = ni(tst) - nitest';                  % Test set: predict obs after 30th week

kinit = repmat(floor(log(q)*4),rep,1);          % number of factors to start with (number of columns of Lambda)
b0 = 1.5; 
b1 = 0.0001; 
epsilon = 1e-4;                                 % threshold limit
prop = 1.00;                                    % proportion of redundant elements within columns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Save posterior samples to file to avoid Matlab's memory contraints - %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('thetaout.txt', 'wt');
fidsigma = fopen('sigmaout.txt', 'wt');
fidpsi = fopen('psiout.txt', 'wt');
fidpredout = fopen('Ytpredout.txt', 'wt');
fidfactor = fopen('factorout.txt', 'wt');
fidalphaone = fopen('alphaoutone.txt', 'wt');
fidalphatwo = fopen('alphaouttwo.txt', 'wt');
fidpreecl = fopen('latentpreecl.txt', 'wt');
fidlbw = fopen('latentlbw.txt', 'wt');
fidrho = fopen('rho.txt', 'wt');
fidacc = fopen('Acceptance.txt', 'wt');
fidgammaone = fopen('gammaone.txt', 'wt');
fidgammatwo = fopen('gammatwo.txt', 'wt');
   
   
fidfirsteta = fopen('firsteta.txt', 'wt'); fidsecondeta = fopen('secondeta.txt', 'wt');
fidthirdeta = fopen('thirdeta.txt', 'wt'); fidfourtheta = fopen('fourtheta.txt', 'wt');
fidfiftheta = fopen('fiftheta.txt', 'wt'); fidsixtheta = fopen('sixtheta.txt', 'wt');
fidseventheta = fopen('seventheta.txt', 'wt'); fideighteta = fopen('eighteta.txt', 'wt');
fidnintheta = fopen('nintheta.txt', 'wt'); fidtentheta = fopen('tentheta.txt', 'wt');

fidfirstbeta = fopen('firstbeta.txt', 'wt'); fidsecondbeta = fopen('secondbeta.txt', 'wt');
fidthirdbeta = fopen('thirdbeta.txt', 'wt'); fidfourthbeta = fopen('fourthbeta.txt', 'wt');
fidfifthbeta = fopen('fifthbeta.txt', 'wt'); fidsixthbeta = fopen('sixthbeta.txt', 'wt');
fidseventhbeta = fopen('seventhbeta.txt', 'wt'); fideightbeta = fopen('eightbeta.txt', 'wt');
fidninthbeta = fopen('ninthbeta.txt', 'wt'); fidtenthbeta = fopen('tenthbeta.txt', 'wt');

fidfirstlambda = fopen('firstlambda.txt', 'wt'); fidsecondlambda = fopen('secondlambda.txt', 'wt');
fidthirdlambda = fopen('thirdlambda.txt', 'wt'); fidfourthlambda = fopen('fourthlambda.txt', 'wt');
fidfifthlambda = fopen('fifthlambda.txt', 'wt'); fidsixthlambda = fopen('sixthlambda.txt', 'wt');
fidseventhlambda = fopen('seventhlambda.txt', 'wt'); fideightlambda = fopen('eightlambda.txt', 'wt');
fidninthlambda = fopen('ninthlambda.txt', 'wt'); fidtenthlambda = fopen('tenthlambda.txt', 'wt');

fidfirstomega = fopen('firstomega.txt', 'wt'); fidsecondomega = fopen('secondomega.txt', 'wt');
fidthirdomega = fopen('thirdomega.txt', 'wt'); fidfourthomega = fopen('fourthomega.txt', 'wt');
fidfifthomega = fopen('fifthomega.txt', 'wt'); fidsixthomega = fopen('sixthomega.txt', 'wt');
fidseventhomega = fopen('seventhomega.txt', 'wt'); fideightomega = fopen('eightomega.txt', 'wt');
fidninthomega = fopen('ninthomega.txt', 'wt'); fidtenthomega = fopen('tenthomega.txt', 'wt');

fidfirstt = fopen('firstt.txt', 'wt'); fidsecondt = fopen('secondt.txt', 'wt');
fidthirdt = fopen('thirdt.txt', 'wt'); fidfourtht = fopen('fourtht.txt', 'wt');
fidfiftht = fopen('fiftht.txt', 'wt'); fidsixtht = fopen('sixtht.txt', 'wt');
fidseventht = fopen('seventht.txt', 'wt'); fideightt = fopen('eightt.txt', 'wt');
fidnintht = fopen('nintht.txt', 'wt'); fidtentht = fopen('tentht.txt', 'wt');

fidfirstdelta = fopen('firstdelta.txt', 'wt'); fidseconddelta = fopen('seconddelta.txt', 'wt');
fidthirddelta = fopen('thirddelta.txt', 'wt'); fidfourthdelta = fopen('fourthdelta.txt', 'wt');
fidfifthdelta = fopen('fifthdelta.txt', 'wt'); fidsixthdelta = fopen('sixthdelta.txt', 'wt');
fidseventhdelta = fopen('seventhdelta.txt', 'wt'); fideightdelta = fopen('eightdelta.txt', 'wt');
fidninthdelta = fopen('ninthdelta.txt', 'wt'); fidtenthdelta = fopen('tenthdelta.txt', 'wt');

fidfirsttau = fopen('firsttau.txt', 'wt'); fidsecondtau = fopen('secondtau.txt', 'wt');
fidthirdtau = fopen('thirdtau.txt', 'wt'); fidfourthtau = fopen('fourthtau.txt', 'wt');
fidfifthtau = fopen('fifthtau.txt', 'wt'); fidsixthtau = fopen('sixthtau.txt', 'wt');
fidseventhtau = fopen('seventhtau.txt', 'wt'); fideighttau = fopen('eighttau.txt', 'wt');
fidninthtau = fopen('ninthtau.txt', 'wt'); fidtenthtau = fopen('tenthtau.txt', 'wt');

fidfirstPtht = fopen('firstPtht.txt', 'wt'); fidsecondPtht = fopen('secondPtht.txt', 'wt');
fidthirdPtht = fopen('thirdPtht.txt', 'wt'); fidfourthPtht = fopen('fourthPtht.txt', 'wt');
fidfifthPtht = fopen('fifthPtht.txt', 'wt'); fidsixthPtht = fopen('sixthPtht.txt', 'wt');
fidseventhPtht = fopen('seventhPtht.txt', 'wt'); fideightPtht = fopen('eightPtht.txt', 'wt');
fidninthPtht = fopen('ninthPtht.txt', 'wt'); fidtenthPtht = fopen('tenthPtht.txt', 'wt');


% Initialize matrix of coefficients
 
thetaout = zeros(1,q*n);
omegaout = zeros(1, r*5);
betaout = zeros(1,r*5);
sigmaout = zeros(1,q);
lambdaout = zeros(1,q*5);
etaout = zeros(1,5*n);
deltaout = zeros(1, 5);
Ytpredout = zeros(1, sum(nitestobs));
gammaout = zeros(1, 50);
latentpreecl = zeros(1, n);
latentlbw = zeros(1,n);

type = '%3.3f\t';
stringa = '%3.3f\t';

for i = 2:length(thetaout) - 1
    stringa = strcat(stringa, ' ', type);
end
stringa = strcat(stringa, ' ' , '%3.3f\r\n');
clear thetaout;

stringa2 = '%3.3f\t';
for i = 2:length(betaout) - 1
    stringa2 = strcat(stringa2, ' ', type);
end
stringa2 = strcat(stringa2, ' ' , '%3.3f\r\n');
clear betaout;

stringa3 = '%3.3f\t';
for i = 2:length(sigmaout) - 1
    stringa3 = strcat(stringa3, ' ', type);
end
stringa3 = strcat(stringa3, ' ' , '%3.3f\r\n');

stringa4 = '%3.3f\t';
for i = 2:length(lambdaout) - 1
    stringa4 = strcat(stringa4, ' ', type);
end
stringa4 = strcat(stringa4, ' ' , '%3.3f\r\n');
clear lambaout;

stringa5 = '%3.3f\t';
for i = 2:length(etaout) - 1
    stringa5 = strcat(stringa5, ' ', type);
end
stringa5 = strcat(stringa5, ' ' , '%3.3f\r\n');
clear etaout;

stringa6 = '%3.3f\t';
for i = 2:length(omegaout) - 1
    stringa6 = strcat(stringa6, ' ', type);
end
stringa6 = strcat(stringa6, ' ' , '%3.3f\r\n');
clear omegaout;

stringa7 = '%3.3f\t';
for i = 2:length(Ytpredout) - 1
    stringa7 = strcat(stringa7, ' ', type);
end
stringa7 = strcat(stringa7, ' ' , '%3.3f\r\n');

stringa8 = '%3.3f\t';
for i = 2:length(deltaout) - 1
    stringa8 = strcat(stringa8, ' ', type);
end
stringa8 = strcat(stringa8, ' ' , '%3.3f\r\n');
clear deltaout;

stringa9 = '%3.3f\t';
for i = 2:length(latentpreecl) - 1
    stringa9 = strcat(stringa9, ' ', type);
end
stringa9 = strcat(stringa9, ' ' , '%3.3f\r\n');
clear latentpreecl;

stringa10 = '%3.3f\t';
for i = 2:length(gammaout) - 1
    stringa10 = strcat(stringa10, ' ', type);
end
stringa10 = strcat(stringa10, ' ' , '%3.3f\r\n');
clear gammaout;

for g = 1:rep

    disp(['start replicate','',num2str(g)]);
    disp('-----------------');

    % ---- read data ---- %
    num = 0; k=kinit(g);      % numb. of factors to start with

    % ---- end read data ---- %
  
    % --- Define hyperparameter values --- % 
    as = .5;bs = .25;                          % gamma hyperparameters for diagonal elements of inv(Sigma)
    df = 5;                                    % gamma hyperparameters for t_{ij}
    ad1 = 1.5;bd1 = 1;                         % gamma hyperparameters for delta_1
    ad2 = 1.5;bd2 = 1;                         % gamma hyperparameters delta_h, h >= 2
    adf = 1; bdf = 1;                          % gamma hyperparameters for ad1 and ad2 or df
  
    % --- Initial values --- %
    apsi = .5; bpsi = .2;                      % gamma hyperparameters for psi
    Psiinv = gamrnd(apsi, 1/(bpsi));
    sig = gamrnd(as,1/bs,q,1);                 % diagonals of sigmainv
    Sigma = diag(1./sig);                      % Sigma (inv gamma diagonal elements)
    Sigma = (Sigma + Sigma')/ 2;
    Sigmainv = inv(Sigma);
    Sigmainv = (Sigmainv + Sigmainv') / 2;
    Lambda = zeros(q,k);                       % loading matrix
    a = [norminv(.10, 0, 1) norminv(.08333, 0, 1)]';    % Hyper. means intercept latent variables
    b = inv([.75  0; 0 .75]);                           % Hyper. var. intercept latent variables
    alpha = mvnrnd(a , inv(b));
    gamma = mvnrnd(zeros(2, k), eye(k));
    rho = 0;
    L = mvnrnd(repmat(0, n, 2), eye(2))';
    Sigmatilde = [1, rho; rho 1];
    invSigtilde = inv(Sigmatilde);
    Acc = zeros(nrun, 1);                       % Acceptance probability MH

    % -- Initialize CovOmega -- %
    CovOmega = gamrnd(.5, .5, [r k]);
    
    % -- Initialize Cauchy prior on Beta coefficients -- %
    Beta = zeros(r,k);
    for l = 1:k
        Beta(:,l) = mvnrnd(zeros(1,r), diag(1./(CovOmega(:,l))));
    end
 
    InvEta = eye(k);                                                   
    eta =  mvnrnd(O*Beta, InvEta);             % latent factors   
    THETA = mvnrnd((Lambda*eta')', Sigma)';    % Basis function coefficients
    t = gamrnd(df/2,2/df,[q,k]);               % local shrinkage coefficients
    delta = ...
    [gamrnd(ad1,bd1);gamrnd(ad2,bd2,[k-1,1])]; % gobal shrinkage coefficients multilpliers
    tau = cumprod(delta);                      % global shrinkage coefficients
    Ptht = (t .* repmat(tau',[q,1]));
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ----- Start Gibbs sampling ------ %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:nrun
        
      % -- Update Lambda -- %
        Lambda = zeros(q,k);
        for j = 1:q
            Vlam1 = diag(Ptht(j,:)) + sig(j)*(eta'*eta);  
            Vlam = inv(Vlam1);
            Vlam = (Vlam + Vlam')/2;
            Elam = Vlam*sig(j)*eta'*(THETA(j,:))';                                     
            Lambda(j,:) = mvnrnd(Elam',Vlam);
        end
        k = size(Lambda,2);
       
      % -- Update tij's -- %
        t = gamrnd(df/2 + 0.5,1./(df/2 + bsxfun(@times,Lambda.^2,tau'))); % Local shrinkage coefficients
        
      % -- Update delta -- %
        ad = ad1 + q*k/2;               
        bd = bd1 + 0.5* (1/delta(1))*sum(tau'.*(sum(t.*Lambda.^2)));
        delta(1) = gamrnd(ad,1/bd);
        tau = cumprod(delta);
        for h = 2:k
            ad = ad2 + q*(k-h+1)/2;     
            temp1 = tau'.*(sum(t.*Lambda.^2));
            bd = bd2 + 0.5* (1/delta(h))*sum(temp1(h:k));
            delta(h) = gamrnd(ad,1/bd);
            tau  = cumprod(delta);
        end
        
      % -- Update precision parameters -- %
        Ptht = bsxfun(@times,t,tau');
        
      % -- Update Sigma -- %         
        THETAtil = THETA - Lambda*eta';
        sig = gamrnd(as + n/2, 1./(bs+0.5*sum(THETAtil.^2,2)));   
        Sigma = diag(1./sig);                                     
        Sigma = (Sigma + Sigma')/ 2;
        Sigmainv = inv(Sigma);
        Sigmainv = (Sigmainv + Sigmainv') / 2;
       
      % -- Update Psi -- %
        yy = y(1:N, 1);
        F = zeros(N,1);
        for h = 1:n
            F(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),1) = X(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:)*THETA(:,h);
        end
        Psiinv = gamrnd(apsi + (N/2),1./(bpsi + 0.5*sum((yy(1:N) - F).^2)));                                                                 
        psiout = Psiinv;
        
      % -- Update of Cov. Omega -- %
        for l = 1:k
           for g = 1:r
               CovOmega(g,l) = gamrnd(1, 1/(.5 + .5*(Beta(g,l))^2));
           end
        end
       
      % -- Update of Beta -- %
        Beta = zeros(r,k);
        for l = 1:k
            CovCauchy = diag(1./(CovOmega(:,l)));
            CovCauchy = inv(CovCauchy);             
            CovCauchy = (CovCauchy + CovCauchy') / 2;
            BetaPstCov = inv(CovCauchy + (O')*eye(n)*O);          
            BetaPostCov = (BetaPstCov + BetaPstCov')/2;
            
            BetaPostMean = BetaPostCov*((O')*eta(:, l)); 
            Beta(:, l) = mvnrnd(BetaPostMean', BetaPostCov); 
        end
      
      % -- Update of alpha -- % 
        babba = zeros(2,1);
        for h = 1:n
            mi = L(:,h) - gamma*(eta(h,:)');
            babba = babba + mi;
        end
        covalpha = inv(b + n*invSigtilde);
        covalpha = (covalpha + covalpha')/2;
        alpha = mvnrnd((covalpha*(b*a + invSigtilde*babba))', covalpha)'; 
        
      % -- Update of gamma -- %
        Tilde = blkdiag(invSigtilde);
        for h = 2:n
            Tilde = blkdiag(Tilde, invSigtilde);
        end
        etaone = kron(eta, [1, 0]');
        covgamma = (etaone'*Tilde*etaone + eye(k));
        covgamma = (covgamma + covgamma')/2;
        invcovgamma = inv(covgamma);
        invcovgamma = (invcovgamma + invcovgamma')/2;
        meangamma = etaone'*Tilde*kron((L - repmat(alpha, 1, n))', [1, 0]') ;
        covgamma = (covgamma + covgamma')/2;
        gamma = mvnrnd((invcovgamma*meangamma)', invcovgamma);
        clear Tilde;
        
      % -- Update of eta -- % 
        for h = 1:n
            x = X(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:);
            inner = ((1/Psiinv)*eye(ni(h)) + x*Sigma*(x'));
            inn = (inner + inner') / 2;
            invinner = inv(inn);
            invinner = (invinner + invinner') / 2;
            etavar = (Lambda')*(x')*(invinner)*x*Lambda + eye(k) + gamma'*invSigtilde *gamma;
            etav = (etavar + etavar') / 2;
            invetavar = inv(etav);
            invetar = (invetavar + invetavar')/2;
            meaneta = eye(k)*(Beta')*(O(h,:)') + (Lambda')*(x')*invinner*(y(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:)) + ...
                 + gamma'*invSigtilde*(L(:,h) - alpha);
            eta(h,:) = mvnrnd(invetar*meaneta, invetar);
        end
      
      % -- Update of rho -- %  
        rhonew = tnormrnd(rho, 1, -1, 1);
        Sigmatildenew = [1 rhonew; rhonew 1]; invSigtildenew = inv(Sigmatildenew);
        likeold = 0; likenew = 0;
        for h = 1:n
            likenew = likenew + (- (1/2)*log(det((2*pi).*Sigmatildenew)) -(1/2)*(L(:,h) - alpha - gamma*eta(h,:)')'*...
            invSigtildenew*(L(:,h) - alpha - gamma*eta(h,:)'));
            likeold = likeold + (- (1/2)*log(det((2*pi).*Sigmatilde)) -(1/2)*(L(:,h) - alpha - gamma*eta(h,:)')'*...
            invSigtilde*(L(:,h) - alpha - gamma*eta(h,:)'));
        end
        ratio = sum(likenew - likeold) + tnormpdf(rho, rhonew, 1, -1, 1) - tnormpdf(rhonew, rho, 1, -1, 1); 
        if log(unifrnd(0,1,1)) < ratio 
            rho = rhonew;
            Acc(i) = 1;
        end
        Sigmatilde = [1 rho; rho 1]; invSigtilde = inv(Sigmatilde);
        
     % -- Update latent for preeclampsia and low birth weight -- %
       for j = 1:length(train)
           h = train(j);
           if resp(1, h) == 1
              L(1, h) = tnormrnd(alpha(1) + gamma(1,:)*eta(h,:)' + rho*(L(2,h) - alpha(2) - gamma(2,:)*eta(h,:)'), sqrt(1 - rho^2), 0, Inf);
           else if resp(1,h) == 0
              L(1, h) = tnormrnd(alpha(1) + gamma(1,:)*eta(h,:)' + rho*(L(2,h) - alpha(2) - gamma(2,:)*eta(h,:)'), sqrt(1 - rho^2), -Inf, 0);
               end
           end
           if resp(2,h) == 1
              L(2, h) = tnormrnd(alpha(2) + gamma(2,:)*eta(h,:)' + rho*(L(1,h) - alpha(1) - gamma(1,:)*eta(h,:)'), sqrt(1 - rho^2), 0, Inf);
           else if resp(2,h) == 0
              L(2, h) = tnormrnd(alpha(2) + gamma(2,:)*eta(h,:)' + rho*(L(1,h) - alpha(1) - gamma(1,:)*eta(h,:)'), sqrt(1 - rho^2), -Inf, 0);
               end
           end
       end
    
       for j = 1:length(test)
           h = test(j);
           L(:,h) = mvnrnd((alpha + gamma*eta(h,:)'), Sigmatilde);
       end

   
      % -- Update Theta -- % 
        for h = 1:n
            x = X(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:);
            covfun = Sigmainv + Psiinv*(x')*x;
            covfun = (covfun + covfun') / 2;
            Invcovfun = inv(covfun);
            Invcovfun = (Invcovfun + Invcovfun') / 2;
            meanvec = (Invcovfun*(Sigmainv*(Lambda*eta(h,:)') + Psiinv*(x')*(y(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:))))';
            THETA(:,h) = mvnrnd(meanvec, Invcovfun)';
        end

      % -- Impute missing y values -- %
        Omega = Lambda*Lambda' + Sigma;
        Ytpredout = zeros(1, sum(nitestobs));  
            for j = 1:length(tst)
                h = tst(j);
        
                x = X(sum(ni(1:(h-1))) + 1 : sum(ni(1:h)),:);    
                covfun = x*Omega*x' + (1/Psiinv)*eye(ni(h));
                covfun = (covfun + covfun')/2;
                pace = inv((covfun((1:nitest(j)),(1:nitest(j)))));
                Omegaygx = covfun((nitest(j)+1) : end,(nitest(j) + 1) :end ) - covfun((nitest(j)+1) : end, (1:nitest(j)))*pace*covfun((1:nitest(j)),(nitest(j)+1):end);
                Omegaygx = (Omegaygx + Omegaygx')/2;
                
                Yt = y(id == h);
                meanvec = x*Lambda*(Beta')*(O(h,:)');
                
                muygx = meanvec((nitest(j)+1):end,1) + covfun((nitest(j)+1):end,(1 :nitest(j)))*pace*(Yt(1:nitest(j))-meanvec(1:nitest(j),1));
                Ytpredout(1,sum(nitestobs(1: (j-1))) + 1 : sum(nitestobs(1:j))) = mvnrnd(muygx',(Omegaygx));
                y(sum(ni(1:(h-1))) + nitest(j) + 1 : sum(ni(1:h)),1) = Ytpredout(1,sum(nitestobs(1: (j-1))) + 1 : sum(nitestobs(1:j)));
            end
        Ytpredout = Ytpredout';  
        
      %--------------------------------------------%
        prob = 1/exp(b0 + b1*i);                % probability of adapting
        uu = rand;
        lind = sum(abs(Lambda) < epsilon)/q;    % proportion of elements in each column less than eps in magnitude
        vec = lind >=prop;num = sum(vec);       % number of redundant columns
        adaptout(i) = uu < prob;
        if uu < min(prob,0.001*(g>=5000) + (g < 5000))
            if  i > 20 && num == 0 && all(lind < 0.995)
                k = k + 1;
                Lambda(:,k) = zeros(q,1);
                eta(:,k) = normrnd(0,1,[n,1]);
                CovOmega(:, k) = gamrnd(.5, .5, [r,1]);
                Beta(:, k) = mvnrnd(zeros(1,r),diag(1./(CovOmega(:,k))));
                gamma(:, k) = mvnrnd([0 0], eye(2));
                t(:,k) = gamrnd(df/2,2/df,[q,1]);
                delta(k) = gamrnd(ad2,1/bd2);
                tau = cumprod(delta);
                Ptht = bsxfun(@times,t,tau');
            elseif num > 0
                nonred = setdiff(1:k,find(vec));
                k = max(k - num,1);
                Lambda = Lambda(:,nonred);
                t = t(:,nonred);
                eta = eta(:,nonred);
                gamma = gamma(:, nonred);
                CovOmega = CovOmega(:, nonred);
                Beta = Beta(:, nonred);
                delta = delta(nonred);
                tau = cumprod(delta);
                Ptht = bsxfun(@times,t,tau');                    
            end
        end
        nofout(i+1) = k;
        
       % -- Save sampled values (after thinning) -- %
        if mod(i, thin) == 0 
              Gammaone = zeros(50, 1);
              Gammatwo = zeros(50, 1);
              Gammaone(1:k, 1) = gamma(1, :); Gammatwo(1:k, 1) = gamma(2, :); 
              fprintf(fidgammaone, stringa10, Gammaone);
              fprintf(fidgammatwo, stringa10, Gammatwo);
              clear Gammaone; clear Gammatwo;
              
              fprintf(fidpreecl, stringa9, L(1,:));
              fprintf(fidlbw, stringa9, L(2,:));
              
              fprintf(fidalphaone, '%3.3f\t', alpha(1));
              fprintf(fidalphatwo, '%3.3f\t', alpha(2));
              fprintf(fidrho, '%3.3f\t', rho);
              fprintf(fidacc, '%3.3f\t', Acc(i));
              
              Etaout = zeros(n*50, 1);
              Etaout(1:n*k, 1) = reshape(eta, n*k,1); 
              fprintf(fidfirsteta, stringa5, Etaout(1:n*5, 1));
              fprintf(fidsecondeta, stringa5, Etaout((n*5) + 1: 2*(n*5), 1));
              fprintf(fidthirdeta, stringa5, Etaout(2*(n*5) + 1: 3*(n*5), 1));
              fprintf(fidfourtheta, stringa5, Etaout(3*(n*5) + 1: 4*(n*5), 1));
              fprintf(fidfiftheta, stringa5, Etaout(4*(n*5) + 1: 5*(n*5), 1));
              fprintf(fidsixtheta, stringa5, Etaout(5*(n*5) + 1: 6*(n*5), 1));
              fprintf(fidseventheta, stringa5, Etaout(6*(n*5) + 1: 7*(n*5), 1));
              fprintf(fideighteta, stringa5, Etaout(7*(n*5) + 1: 8*(n*5), 1));
              fprintf(fidnintheta, stringa5, Etaout(8*(n*5) + 1: 9*(n*5), 1));
              fprintf(fidtentheta, stringa5, Etaout(9*(n*5) + 1: 10*(n*5), 1));
              clear Etaout;
              
              Betaout = zeros(r*50, 1);
              Betaout(1:r*k, 1) = reshape(Beta, r*k,1); 
              fprintf(fidfirstbeta, stringa2, Betaout(1:r*5, 1));
              fprintf(fidsecondbeta, stringa2, Betaout((r*5) + 1: 2*(r*5), 1));
              fprintf(fidthirdbeta, stringa2, Betaout(2*(r*5) + 1: 3*(r*5), 1));
              fprintf(fidfourthbeta, stringa2, Betaout(3*(r*5) + 1: 4*(r*5), 1));
              fprintf(fidfifthbeta, stringa2, Betaout(4*(r*5) + 1: 5*(r*5), 1));
              fprintf(fidsixthbeta, stringa2, Betaout(5*(r*5) + 1: 6*(r*5), 1));
              fprintf(fidseventhbeta, stringa2, Betaout(6*(r*5) + 1: 7*(r*5), 1));
              fprintf(fideightbeta, stringa2, Betaout(7*(r*5) + 1: 8*(r*5), 1));
              fprintf(fidninthbeta, stringa2, Betaout(8*(r*5) + 1: 9*(r*5), 1));
              fprintf(fidtenthbeta, stringa2, Betaout(9*(r*5) + 1: 10*(r*5), 1));
              clear Betaout;
              
              Omegaout = zeros(r*50, 1);
              Omegaout(1:r*k, 1) = reshape(CovOmega, r*k,1); 
              fprintf(fidfirstomega, stringa6, Omegaout(1:r*5, 1));
              fprintf(fidsecondomega, stringa6, Omegaout((r*5) + 1: 2*(r*5), 1));
              fprintf(fidthirdomega, stringa6, Omegaout(2*(r*5) + 1: 3*(r*5), 1));
              fprintf(fidfourthomega, stringa6, Omegaout(3*(r*5) + 1: 4*(r*5), 1));
              fprintf(fidfifthomega, stringa6, Omegaout(4*(r*5) + 1: 5*(r*5), 1));
              fprintf(fidsixthomega, stringa6, Omegaout(5*(r*5) + 1: 6*(r*5), 1));
              fprintf(fidseventhomega, stringa6, Omegaout(6*(r*5) + 1: 7*(r*5), 1));
              fprintf(fideightomega, stringa6, Omegaout(7*(r*5) + 1: 8*(r*5), 1));
              fprintf(fidninthomega, stringa6, Omegaout(8*(r*5) + 1: 9*(r*5), 1));
              fprintf(fidtenthomega, stringa6, Omegaout(9*(r*5) + 1: 10*(r*5), 1));
              clear Omegaout;
              
              Lambdaout = zeros(q*50, 1);
              Lambdaout(1:q*k, 1) = reshape(Lambda, q*k,1); 
              fprintf(fidfirstlambda, stringa4, Lambdaout(1:q*5, 1));
              fprintf(fidsecondlambda, stringa4, Lambdaout((q*5) + 1: 2*(q*5), 1));
              fprintf(fidthirdlambda, stringa4, Lambdaout(2*(q*5) + 1: 3*(q*5), 1));
              fprintf(fidfourthlambda, stringa4, Lambdaout(3*(q*5) + 1: 4*(q*5), 1));
              fprintf(fidfifthlambda, stringa4, Lambdaout(4*(q*5) + 1: 5*(q*5), 1));
              fprintf(fidsixthlambda, stringa4, Lambdaout(5*(q*5) + 1: 6*(q*5), 1));
              fprintf(fidseventhlambda, stringa4, Lambdaout(6*(q*5) + 1: 7*(q*5), 1));
              fprintf(fideightlambda, stringa4, Lambdaout(7*(q*5) + 1: 8*(q*5), 1));
              fprintf(fidninthlambda, stringa4, Lambdaout(8*(q*5) + 1: 9*(q*5), 1));
              fprintf(fidtenthlambda, stringa4, Lambdaout(9*(q*5) + 1: 10*(q*5), 1));
              clear Lambdaout;
              
              Tout = zeros(q*50, 1);
              Tout(1:q*k, 1) = reshape(t, q*k,1); 
              fprintf(fidfirstt, stringa4, Tout(1:q*5, 1));
              fprintf(fidsecondt, stringa4, Tout((q*5) + 1: 2*(q*5), 1));
              fprintf(fidthirdt, stringa4, Tout(2*(q*5) + 1: 3*(q*5), 1));
              fprintf(fidfourtht, stringa4, Tout(3*(q*5) + 1: 4*(q*5), 1));
              fprintf(fidfiftht, stringa4, Tout(4*(q*5) + 1: 5*(q*5), 1));
              fprintf(fidsixtht, stringa4, Tout(5*(q*5) + 1: 6*(q*5), 1));
              fprintf(fidseventht, stringa4, Tout(6*(q*5) + 1: 7*(q*5), 1));
              fprintf(fideightt, stringa4, Tout(7*(q*5) + 1: 8*(q*5), 1));
              fprintf(fidnintht, stringa4, Tout(8*(q*5) + 1: 9*(q*5), 1));
              fprintf(fidtentht, stringa4, Tout(9*(q*5) + 1: 10*(q*5), 1));
              clear Tout;
              
              Deltaout = zeros(50, 1);
              Deltaout(1:k, 1) = reshape(delta, k,1); 
              fprintf(fidfirstdelta, stringa8, Deltaout(1:5, 1));
              fprintf(fidseconddelta, stringa8, Deltaout((5) + 1: 2*(5), 1));
              fprintf(fidthirddelta, stringa8, Deltaout(2*(5) + 1: 3*(5), 1));
              fprintf(fidfourthdelta, stringa8, Deltaout(3*(5) + 1: 4*(5), 1));
              fprintf(fidfifthdelta, stringa8, Deltaout(4*(5) + 1: 5*(5), 1));
              fprintf(fidsixthdelta, stringa8, Deltaout(5*(5) + 1: 6*(5), 1));
              fprintf(fidseventhdelta, stringa8, Deltaout(6*(5) + 1: 7*(5), 1));
              fprintf(fideightdelta, stringa8, Deltaout(7*(5) + 1: 8*(5), 1));
              fprintf(fidninthdelta, stringa8, Deltaout(8*(5) + 1: 9*(5), 1));
              fprintf(fidtenthdelta, stringa8, Deltaout(9*(5) + 1: 10*(5), 1));
              clear Deltaout;
              
              Tauout = zeros(50, 1);
              Tauout(1:k, 1) = reshape(tau, k,1); 
              fprintf(fidfirsttau, stringa8, Tauout(1:5, 1));
              fprintf(fidsecondtau, stringa8, Tauout((5) + 1: 2*(5), 1));
              fprintf(fidthirdtau, stringa8, Tauout(2*(5) + 1: 3*(5), 1));
              fprintf(fidfourthtau, stringa8, Tauout(3*(5) + 1: 4*(5), 1));
              fprintf(fidfifthtau, stringa8, Tauout(4*(5) + 1: 5*(5), 1));
              fprintf(fidsixthtau, stringa8, Tauout(5*(5) + 1: 6*(5), 1));
              fprintf(fidseventhtau, stringa8, Tauout(6*(5) + 1: 7*(5), 1));
              fprintf(fideighttau, stringa8, Tauout(7*(5) + 1: 8*(5), 1));
              fprintf(fidninthtau, stringa8, Tauout(8*(5) + 1: 9*(5), 1));
              fprintf(fidtenthtau, stringa8, Tauout(9*(5) + 1: 10*(5), 1));
              clear Tauout;
              
              Pthtout = zeros(q*50, 1);
              Pthtout(1:q*k, 1) = reshape(Ptht, q*k,1); 
              fprintf(fidfirstPtht, stringa4, Pthtout(1:q*5, 1));
              fprintf(fidsecondPtht, stringa4, Pthtout((q*5) + 1: 2*(q*5), 1));
              fprintf(fidthirdPtht, stringa4, Pthtout(2*(q*5) + 1: 3*(q*5), 1));
              fprintf(fidfourthPtht, stringa4, Pthtout(3*(q*5) + 1: 4*(q*5), 1));
              fprintf(fidfifthPtht, stringa4, Pthtout(4*(q*5) + 1: 5*(q*5), 1));
              fprintf(fidsixthPtht, stringa4, Pthtout(5*(q*5) + 1: 6*(q*5), 1));
              fprintf(fidseventhPtht, stringa4, Pthtout(6*(q*5) + 1: 7*(q*5), 1));
              fprintf(fideightPtht, stringa4, Pthtout(7*(q*5) + 1: 8*(q*5), 1));
              fprintf(fidninthPtht, stringa4, Pthtout(8*(q*5) + 1: 9*(q*5), 1));
              fprintf(fidtenthPtht, stringa4, Pthtout(9*(q*5) + 1: 10*(q*5), 1));
              clear Pthtout;
             
              fprintf(fidpredout, stringa7, Ytpredout);
              fprintf(fidfactor, '%3.3f\t', k);
              fprintf(fidpsi, '%3.3f\t', psiout);
              thtout = THETA(:); fprintf(fid, stringa, thtout);
              sigmaout = sig;
              fprintf(fidsigma, stringa3, sigmaout);

              clear thtout;
              clear sigmaout;
              clear psiout;
              clear Ytpredout;
             
         end
          
      [ i cputime]
         
    end

    nofrep(g,:) = nofout';
    adrep(g,:) = sum(adaptout);
    
    disp('-------------------'); 
    disp(['replicate',num2str(g),'complete']);

end

fclose(fid); 
fclose(fidsigma); fclose(fidpsi); 
fclose(fidpredout); fclose(fidfactor);
fclose(fidalphaone); fclose(fidalphatwo); fclose(fidpreecl);
fclose(fidlbw); fclose(fidgammaone); fclose(fidgammatwo);
fclose(fidrho); fclose(fidacc);
fclose(fidfirsteta); fclose(fidsecondeta); fclose(fidthirdeta); fclose(fidfourtheta); fclose(fidfiftheta);
fclose(fidsixtheta); fclose(fidseventheta); fclose(fideighteta); fclose(fidnintheta); fclose(fidtentheta);
fclose(fidfirstbeta); fclose(fidsecondbeta); fclose(fidthirdbeta); fclose(fidfourthbeta); fclose(fidfifthbeta);
fclose(fidsixthbeta); fclose(fidseventhbeta); fclose(fideightbeta); fclose(fidninthbeta); fclose(fidtenthbeta);
fclose(fidfirstlambda); fclose(fidsecondlambda); fclose(fidthirdlambda); fclose(fidfourthlambda); fclose(fidfifthlambda);
fclose(fidsixthlambda); fclose(fidseventhlambda); fclose(fideightlambda); fclose(fidninthlambda); fclose(fidtenthlambda);
fclose(fidfirstomega); fclose(fidsecondomega); fclose(fidthirdomega); fclose(fidfourthomega); fclose(fidfifthomega);
fclose(fidsixthomega); fclose(fidseventhomega); fclose(fideightomega); fclose(fidninthomega); fclose(fidtenthomega);
fclose(fidfirstt); fclose(fidsecondt); fclose(fidthirdt); fclose(fidfourtht); fclose(fidfiftht);
fclose(fidsixtht); fclose(fidseventht); fclose(fideightt); fclose(fidnintht); fclose(fidtentht);
fclose(fidfirstdelta); fclose(fidseconddelta); fclose(fidthirddelta); fclose(fidfourthdelta); fclose(fidfifthdelta);
fclose(fidsixthdelta); fclose(fidseventhdelta); fclose(fideightdelta); fclose(fidninthdelta); fclose(fidtenthdelta);
fclose(fidfirsttau); fclose(fidsecondtau); fclose(fidthirdtau); fclose(fidfourthtau); fclose(fidfifthtau);
fclose(fidsixthtau); fclose(fidseventhtau); fclose(fideighttau); fclose(fidninthtau); fclose(fidtenthtau);
fclose(fidfirstPtht); fclose(fidsecondPtht); fclose(fidthirdPtht); fclose(fidfourthPtht); fclose(fidfifthPtht);
fclose(fidsixthPtht); fclose(fidseventhPtht); fclose(fideightPtht); fclose(fidninthPtht); fclose(fidtenthPtht);

save(strcat('predoutp_',num2str(n),'ktr_',num2str(ktr),'rep_',num2str(rep)),'adrep');

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Computing marginal probability of preeclampsia and low birth weight -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load firsteta.txt; load secondeta.txt; load thirdeta.txt; load fourtheta.txt;
load fiftheta.txt; load sixtheta.txt; load seventheta.txt; load eighteta.txt;
load nintheta.txt; load tentheta.txt;
load gammaone.txt; load gammatwo.txt; load alphaoutone.txt; load alphaouttwo.txt;
load rho.txt; 

% -- probability of preeclampsia and low birth weight -- %
 alphaone = alphaoutone; alphatwo = alphaouttwo; 
 probpreecl = zeros(4000, 200);
 problbw = zeros(4000, 200);
 for j = 1001:5000
     Sigmatilde = [1 rho(j); rho(j) 1];
     for h = 1:n
         set = [(h-1)+1, n + (h - 1 + 1), 2*n + (h-1)+1, 3*n + (h-1) + 1, 4*n + (h-1) + 1];
         eta = [firsteta(j, set) secondeta(j, set) thirdeta(j, set) fourtheta(j, set) fiftheta(j, set) ...
             sixtheta(j, set) seventheta(j, set) eighteta(j, set) nintheta(j, set) tentheta(j, set)];
         gnu = mvncdf([0, 0], [Inf, Inf], [alphaone(j) + gammaone(j,:)*eta', alphatwo(j) + gammatwo(j,:)*eta'], Sigmatilde);
         probpreecl(j,h) =  gnu + ...
             mvncdf([0, -Inf], [Inf, 0], [alphaone(j) + gammaone(j,:)*eta', alphatwo(j) + gammatwo(j,:)*eta'], Sigmatilde);
         problbw(j,h) = gnu + ...
             mvncdf([-Inf, 0], [0, Inf], [alphaone(j) + gammaone(j,:)*eta', alphatwo(j) + gammatwo(j,:)*eta'], Sigmatilde);
     end
     j
 end
 
dlmwrite('probpreeclampsia.txt', probpreecl(1001:5000,:), 'delimiter', '\t', 'precision', 6) 
dlmwrite('problowbweight.txt', problbw(1001:5000, :), 'delimiter', '\t', 'precision', 6)
 
probpreeclampsia = dlmread('probpreeclampsia.txt');
problbw = dlmread('problowbweight.txt');
mean(probpreeclampsia(1:4000, test))
std(probpreeclampsia(:,test))./sqrt(4000)
mean(problbw(1:4000, test))
std(problbw(:,test))./sqrt(4000)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Plotting estimated trajectories -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load thetaout.txt;
load Design;
thtout = thetaout;
its = (burn / thin + 1): round(i / thin);                                          
cnt = 0; 
m = 290;
ib = unidrnd(n, 6 , 1);
for h = 1:6
  i = ib(h);
  cnt = cnt + 1;
  subplot(3,2,cnt)
  thti = thtout(its,((i-1)*q+1):(i*q));
  Eyi = (Xb*thti')';
  est = zeros(290,3); 
  est(:,1) = mean(Eyi)'; 
  est(:,2:3) = prctile(Eyi,[2.5 97.5])';   % 95% pointwise credible interval
  plot((tij(id == i)*m)/7,y(id == i),'o','MarkerSize',5, 'Color', 'k')
  title(['Subject ' num2str(i)])
  xlabel(['Gestational age (Weeks)'])
  ylabel(['MAP'])
  xlim([0 42])
  %ylim([65 110])
  line((tg*m)/7, est(:,1),'LineStyle', '-','Color', 'b','LineWidth', 2)        % posterior mean estimate
  line((tg*m)/7, est(:,2),'LineStyle','--','LineWidth',1.5)                    % 97% pointwise intervals
  line((tg*m)/7, est(:,3),'LineStyle','--','LineWidth',1.5)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Computing predictive errors -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Ytpredout.txt;
load Design;
prova = [nitestobs tst];
MSPE = zeros(4000, sum(nitestobs));
MAAB = zeros(4000, sum(nitestobs));
MMAB= zeros(4000, length(tst));
for j = 1:length(tst);
    h = tst(j);
    obs = tij(id == h).*290./7;
    toplot = y(id == h);
    X_new = prova(:,2) - h;
    X_new_new = abs(X_new);
    [dummy, index] = min(X_new_new);
    MSPE(:, sum(nitestobs(1: (index - 1))) + 1 : sum(nitestobs(1:index))) =  bsxfun(@minus,Ytpredout(1001:5000, sum(nitestobs(1: (index - 1))) + 1 : sum(nitestobs(1:index))),toplot(1: length(obs((obs > 30))))').^2;
    MAAB(:, sum(nitestobs(1: (index - 1))) + 1 : sum(nitestobs(1:index))) =  abs(bsxfun(@minus,Ytpredout(1001:5000, sum(nitestobs(1: (index - 1))) + 1 : sum(nitestobs(1:index))),toplot(1: length(obs((obs > 30))))'));
    MMAB(:, j) =  max(abs(bsxfun(@minus,Ytpredout(1001:5000, sum(nitestobs(1: (index - 1))) + 1 : sum(nitestobs(1:index))),toplot(1: length(obs((obs > 30))))')), [], 2);
end
MSPE = mean(mean(MSPE))
MAAB = mean(mean(MAAB))
MMAB = mean(max(MMAB, [], 2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Side-by-side boxplots to study impact of covariates -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load firstbeta.txt; load secondbeta.txt; load thirdbeta.txt; load fourthbeta.txt;
load fifthbeta.txt; load sixthbeta.txt; load seventhbeta.txt; load eightbeta.txt;
load ninthbeta.txt; load tenthbeta.txt;
beta = [firstbeta secondbeta thirdbeta fourthbeta fifthbeta sixthbeta seventhbeta eightbeta ninthbeta tenthbeta];
normmatrix = zeros(4000, 10);
for j = 1001:5000
    Beta = reshape(beta(j,:), r, 50);
    for i = 1:10
     normmatrix(j,i) = norm(Beta(i,1:10));
    end
end

h = boxplot(normmatrix(1001:5000, : ))
set(h(7,:),'Visible','off') % remove outliers

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of latent factors (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firsteta.txt; load secondeta.txt; load thirdeta.txt; load fourtheta.txt;
% load fiftheta.txt; load sixtheta.txt; load seventheta.txt; load eighteta.txt;
% load nintheta.txt; load tentheta.txt;
%  
% Firsteta = reshape(mean(firsteta(1001:5000,:)), n, 5); Secondeta = reshape(mean(secondeta(1001:5000,:)), n, 5);
% Thirdeta = reshape(mean(thirdeta(1001:5000,:)), n, 5); Fourtheta = reshape(mean(fourtheta(1001:5000,:)), n, 5);
% Sixtheta = reshape(mean(sixtheta(1001:5000,:)), n, 5); Seventheta = reshape(mean(seventheta(1001:5000,:)), n, 5);
% Eighteta = reshape(mean(eighteta(1001:5000,:)), n, 5); Nintheta = reshape(mean(nintheta(1001:5000,:)), n, 5);
% Tentheta = reshape(mean(tentheta(1001:5000,:)), n, 5); Fiftheta = reshape(mean(fiftheta(1001:5000,:)), n, 5);
%  
% eta = [Firsteta Secondeta Thirdeta Fourtheta Fiftheta Sixtheta Seventheta Eighteta Nintheta Tentheta];
% dlmwrite('PostEta.txt', eta, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of covariate's coefficients (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firstbeta.txt; load secondbeta.txt; load thirdbeta.txt; load fourthbeta.txt;
% load fifthbeta.txt; load sixthbeta.txt; load seventhbeta.txt; load eightbeta.txt;
% load ninthbeta.txt; load tenthbeta.txt;
%  
% Firstbeta = reshape(mean(firstbeta(1001:5000,:)), r, 5); Secondbeta = reshape(mean(secondbeta(1001:5000,:)), r, 5);
% Thirdbeta = reshape(mean(thirdbeta(1001:5000,:)), r, 5); Fourthbeta = reshape(mean(fourthbeta(1001:5000,:)), r, 5);
% Sixthbeta = reshape(mean(sixthbeta(1001:5000,:)), r, 5); Seventhbeta = reshape(mean(seventhbeta(1001:5000,:)), r, 5);
% Eightbeta = reshape(mean(eightbeta(1001:5000,:)), r, 5); Ninthbeta = reshape(mean(ninthbeta(1001:5000,:)), r, 5);
% Tenthbeta = reshape(mean(tenthbeta(1001:5000,:)), r, 5); Fifthbeta = reshape(mean(fifthbeta(1001:5000,:)), r, 5);
%  
% beta = [Firstbeta Secondbeta Thirdbeta Fourthbeta Fifthbeta Sixthbeta Seventhbeta Eightbeta Ninthbeta Tenthbeta];
% dlmwrite('PostBeta.txt', beta, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of covariance of Beta (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firstomega.txt; load secondomega.txt; load thirdomega.txt; load fourthomega.txt;
% load fifthomega.txt; load sixthomega.txt; load seventhomega.txt; load eightomega.txt;
% load ninthomega.txt; load tenthomega.txt;
 
% Firstomega = reshape(mean(firstomega(1001:5000,:)), r, 5); Secondomega = reshape(mean(secondomega(1001:5000,:)), r, 5);
% Thirdomega = reshape(mean(thirdomega(1001:5000,:)), r, 5); Fourthomega = reshape(mean(fourthomega(1001:5000,:)), r, 5);
% Sixthomega = reshape(mean(sixthomega(1001:5000,:)), r, 5); Seventhomega = reshape(mean(seventhomega(1001:5000,:)), r, 5);
% Eightomega = reshape(mean(eightomega(1001:5000,:)), r, 5); Ninthomega = reshape(mean(ninthomega(1001:5000,:)), r, 5);
% Tenthomega = reshape(mean(tenthomega(1001:5000,:)), r, 5); Fifthomega = reshape(mean(fifthomega(1001:5000,:)), r, 5);
 
% omega = [Firstomega Secondomega Thirdomega Fourthomega Fifthomega Sixthomega Seventhomega Eightomega Ninthomega Tenthomega];
% dlmwrite('PostOmega.txt', omega, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of loading matrix (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firstlambda.txt; load secondlambda.txt; load thirdlambda.txt; load fourthlambda.txt;
% load fifthlambda.txt; load sixthlambda.txt; load seventhlambda.txt; load eightlambda.txt;
% load ninthlambda.txt; load tenthlambda.txt;
%  
% Firstlambda = reshape(mean(firstlambda(1001:5000,:)), q, 5); Secondlambda = reshape(mean(secondlambda(1001:5000,:)), q, 5);
% Thirdlambda = reshape(mean(thirdlambda(1001:5000,:)), q, 5); Fourthlambda = reshape(mean(fourthlambda(1001:5000,:)), q, 5);
% Sixthlambda = reshape(mean(sixthlambda(1001:5000,:)), q, 5); Seventhlambda = reshape(mean(seventhlambda(1001:5000,:)), q, 5);
% Eightlambda = reshape(mean(eightlambda(1001:5000,:)), q, 5); Ninthlambda = reshape(mean(ninthlambda(1001:5000,:)), q, 5);
% Tenthlambda = reshape(mean(tenthlambda(1001:5000,:)), q, 5); Fifthlambda = reshape(mean(fifthlambda(1001:5000,:)), q, 5);
%  
% lambda = [Firstlambda Secondlambda Thirdlambda Fourthlambda Fifthlambda Sixthlambda Seventhlambda Eightlambda Ninthlambda Tenthlambda];
% dlmwrite('PostLambda.txt', lambda, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of local shrinkage coeff. (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firstt.txt; load secondt.txt; load thirdt.txt; load fourtht.txt;
% load fiftht.txt; load sixtht.txt; load seventht.txt; load eightt.txt;
% load nintht.txt; load tentht.txt;
 
% Firstt = reshape(mean(firstt(1001:5000,:)), q, 5); Secondt = reshape(mean(secondt(1001:5000,:)), q, 5);
% Thirdt = reshape(mean(thirdt(1001:5000,:)), q, 5); Fourtht = reshape(mean(fourtht(1001:5000,:)), q, 5);
% Sixtht = reshape(mean(sixtht(1001:5000,:)), q, 5); Seventht = reshape(mean(seventht(1001:5000,:)), q, 5);
% Eightt = reshape(mean(eightt(1001:5000,:)), q, 5); Nintht = reshape(mean(nintht(1001:5000,:)), q, 5);
% Tentht = reshape(mean(tentht(1001:5000,:)), q, 5); Fiftht = reshape(mean(fiftht(1001:5000,:)), q, 5);
 
% t = [Firstt Secondt Thirdt Fourtht Fiftht Sixtht Seventht Eightt Nintht Tentht];
% dlmwrite('PostT.txt', t, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of global shrinkage coeff. multipl. (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firstdelta.txt; load seconddelta.txt; load thirddelta.txt; load fourthdelta.txt;
% load fifthdelta.txt; load sixthdelta.txt; load seventhdelta.txt; load eightdelta.txt;
% load ninthdelta.txt; load tenthdelta.txt;
 
% Firstdelta = reshape(mean(firstdelta(1001:5000,:)), 1, 5); Seconddelta = reshape(mean(seconddelta(1001:5000,:)), 1, 5);
% Thirddelta = reshape(mean(thirddelta(1001:5000,:)), 1, 5); Fourthdelta = reshape(mean(fourthdelta(1001:5000,:)), 1, 5);
% Sixthdelta = reshape(mean(sixthdelta(1001:5000,:)), 1, 5); Seventhdelta = reshape(mean(seventhdelta(1001:5000,:)), 1, 5);
% Eightdelta = reshape(mean(eightdelta(1001:5000,:)), 1, 5); Ninthdelta = reshape(mean(ninthdelta(1001:5000,:)), 1, 5);
% Tenthdelta= reshape(mean(tenthdelta(1001:5000,:)), 1, 5); Fifthdelta = reshape(mean(fifthdelta(1001:5000,:)), 1, 5);
 
% delta = [Firstdelta Seconddelta Thirddelta Fourthdelta Fifthdelta Sixthdelta Seventhdelta Eightdelta Ninthdelta Tenthdelta];
% dlmwrite('PostDelta.txt', delta, 'delimiter', '\t', 'precision', 6);
% clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Saving posterior means of global shrinkage coeff. (after burn-in) -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load firsttau.txt; load secondtau.txt; load thirdtau.txt; load fourthtau.txt;
% load fifthtau.txt; load sixthtau.txt; load seventhtau.txt; load eighttau.txt;
% load ninthtau.txt; load tenthtau.txt;
 
% Firsttau = reshape(mean(firsttau(1001:5000,:)), 1, 5); Secondtau = reshape(mean(secondtau(1001:5000,:)), 1, 5);
% Thirdtau = reshape(mean(thirdtau(1001:5000,:)), 1, 5); Fourthtau = reshape(mean(fourthtau(1001:5000,:)), 1, 5);
% Sixthtau = reshape(mean(sixthtau(1001:5000,:)), 1, 5); Seventhtau = reshape(mean(seventhtau(1001:5000,:)), 1, 5);
% Eighttau = reshape(mean(eighttau(1001:5000,:)), 1, 5); Ninthtau = reshape(mean(ninthtau(1001:5000,:)), 1, 5);
% Tenthtau = reshape(mean(tenthtau(1001:5000,:)), 1, 5); Fifthtau = reshape(mean(fifthtau(1001:5000,:)), 1, 5);
 
% tau = [Firsttau Secondtau Thirdtau Fourthtau Fifthtau Sixthtau Seventhtau Eighttau Ninthtau Tenthtau];
% dlmwrite('PostTau.txt', tau, 'delimiter', '\t', 'precision', 6);
% clear all;

% load firstPtht.txt; load secondPtht.txt; load thirdPtht.txt; load fourthPtht.txt;
% load fifthPtht.txt; load sixthPtht.txt; load seventhPtht.txt; load eightPtht.txt;
% load ninthPtht.txt; load tenthPtht.txt;
 
% FirstPtht = reshape(mean(firstPtht(1001:5000,:)), q, 5); SecondPtht = reshape(mean(secondPtht(1001:5000,:)), q, 5);
% ThirdPtht = reshape(mean(thirdPtht(1001:5000,:)), q, 5); FourthPtht = reshape(mean(fourthPtht(1001:5000,:)), q, 5);
% SixthPtht = reshape(mean(sixthPtht(1001:5000,:)), q, 5); SeventhPtht = reshape(mean(seventhPtht(1001:5000,:)), q, 5);
% EightPtht = reshape(mean(eightPtht(1001:5000,:)), q, 5); NinthPtht = reshape(mean(ninthPtht(1001:5000,:)), q, 5);
% TenthPtht = reshape(mean(tenthPtht(1001:5000,:)), q, 5); FifthPtht = reshape(mean(fifthPtht(1001:5000,:)), q, 5);
 
% Ptht = [FirstPtht SecondPtht ThirdPtht FourthPtht FifthPtht SixthPtht SeventhPtht EightPtht NinthPtht TenthPtht];
% dlmwrite('PostPtht.txt', Ptht, 'delimiter', '\t', 'precision', 6);
% clear all;

