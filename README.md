an implementation and evaluation of two CATE estimators — dragonnet (a multi-task neural network, PyTorch) and an x-learner with cross-fitting (scikit-learn) — benchmarked on the semi-synthetic ihdp dataset. targeted regularization reduced dragonnet's PEHE by 10.5% over the base model, and both models beat a naive baseline. treatment effect heterogeneity was analyzed using the BLP / GATES / CLAN framework from Chernozhukov et al., pooled across 100 replications using monte carlo approximation and wilcoxon signed-rank test.

## below is my intuitive understanding and derivation of the techniques i used in this project (still a work in progress)


in causal inference, treatment effect is defined as an individual's potential outcome when treated ($Y_1$) minus the potential outcome when untreated ($Y_0$). ATE (average treatment effect) is $Y_1 - Y_0$ averaged across everyone. and CATE (conditional average treatment effect) is averaged across people who non-negotiably share a certain trait or exact set of traits $z$: "for people exactly like this, what is the average $Y_1 - Y_0$?" the proper cate procedure really is about averaging within the defined group, but models don't do this; they estimate it by fitting a function of $z$.

the ihdp dataset came from an randomized controled experiment in the 1980s. jennifer hill later turned this dataset into synthetic observational data to help study causal machine learning. she did this by artificially introducing confounders: removing non-white mothers in the treated group. when confounding exists, a naive difference in average treatment effect across groups will produce a biased result. here, the treated group being white is associated with socioeconomic advantages that affect the outcome variable in the study. thus the outcome variable is artificially inflated. the goal of CATE estimators in causal machine learning is to say "instead of a biased naive difference in average outcome, let's study the outcome for individuals that share the same confounders."

## blp (best linear predictor)

in chernozhukov's blp, the basic regressed equation is

$$Y = b_1(T-p) + b_2(T-p)(s-s_{mean}) + \epsilon$$

the goal of blp is to figure out the regression coefficients $b_1$ and $b_2$, because they tell us something useful. i will eventually explain why.

rewriting this equation helped me understand. let's factor out the $(T-p)$.

```math
Y = \left[b_1 + b_2(s-s_{mean})\right](T-p) + \epsilon
```

or

$$Y = c(T-p) + \epsilon$$

where $c$ is the coefficient of $(T-p)$. $Y$ is the outcome and $T$ is treatment status (0 or 1). each individual has their own $s$ and $p$ values, which are plugged into this equation. $\epsilon$ represents error, the difference between the regression's prediction of $Y$ and the real outcome $Y$.

### showing that coefficient $c$ represents the treatment effect

the best this equation can do to define an individual is use their $T$, $s$, and $p$ values. so treat the individual here as a configuration of these variables' values. if $T$ is hypothetically set to 1, then output $Y$ answers "what does this individual look like when treated?" this is nothing but $Y_1$ for that individual. if $T$ is set to 0, output $Y$ answers "what does this individual look like at baseline?", which is $Y_0$.

if $T=1$, $Y_1 = c(1-p)$ <br>
if $T=0$, $Y_0 = c(0-p)$ <br>
$Y_1 - Y_0 = c(1-p) - c(0-p)$ <br>
$Y_1 - Y_0 = c - p \cdot c + p \cdot c = c$ <br>
$c = Y_1 - Y_0$

so for that individual, the coefficient $c$ is $Y_1 - Y_0$. this is definitionally the treatment effect. but here $Y_1$ and $Y_0$ are blp's estimations through regression, and so $c$ represents the linear approximation of treatment effect as well. this is why i omitted the error term in the calculation. error is just the gap between approximation and real. in summary, the linear approximation of treatment effect is the slope of the regression line when regressing $Y$ on $T-p$, the factor by which $Y$ changes as $T-p$ changes. this is true for $c$ regardless of what is inside $c$.

### understanding $b_1$

let's go back to $`Y = \left[b_1 + b_2(s-s_{mean})\right](T-p) + \epsilon`$. look at the structure of the coefficient. it's saying $b_1$ is the base coefficient. and to it is added some deviation on $b_1$. this deviation depends on variable $s$, which varies from individual to individual. regardless of what $s$ represents as a variable, if $s$ is higher than average, $s-s_{mean}$ will be positive, and $b_1$ will be boosted. if $s$ is lower than average, $s-s_{mean}$ will be negative, and $b_1$ will be reduced. notice that $s$ is the only thing that changes across individuals in this coefficient term. the equation regresses across multiple participants, and a person's linear approximation of treatment effect is $c = b_1 + b_2(s-s_{mean})$. if we add up all the individual $c$'s and take the average, that average will become

$$b_1 + b_2(s_{mean}-s_{mean}) = b_1 + b_2 \cdot 0 = b_1$$

across all participants, $b_1$ is the average coefficient. therefore, $b_1$ is the linear approximation of average treatment effect.

### understanding $b_2$

for a non-average person, their treatment effect will be $b_1$ plus some deviation in $s$. you are simply taking $b_1$ and applying an addition or subtraction to it. how much does $s-s_{mean}$ change $b_1$? that depends on your value of $b_2$. if $b_2$ is high, it is going to amplify the change for all individuals. an above-average $s$ will lead to a higher addition to $b_1$, and a below-average $s$ will lead to a higher subtraction from $b_1$. we can see that $b_2$ is essentially modulating the spread of $s$ before it gets passed on to $b_1$. a high $b_2$ increases the spread and a low $b_2$ squishes the spread. now, $b_2$ is a regression coefficient. think of the role of regression. it is to choose a value of $b_2$ such that the prediction on outcome $Y$ has low residual error. if it chooses a $b_2$ of 1, it is saying that the spread of $s$ does not need to change in order to best predict $Y$. if $b_2$ is between 0 and 1, it is saying the original spread is too high and nerfing it will make it a better predictor of $Y$. if $b_2 > 1$, it is saying the original spread is too little and needs to be buffed. in summary, $b_2$ is determining whether the spread of $s$ is on the right scale to predict $Y$.

now look at this cool trick. we know that the coefficient is the linear approximation of treatment effect. we know that $b_1$ is the linear approximation of average treatment effect.

$$c = b_1 + b_2(s-s_{mean})$$

therefore,

$$\text{blp treatment effect} = \text{blp average treatment effect} + b_2(s-s_{mean})$$

$$\text{blp treatment effect} - \text{blp average treatment effect} = b_2(s-s_{mean})$$

looking at this form makes it more intuitive what the regression is evaluating. imagine we have our own outside estimated treatment effects for each individual and plug them into $s$. the left side represents what the spread should look like under linear regression. the right side is our own spread. if $b_2 = 1$, it tells us our spread is on the same scale as what blp wants in order to reduce residual error from real $Y$. in conclusion, blp is simply one way to validate the scale of spread of some external individual treatment effect scores, usually presented as estimated CATEs. it is validating their claimed level of heterogeneity.

### why $T-p$ and not merely $T$?

let's start from the ground up. there are two groups, control and treatment. let's say there is a variable $x$ that is overrepresented in the treatment group and that also happens to provide an advantage to the treatment group in producing the outcome. a basic way to compare the effects of the treatment in an rct is to measure the difference between the control group's outcome and the treated group's outcome. but here, that difference includes both the effect of the treatment and the advantage that the treatment group's characteristic provides.

a natural solution is not to remove the advantage, but to have it equally represented in both groups. if both groups get the same advantage, a difference in the groups' outcomes is not skewed by the advantage... unless one group systematically harnesses the advantage differently due to some other variable, but let's assume for simplicity that that is not an issue. scan both control and treatment group for the presence of the advantage, and observe the distribution. if the treatment side has nine participants with the advantage and the control side has only one with the advantage, their voices must be in a ratio of 1:9 for equal representation.

if you remember, treatment effect can be represented as the slope of the regression line when regressing $Y$ on $T-p$ for a group of people. the math works even without a $-p$.

$Y = cT$ <br>
$Y_1 = c \cdot 1 = c$ <br>
$Y_0 = c \cdot 0 = 0$ <br>
$Y_1 - Y_0 = c - 0 = c$

so, to be more clear, treatment effect is quite literally the slope of the line when regressing outcomes $Y$ on treatment status $T$. this is what we want to do to find out a linear approximation of treatment effect. but don't do a normal regression skewed by imbalance within a trait. do weighted regression. look at only those people who have the trait. say "percentage" here refers to the percentage of those people who are on the treated side. let's say nine people on treated and one person in control. if the person is in the control group, with a low percent of people on his side, he should get more weight. a weight of 90%. so make the weight = percentage. if someone is in the treated group, a part of the 90%, they should be nerfed to 10% weight. so, 100-percentage.

in a real experiment, there may be many different traits assessed. we can use machine learning to put those together and automatically assign each individual a percentage. "given the group of people of your type, what does that group's distribution of control-treated look like?" distribution also tells us about probability: "among people of this type, how likely are people to fall into the treated group?" this is known as propensity score ($p$), and it is between 0 and 1. it is commonly thought of as probability, but it helped me here to think in terms of distribution. someone whose traits map to $p=.9$ tells us there are many more people of that sort in the treated group. if that person specifically is in the control group, they need to be given high weight. $T=0$, $w=p$. $T=1$, $w=1-p$. something key to note: among all people of a certain type, their propensities are the same. but among all people who share a propensity, they may belong to different types.

continuing with the regression, there's a problem. say group A is of a type and has $p=.9$. one control, nine treated. 10 people total. the total weight of this group is $1(.9) + 9(1-.9) = 1.8$. now say group B also has 10 people. five control and five treated. the total weight of this group is $5 \cdot .5 + 5(1-.5) = 5$. both have 10 people, yet one group is getting more weight than the other due to our transformation. yes, initially everyone had a weight of 1, so each group had the same weight of 10. but now, a group of people with 50-50 distribution has the most total weight, while groups with an extreme imbalance have the least weight. should different groups end up with different weights? i don't think that's fair. let's make another correction to the weights. multiply the weights by $\frac{1}{p(1-p)}$. this specific formula nerfs the weight of the group that has $p=.5$ the most and nerfs the imbalanced groups the least, to counteract the side effect. so in total, the proper weights are

$$T=0: \quad \frac{p}{p(1-p)} = \frac{1}{1-p}$$

$$T=1: \quad \frac{1-p}{p(1-p)} = \frac{1}{p}$$

this technique is called inverse propensity weighting.

a second method of forcing equal representation within a type group, instead of giving weights to points, is to shift each group on the x-axis so that visually, the high-representation points are near the mean $x$ of the dataset and they get less statistical leverage, while the low points are far from the center so they get higher leverage. leverage as in power to influence the slope. so the minority people, or high-leverage points, have a greater ability to influence the predicted effect that changing $T$ has on changing $Y$. the precise shift for this is $T-p$. a $p$ of $.5$ will shift those points from 0 and 1 to $-.5$ and $.5$ on the x-axis. a $p$ of $.9$ will shift the one datapoint to $x=-.9$ and the nine points to $x=.1$. this corresponds to a 9:1 leverage. but we still want to include the $\frac{1}{p(1-p)}$ part. it's funny how it plays out: even with the leverage method, we can see that the group on $-.5$ to $.5$ has greater leverage overall than the $-.9$ to $.1$ group. it's because in the second group, a vast majority of the points are clustered near 0, the mean $x$, which basically has no leverage on the slope while regressing the dataset. so we still want to nerf the $p=.5$ group the most. this is the technique used in blp.

the other interpretation is that you're doing $T$ minus predicted $T$ using $x$, which strips the part of $T$ that is explained by $x$. this isolates the change in $T$ that does not vary with $x$, in order to study the effect of that change in $T$ on $Y$. this is the same idea behind DML. both $Y$ and $T$ are regressed on $x$ and stripped of the part that depends on $x$, and then clean $Y$ is regressed on clean $T$.

### about extra terms you may see in blp

in the base equation, $\epsilon$ is whatever error between prediction and outcome that is not explained by the two treatment terms. let's take, for example, wealth as a variable. wealth is something that can predict $Y$ independent of treatment, independent of the experiment. if we throw that in there, we have a third thing alongside treatment terms that helps predict $Y$, so the "unexplained" value $\epsilon$ decreases. oh, there's another variable: weight. throw that one in there too. oh wait, we have 25 covariates we can throw in. 25 opportunities to help reduce $\epsilon$. why does this matter? well, if we reduce $\epsilon$, we can reduce the variance of the $b$ coefficients. look at the variance formula:

$$\text{var}(b) \approx \frac{\epsilon/(n-k)}{n \cdot \text{variance of whatever } b \text{ is describing}}$$

reducing the numerator's $\epsilon$ will reduce the overall variance of coefficient $b$. however, there is a catch with adding too many variables. it can increase $k$, which is the number of covariates being estimated. that can increase variance of $b$. so reducing the variance of $b$ is about finding the right balance between predictive power and too many degrees of freedom $k$. a covariate that does not explain $Y$ well will not reduce $\epsilon$ by much, but it will cost you by increasing $k$.

adding terms to the regression to reduce variance or decrease confounding is called adding a "nuisance parameter". in blp, a very useful one to add is a prediction of $Y_0$. this encapsulates the predictive power of all covariates into one term. many rcts may also use ANCOVA to add in covariates as "precision variables" to improve variance. while the experiments are randomized and the covariates don't affect the treatment, they may help predict the outcome.

## x-learner

x-learner comes in multiple steps. first, participants are split into control and treated. each group's outcome is fitted using covariates. now that we have the ability to get $Y_0$ and $Y_1$, we can predict them for a new dataset and subtract them to get $d$, the treatment effects, and stop here. this approach is called a t-learner.

however, x-learner goes one step further. using your training data, the model for treated is used to predict the hypothetical treated outcomes for the control group's participants, and vice versa. this produces counterfactuals for each group. you then find the difference between a person's real outcome and their estimated counterfactual ($Y_1 - Y_0$) to get $d$. after that, you fit $d$ onto that group's covariates. so now your predicted $d$ is built not only on the other group's covariates but also on its own group's covariates. using more datapoints provides more information for the model and thus allows for a better prediction and lower variance.

finally, the two models are used to predict the $d$ values for a new set of data. because there are two models, it takes a weighted average of the two based on propensity score. given a low propensity, that individual's neighborhood's treated population is thin. therefore the control group's counterfactuals are based on thin data. and this model with high variance gets spread to a high number of datapoints in control. so $d_0$ has a high number of datapoints to fit on, but each one is high variance. in the treated group in the same situation, there is good control data and therefore good counterfactuals. so there are very accurate $d_1$ datapoints to fit on, but the total number of datapoints is lower. which bottleneck is worse? a good number of high-variance datapoints, or a bad number of low-variance datapoints? i would prefer the low-datapoint group, because while it produces high variance, that calculation happens only once. in the other group, a high variance calculation happens for each datapoint. in summary, the x-learner for a low propensity wants to bias towards the $d_1$ model, and a high $p$ wants to bias towards $d_0$. the specific formula for estimated treatment effect is

$$p \cdot d_0 + (1-p)d_1$$

these are called estimated cates because, while there's no "conditional average" procedure, a person's estimated counterfactual roughly maps to "what does this type of person with these covariates produce as a counterfactual on average?"
