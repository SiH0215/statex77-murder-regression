# Sys.setlocale("LC_COLLATE", "ko_KR.UTF-8")

#-------------------------------------------------------------------------------
# Y = (a^)*X + (e)
# (1) 회귀 분석 : 
# (2) 회귀 진단 : 다중/다변량(Xs) 다중공선성 체크 + 오차항 4가지 체크 
#-------------------------------------------------------------------------------
#                            x1,x2,x3,...... xk
#           
# Y  = Y^   + (e) ~ 0 -> 설명 되지 않는부분. 최소화
#      ㄴ설명되는부분. 최대화   
# Y = (a^)*X + (e) , 여기서 회귀 진단 조건(다중공선성 , 오차항 4가지)에 부합하지 않는 경우 또는 
#              부합시키기 어려운 경우

# X  => 다중공선성 이슈가 있다면,(불체크 x, )
# <car::vif(Y^) 체크 후 문제발생시 3가지 대안>
# (대안 1) 변수 선택/제거기법 : step(Y^)  : forward/backward/stepwise selection/elimination.
# (대안 2) 능형 회귀 분석 = ridge regression *
# (대안 3) 주성분 회귀분석 
#    -> PCA(PC(Principal Component (주성분)끼리 독립시키기) 또는 FA(Factor Analysis, 인자분석) 방식을 통해서 
#              전체 데이터를 가지고 가는 방식을 택해야 함.

#주성분 분석: {  X{x1,x2,...x37} ---------------> pc1,pc2,..pc9  : (37개의 x -->9개의 pc 로 : 차원축소)  
#                ================                 =============
#               ㄴ변수간(Xs) 다중공선성  있음 => PC(주성분) 간독립 } 로 해결
#                                                 Y= β0 + β1*PCi1(공격능력) + β2*PCi2 + ...+ βp*PCip : 주성분(PC) 회귀분석 
# 주성분 회귀분석: PC간독립이용 Y 선형화 (한국프로야구 순위결정예측 논문참고)


# 회귀분석: 다중공선성 발생시 X변수제거 (=>정보손실 문제야기)
# (multi(Xs)colinearity: Xs = cor(Xi, Xj)!=0)
#                             === correlation  상관관계 

#-----------------------------------------------
# 다중/다변량(Xs)회귀 분석 및 회귀 진단 : 1/2   * 
#----------------------------------------------- 
#    (Y 목적 3가지 : "예측", 설명, 분류(의사결정 = DT))
#     Y =   Xs  +  e ↓: 
#   
#  (예측)회귀식 : Y^= (b1^)*x1+ b2^*x2  (추정치 e^(=잔차(residuals = 0 희망할 뿐 )
#                       ㄴ b: 회귀계수(coefficient), 

#  (예측)회귀식 : Y^= 0.2(a1^ =x1,y관측값에 의해 계산산출된 값) *x1+ 0.7*x2 (0.7도 (a2^ =x2,y관측값에 의해 계산산출된 값) )
#     Y =   Xs  : 상관관계 *있어야 선형식   : 상관관계는 회귀계수(Estimate, 추정치 ) 크기와 거의 유사함.  
#----------------------------------------------------------------------------------------------------------------------------------------------
# 
#     (1) Xs변수 : 다중공선성이 없어야 ★독립변수★로서 ==> 회귀분석이 가능함.
#     Y =   Xs  : 다중공선성 없어야 (있으면 없애야 -> 변수제거법)
#                      ㄴ[대안1 : 변수제거법{Backward/Forward/Stepwise} by AIC값이 낮은 변수들의 집합을 택함(낮을수록좋음)) 
#                          ㄴ (이슈 : 정보량 손실) : 그럼에도불구 변수제거 
#                            ㄴ (다중공선성문제, 예측회귀계수 신뢰성떨어드리므로)
# ----------------------------------------------------------------------------------------------------------------------------------------------
#     (3) 오차항에 대한 4-가정 만족할 때, 다중회귀분석 가능함. 

#      Y = Xs !<=  e ↓(=error) * e가 x에 영향을 끼치면 안됨.
#                   ㄴ [오차항 최소화 목표] 오(류:error↓)차항에 관한 4가지 가정
#                       ㄴe(오차항) ~ i,i,d,N(distributed, 분산)
#                           ㄴ(1) i: independently (독립성 가정) : 뭉치면 Xs 예측계수 흔들어 예측 신뢰성에 악영향 
#                            ㄴ(2) i: identically (동/등분산 가정 = sigma^2(일정한 분산)) : "분산이 일정"해야 함 
#                                   (먼지 차분 가라앉아 있어야 하듯, 변동가능성이 적어야 함 )
#                             ㄴ(3) e ~ N( , sigma^2) Normal distribution(정규성 가정)
#                              ㄴ(4) e 자기상관성 가지면 안된다. (뭉치지 않고, 어떤 패턴을 가지면 안된다. )
#---------------------------------------------------------------------------------------  



# 데이터 분석/모델링 5단계

# (Step1):(data engineer 주된 역할) 데이터 수집/추출/정제(이상여부 로직 체크)/저장
# (Step2):데이터 탐색 (psych::describe, psych::pairs.pannels)및 준비, 파생변수 생성(_gr), CrossTable,..
#            데이타셋 분리(모형개발용(train), 검증용(validation), 평가용(test))
# (Step3):데이터 모델링(모형화) *가장 중요
# (Step4):모형 성능(정확도) 평가 : Y-(Y^) = e ↓
# (Step5):모형 성능(정확도) 개선 : 평가결과 모형을 추가적 개선
#     (선택) (Step6):회귀 모델로 예측(보조적 단계)
#-----------------------------------------------------------------
# Step1 데이터 수집 : dataset : state.x77
# 데이터분석 기간 : 예측기간의 3~4배수 모델링 기간 확보 필요
#     - 모델용(훈련용): 4년치(Trend, Cycle, Seasonality)
#     - 평가용(검증용): 1년치(향후 1년 내 예측)
#
# 데이터분석 최소 샘플 수 확보
#     - 모델용(훈련용): w1
#     - 평가용(검증용): w2
#     w1 : w2 = 50%/60%/70%/80% : 50%/40%/30%/20%
#    ===
#     ㄴ{w1_1 : w1_2} : w2 = {40% : 40%} : 20%
#           ==> 모형 검증 2단계 (안정적 모형)
#------------------------------------
# crtl+enter : 실행, 
# crtl+shift++ : 확대 /crtl+ -  축소 
# alt+ +  : 자판 회복

# crtl+shift+c : 일괄주석처리 
# crtl+ s      : 저장하기 

# (Step2):데이터 탐색 및 준비, 파생변수 생성, 데이타셋 분리(모형개발용, 검증용, 평가용)
# 
# install.packages("dplyr") # 데이터 전처리 
# install.packages(c("tidyverse","ggplot2")) # 데이터 전처리  , 시각화 
# install.packages("psych") # 요약 통계량 
# install.packages("readxl") # 엑셀불러오기 
# install.packages("skimr")  # 데이타구조 분리, 요약 통계량 
# install.packages("tibble")
# dplyr::group_by()
library(tidyverse)
library(ggplot2)
library(dplyr)
library(readxl)
library(psych)
library(skimr)
library(tibble)

# (1) 데이터 불러오기 
data(state.x77) #  내장데이타셋 data()
data(iris)

state.x77
# View(state.x77) -> view창으로 확인
dim(state.x77)  #동작 명령어는 ()실행연산 별도 필요 X
# [1] 50  8
colnames(state.x77)

# 데이타 구조 변환전 : matrix형 
class(state.x77)
str(state.x77) #매트릭스 
# num [1:50, 1:8] 3615 365 2212 2110 21198 ...
# - attr(*, "dimnames")=List of 2
# ..$ : chr [1:50] "Alabama" "Alaska" "Arizona" "Arkansas" ...
# ..$ : chr [1:8] "Population" "Income" "Illiteracy" "Life Exp" ...
#-------------------------------------------------------------------------------
# (0)데이터 구조확인 : data.frame이어야만 함
#-------------------------------------------------------------------------------

class(state.x77)  # [1] "matrix" "array"
state.x77_df <- as.data.frame(state.x77)  # 분석을 위해 data.frame으로 변환
str(state.x77_df)
colnames(state.x77_df)
# [1] "Population" "Income"     "Illiteracy"
# [4] "Life Exp"   "Murder"     "HS Grad"   
# [7] "Frost"      "Area"  

plot(state.x77_df$Murder,  state.x77_df$Population)
# "Illiteracy"  <-> "Life Exp"

# 변수 위치 변환 : y변수를 첫 열로 이동
state.x77_df <- state.x77_df[, c(5, 1:4, 6:8)]
#-------------------------------------------------------------------------------
# (1) 시각화
#-------------------------------------------------------------------------------
par(mfrow=c(2,1))
boxplot(state.x77_df)
par(mfrow=c(1,1))

# 변수별 단위 표준화 : scale()
boxplot(scale(state.x77_df))

# 사분위 및 시각화 분석(boxplot)
colnames(state.x77_df)

# y(종속/타겟)변수 탐색적 분석 : 분포
skimr::skim(state.x77_df$Murder)
#     skim_variable n_missing complete_rate  mean    sd    p0   p25   p50   p75  p100 hist 
#   1 data                  0             1  7.38  3.69   1.4  4.35  6.85  10.7  15.1 ▆▇▃▇▂
quantile(state.x77_df[["Murder"]], p=c(.05, .25, .5, .75, .95))
#    5%    25%    50%    75%    95% 
# 2.300  4.350  6.850 10.675 12.885 
#               -----> 중앙값 (median)

# state.x77_df[["Murder"]] -> 리스트  (= state.x77_df[ ,"Murder"])

par(mfrow=c(1, 2))
hist(state.x77_df[["Murder"]], freq=F)
lines(density(state.x77_df[["Murder"]]), ylab="Murder")

boxplot(state.x77_df[["Murder"]], ylab="Murder")

#-------------------------------------------------------------------------------
# (2) x변수 분석
#-------------------------------------------------------------------------------

# 중요 x변수 중 일부
boxplot(state.x77_df[["Illiteracy"]], ylab="Illiteracy")
boxplot(state.x77_df[["Population"]]/1000000, ylab="Population (millions)")


#-------------------------------------------------------------------------------
# (3) 상관계수 cor(y, x)간 상관관계가 있어야 회귀분석이 가능함
#-------------------------------------------------------------------------------

# 한번에 체크 가능한 방법
psych::describe(state.x77_df)
psych::pairs.panels(state.x77_df[, c("Murder", "Illiteracy", "Population", "Area")])

cor(state.x77_df$Murder, state.x77_df$Illiteracy) # [1] 0.7029752 > 0.23 상관관계 있음
cor(state.x77_df$Murder, state.x77_df$Population) # [1] 0.3436428 > 0.23 상관관계 있음

# 상관관계 그래프
# install.packages("corrplot")
library(corrplot)


# 전체 상관계수 파악
cor(state.x77_df)
round(cor(state.x77_df), 2)
#            Murder Population Income Illiteracy Life Exp HS Grad Frost  Area
# Murder       1.00       0.34  -0.23       0.70    -0.78   -0.49 -0.54  0.23
# Population   0.34       1.00   0.21       0.11    -0.07   -0.10 -0.33  0.02
# Income      -0.23       0.21   1.00      -0.44     0.34    0.62  0.23  0.36
# Illiteracy   0.70       0.11  -0.44       1.00    -0.59   -0.66 -0.67  0.08
# Life Exp    -0.78      -0.07   0.34      -0.59     1.00    0.58  0.26 -0.11
# HS Grad     -0.49      -0.10   0.62      -0.66     0.58    1.00  0.37  0.33
# Frost       -0.54      -0.33   0.23      -0.67     0.26    0.37  1.00  0.06
# Area         0.23       0.02   0.36       0.08    -0.11    0.33  0.06  1.00


corrplot::corrplot(cor(state.x77_df), method="number")
corrplot::corrplot(cor(state.x77_df), method="pie") # x11()+ : active window
# help(corrplot)

options(scipen = 100) #  소숫점 형태로 출력   p-value = 0.00000001
# options(scipen = 1) -> 지수(e) 형태로 출력  p-value = 1e-08

cor.test(state.x77_df$Murder, state.x77_df$Illiteracy)


cor.test(state.x77_df$Murder, state.x77_df$Population)
# 상관관계(진짜 있는가?) 검증 (0.23에 근접하기 때문)

# Pearson's product-moment correlation
# 
# data:  state.x77_df$Murder and state.x77_df$Population
# t = 2.5352, df = 48, p-value = 0.01455 < 0.05(5%) (95% 신뢰구간) Ho(귀무=상관관계없음)기각
#                      =================   =====
#  [상관관계검증]    ㄴ p-value < 0.05 더 작을수록, 해당 x변수는 y변수와의 관계에 있어서 더 강한 상관관계를 갖는 변수이다.
#  [변변별력(예측력, 분류, 설명력)] ㄴ p-value < 0.05 더 작을수록, 해당 x변수는 y변수를 더욱 유의하게 설명력이 높은 변수이다.  ==> 독립x변수
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  0.07220124 0.56768971
# sample estimates:
#       cor 
# 0.3436428 


#-------------------------------------------------------------------------------
# (2) 다중회귀분석 모형화
#-------------------------------------------------------------------------------
# (setp 1) 완전모형 : 모든 x변수를 반영한 뒤 모형화한 것
# 선형회귀모델 돌려라 : lm() : lm = linear model 선형 모형
fit0 <- lm(Murder ~ 0 + Population+Illiteracy+Frost+Area,  # . : 모든 x변수를 투입 후 lm을 돌려라.
           #                  ㄴ 절편항 제외하라. +0 -1
           
           data = state.x77_df)
step(fit0)

fit0 <- lm(Murder~0+., data = state.x77_df)
summary(fit0)


# Call:
#   lm(formula = Murder ~ 0 + ., data = state.x77_df)
# 
# Residuals:  -> y(실제값) - y^(예측값) = e^(오차의 추정값) = 잔차(Residuals)
#   Min      1Q  Median      3Q     Max 
# -4.4282 -1.6601 -0.1824  1.5163  7.7378 
# 
# Coefficients:
#                Estimate   Std. Error t value Pr(>|t|)   
# Population  0.000219385  0.000092725   2.366  0.02256 * 
# Income      0.000063078  0.000820788   0.077  0.93910   
# Illiteracy  3.717076084  1.088679799   3.414  0.00141 **
# `Life Exp`  0.051798256  0.081521616   0.635  0.52854   
# `HS Grad`  -0.046492806  0.080523766  -0.577  0.56669   
# Frost      -0.000570170  0.010294594  -0.055  0.95609   
# Area        0.000009212  0.000005415   1.701  0.09611 . 
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.507 on 43 degrees of freedom
# Multiple R-squared:  0.9203,	Adjusted R-squared:  0.9073 
# F-statistic: 70.89 on 7 and 43 DF,  p-value: < 0.00000000000000022

fit0 <- lm(Murder~., data = state.x77_df)
summary(fit0)
# Call:
#   lm(formula = Murder ~ ., data = state.x77_df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.4452 -1.1016 -0.0598  1.1758  3.2355 
# 
# Coefficients:
#   Estimate    Std. Error t value     Pr(>|t|)    
# (Intercept) 122.180392646  17.886225407   6.831 0.0000000254 ***
#   Population    0.000188036   0.000064737   2.905      0.00584 ** 
#   Income       -0.000159207   0.000572530  -0.278      0.78232    
# Illiteracy    1.373109504   0.832202602   1.650      0.10641    
# `Life Exp`   -1.654869830   0.256211567  -6.459 0.0000000868 ***
#   `HS Grad`     0.032338308   0.057252663   0.565      0.57519    
# Frost        -0.012884070   0.007392415  -1.743      0.08867 .  
# Area          0.000005967   0.000003801   1.570      0.12391    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1.746 on 42 degrees of freedom
# Multiple R-squared:  0.8083,	Adjusted R-squared:  0.7763 
# F-statistic: 25.29 on 7 and 42 DF,  p-value: 0.0000000000003872

plot(fit0)


mean(state.x77$Illiteracy)
# ㄴ연산불가한 matrix형      ----->                           mean(state.x77_df$Illiteracy)
# invalid for atomic vectors   |
#                              |
state.x77_df <- as.data.frame(state.x77)


#-------------------------------------------------------------------------------
# 2. (output) 결과 활용한 분석내용 상세 검토 
names(fit0)
fitted.values(fit0) %>% data.frame() 
fitted.values(fit0)[1:2]



options(scipen=100)
fit0<- lm(formula = Murder ~ 0 +., data = state.x77_df);summary(fit0)

# # fit = 적합도가 높은 모형식= # fit = 적합도가 높은 모형식
#------------------------------------------------------------------------------
# lm(formula = Murder ~ . + 0, data = state_all_df)
#------------------------------------------------------------------------------

# e^
#------------------------------------------------------------------------------
# Residuals: y -y^ = e^ 
#   Min      1Q  Median      3Q     Max 
# -4.4282 -1.6601 -0.1824  1.5163  7.7378 
#------------------------------------------------------------------------------
# picture array (mainframe row =
# =       ==     =   =     ====
par(mfrow = c(2,2))
# y(Murder)와 유효한 설명관계를 가지는 경우
plot(state.x77_df$Illiteracy,state.x77_df$Murder)
plot(state.x77_df$Population,state.x77_df$Murder)

# y(Murder)와 유효한 설명관계를 가지지 않는 경우
plot(state.x77_df$Frost,state.x77_df$Murder)
plot(state.x77_df$`HS Grad`,state.x77_df$Murder)
#------------------------------------------------------------------------------

colnames(state.x77_df)


# Coefficients:   x

#                Estimate Std. Error t value Pr(>|t|)   
# Population    2.194e-04  9.273e-05   2.366  0.02256 * (95% CI에서 유의적 변수로 채택됨  )--> 제 2변수
#   Income      6.308e-05  8.208e-04   0.077  0.93910   
# Illiteracy    3.717e+00  1.089e+00   3.414  0.00141 **(99% CI에서 유의적 변수로 채택됨  ) --> 제 1 변수
#   `Life Exp`  5.180e-02  8.152e-02   0.635  0.52854   
# ` HS Grad`   -4.649e-02  8.052e-02  -0.577  0.56669   
#   Frost      -5.702e-04  1.029e-02  -0.055  0.95609   
#   Area        9.212e-06  5.415e-06   1.701  0.09611 . 
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#    ㄴ 유의 수준(a)
#        **’ 0.01      
#     Illiteracy(x변수)는 y(Murder)를 예측함에 유의적 의미를 가진다.
#
# Residual standard error: 2.507 on 43 degrees of freedom
#------------------------------------------------------------------------------
# Multiple R-squared:  0.9203,   Adjusted R-squared:  0.9073 
#     ㄴ R-squared = RSS / TSS               ㄴ= 1 - (n-1) / (n-p-1)(1-R^2)    * n = # obs , p = x/coefficient개수 

# │R^2 - adj-R^2│  => 다중공선성 심각도 체크 가능      
#                                        ㄴ: (최종)모형식명 car::vif(모형식명)
#------------------------------------------------------------------------------
0.9073-0.9203 # [1] -0.013

#------------------------------------------------------------------------------
# (step2) 1차 다중공선성 체크
# vif = variance(분산변동성) inflation(팽창) factor(계수) -> 분산팽창계수
car::vif(fit0)
# Population     Income Illiteracy `Life Exp`  `HS Grad` 
# 2.569229 107.429489  16.339008   265.677785 148.774788 
# Frost       Area 
# 11.431426   2.831257 
#------------------------------------------------------------------------------
# 예측/추정 모형식 회귀계수값의 신뢰도가 떨어지므로, 반드시 다중공선성 확인 후,
# car::vif(fit0)>10 되어야 채택변수가 됨.
#------------------------------------------------------------------------------
# (대응방안 : 다중공선성  원인 변수 제거하기)
#  (수정모형1) fit_1 : `Life Exp` 265.677785 제거한 모형
fit_1 <- lm(Murder~0 + Population + Income + Illiteracy + 
              `HS Grad` + Frost + Area, data = state.x77_df);summary(fit_1)

fit_1 <- lm(Murder~0 + .-`Life Exp`, data = state.x77_df);summary(fit_1)

# (수정모형1) fit_1 : `Life Exp` 265.677785 제거한 모형의 vif 확인
car::vif(fit_1)
# > car::vif(fit1)
# Population   Income Illiteracy  `HS Grad`      Frost       Area 
# 2.518393  93.820614   4.875907  74.799162   8.457070   1.849077 

#------------------------------------------------------------------------------
# (수정모형2) fit_2 : `Life Exp`, Income 제거한 모형
fit_2 <- lm(Murder~0 + Population + Illiteracy + 
              `HS Grad` + Frost + Area, data = state.x77_df);summary(fit_2)

fit_2 <- lm(Murder~0 + .-`Life Exp`- Income, data = state.x77_df);summary(fit_2)

# (수정모형2) fit_2 : `Life Exp`, Income 제거한 모형의 vif 확인
car::vif(fit_2)
# Population Illiteracy  `HS Grad`      Frost       Area 
# 2.069662   4.104877  16.395395   7.805363   1.843821

#------------------------------------------------------------------------------
# (수정모형3) fit_3 : `Life Exp`, Income, `HS Grad` 제거한 모형
fit_3 <- lm(Murder~0 + Population + Illiteracy  +
              Frost + Area, data = state.x77_df);summary(fit_3)

fit_3 <- lm(Murder~0 + .-`Life Exp`- Income - `HS Grad`,
            data = state.x77_df);summary(fit_3)
# Call:
#   lm(formula = Murder ~ 0 + . - `Life Exp` - Income - `HS Grad`, 
#      data = state.x77_df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.2807 -1.7676  0.0762  1.6760  7.3330 
# 
# Coefficients:
#   Estimate  Std. Error t value           Pr(>|t|)    
# Population 0.000247108 0.000075195   3.286            0.00195 ** 
#   Illiteracy 4.448956518 0.414623269  10.730 0.0000000000000413 ***
#   Frost      0.005190506 0.004219948   1.230            0.22495    
# Area       0.000007470 0.000004153   1.799            0.07863 .  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.44 on 46 degrees of freedom
# Multiple R-squared:  0.9192,	Adjusted R-squared:  0.9122 
# F-statistic: 130.9 on 4 and 46 DF,  p-value: < 0.00000000000000022

# (수정모형3) fit_3 : `Life Exp`, Income, `HS Grad` 제거한 모형의 vif 확인
car::vif(fit_3)
# Population Illiteracy      Frost       Area 
#   1.784327   2.502763   2.028539   1.758877 


a <- summary(fit_3)
a$adj.r.squared # [1] 0.9122005
1-a$adj.r.squared # [1] 0.08779945
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 2. 변수선택법
#------------------------------------------------------------------------------
# F-statistic: 70.89 on 7 and 43 DF,  p-value: < 2.2e-16
# (주의 사항)
# ㄴ 회귀식 후보 : 단, 채택변수 정리전임.
#     Murder = 3.717*Illiteracy  + 2.194e-04 * Population
#     ㄴ 최종 모형식 아님, 회귀계수도 아님. xxx.

#(정리 #1)
#------------------------------------------------------------------------------
# [예측값 검증] 모형식에서 해당변수만 입력후, 모델링 해야 
# 실제값과 예측값 정확히 비교가능함. 
#------------------------------------------------------------------------------
fit0<- lm(formula = Murder ~ 0 +Illiteracy+Population, data = state.x77_cor)
summary(fit0)
# Call:
#   lm(formula = Murder ~ 0 + Illiteracy + Population, data = state.x77_cor)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -4.403 -1.560  0.201  2.005  8.819 
# 
# Coefficients:
#              Estimate Std. Error t value             Pr(>|t|)    
#   Illiteracy 5.0340248  0.3625844   13.88 < 0.0000000000000002 ***
#   Population 0.0002775  0.0000779    3.56              0.00084 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.56 on 48 degrees of freedom
# Multiple R-squared:  0.907,   Adjusted R-squared:  0.903 
# F-statistic:  234 on 2 and 48 DF,  p-value: <0.0000000000000002


# 2. 변수선택법

fit_4 <- step(fit0)  # step() -> 불필요한 변수 제거

# Start:  AIC=96
# Murder ~ 0 + Illiteracy + Population
# 
#              Df Sum of Sq  RSS AIC(오차)
# <none>                     315  96
# - Population  1        83  398 106 <== [1] 10 AIC오차항 떨어뜨리는 데 기여 ==> 제 2변수 
# - Illiteracy  1      1264 1579 175 <== [1] 69 AIC오차항 떨어뜨리는 데 기여 ==> 제 1변수 
106-96   # [1] 10
175-106  # [1] 69

# Call:
#   lm(formula = Murder ~ 0 + Illiteracy + Population, data = state.x77_cor)
# 
# Coefficients:
# Illiteracy  Population  
# 5.034025    0.000278 


car::vif(fit_4);summary(fit_4)
# Population Illiteracy       Area 
#   1.758783   2.164385   1.575637 

#------------------------------------------------------------------------------
# 0929 숙제
#------------------------------------------------------------------------------
best_fit <- lm(formula = Murder ~ 0 + Illiteracy + Population, data=state.x77_df)
summary(best_fit)
# Call:
#   lm(formula = Murder ~ 0 + Illiteracy + Population, data = state.x77_df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.4026 -1.5603  0.2005  2.0052  8.8193 
# 
# Coefficients:
#   Estimate Std. Error t value             Pr(>|t|)    
#   Illiteracy 5.03402482 0.36258444  13.884 < 0.0000000000000002 ***
#   Population 0.00027752 0.00007788   3.563             0.000839 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.561 on 48 degrees of freedom
# Multiple R-squared:  0.9071,	Adjusted R-squared:  0.9032 
# F-statistic: 234.4 on 2 and 48 DF,  p-value: < 0.00000000000000022

# AIC값 확인
step(best_fit)
# Start:  AIC=96

# 다중공선성 체크
car::vif(best_fit)>10
# Illiteracy Population 
# FALSE      FALSE 

# install.packages("MASS")
library(MASS)
MASS::stepAIC(best_fit)
# Start:  AIC=96



#------------------------------------------------------------------------------

# 3.다중공선성 체크 
library(car)  # classfication and regression 분류회귀 패키지 내 
car::vif(fit0)
# Illiteracy Population 
#  1.737      1.737 
car::vif(fit0)>10 

#(정리 #2)
#----------------------------------------------------------------------
# 회귀결정식 # fit = 적합도가 높은 모형식
# y^= Murder = 5.0340248*Illiteracy + 0.0002775 * Population
#----------------------------------------------------------------------
fit0$fitted.values
head(state.x77_cor,1)

state.x77_cor[1,4]  # [1] 2.1
#          Illiteracy
# Alabama        2.1
state.x77_cor[1,3]  # [1] 3624

#          Life Exp Murder Population Income Illiteracy HS Grad Frost  Area
# Alabama    69.05   15.1       3615   3624        2.1    41.3    20 50708
y1_forecasts = 5.0340248*state.x77_cor[1,4] + 0.0002775* state.x77_cor[1,3] 
y1_forecasts = 5.0340248*2.1                + 0.0002775* 3624
y1_forecasts #  11.58


# 오차값1 = 실제y1값  - y1_forecasts
e1  =     15.1  -  y1_forecasts  
e1   # [1] 3.525

state.x77_cor[1,1]- fitted.values(fit0)[1]
# Alabama 
# 3.525

residuals(fit0)[1]
# Alabama 
# 3.525
#------------------------------------------------------------------------

fit0<- lm(formula = Murder ~ 0+., data = state.x77_df);summary(fit0)
summary(fit0)[1:10]

summary(fit0)$adj.r.squared # [1] 0.9073

summary(fit0)[9]
# $adj.r.squared
# [1] 0.9072788

# nrow(state.x77_df)
# ncol(state.x77_df)
#------------------------------------------------------------------------------
# Multiple R-squared: 0.9203 : 92.03% (모형에 의해 설명되는 양)

# Adjusted R-squared: 0.9073 : 90.73%
# Adjusted R-squared = 1-{(n-1)/(n-p-1)*(1-R^2)} * p= X 변수갯수
# =>=> p=모형식에 채택된 변수의 갯수
# =>=> p 모형에 투입되는 변수 개수가 커질수록 adjusted R-squared가 
#        낮아지는 패널티 부여됨
#        (R-squared가 허수로서 높아진다)
# Adjusted R-squared : 우선적으로 볼 것!
# =>=> 다중공선성 (분석시 이 부분 체크 필수)

# (다중공선성으로 인해 (변수를 많이 넣었을 때)
# 개별독립변수가 고유하게 Y를 설명(선형linear)하지 못해
# 예측치에 대한 신뢰성이 떨어짐을 뜻함)

# 모형에 채택된 변수의 수(=p)가 많으면 adjusted R-squared 를 먼저 봐야함
# adjusted R-squared minus R-squared의 값의 차이(1.3%로 낮음)를 봐야함
# *** (값의 갭이 크면 반드시 다중공선성을 체크 해야함)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                           [(n-1)*(1- R-squared)]
# Adjusted R-squared = 1 -  ----------------------
#                              [n- 1 - p ]                 p= x 변수 갯수 
#  x변수 갯수인 p가 많을수록 Adjusted R-squared 값이 떨어진다.
#               ===========> 패널티 부여하는 것임. 
#                              ===> 차이값 클수록 다중공선성 문제 의심됨. 
#    |R-squared- Adjusted R-squared | =  
0.9203 - 0.9073  # [1] 0.013
#------------------------------------------------------------------------------
# cor(y,x) 상관관계 있어야 함이 확인됨.
# state.x77_df <- state.x77_cor
cor(state.x77_df$Population, state.x77_df$Murder) # [1] 0.7029752
cor.test(state.x77_df$Population, state.x77_df$Murder)
# t = 2.5352, df = 48, p-value = 0.01455 < 0.05 이므로 차이없다는 Ho 기각, 상관 관계 있음 
# ==> 독립변수로 y(Murder)를 설명하는 채택변수가 됨.

cor(state.x77_df$Illiteracy, state.x77_df$Murder) # [1] 0.3436428
cor.test(state.x77_df$Illiteracy, state.x77_df$Murder)
# t = 6.8479, df = 48, p-value = 1.258e-08 < 0.05 이므로 차이없다는 Ho 기각, 상관 관계 있음 
# ==> 독립변수로 y(Murder)를 설명하는 채택변수가 됨.

#-------------------------------------------------------------------------------
# 다중 공선성 체크 #1 : cor(x_i, x_j) ~0  ( cor.test로 확인요망)
#-------------------------------------------------------------------------------
# best_fit : x변수간 다중공선성 없음을 검증 : cor.test()
#-------------------------------------------------------------------------------
cor(state.x77_df$Population, state.x77_df$Illiteracy) # [1] 0.1076224
cor.test(state.x77_df$Population, state.x77_df$Illiteracy)

#            Pearson's product-moment correlation
# 
# data:  state.x77_df$Population and state.x77_df$Illiteracy
# t = 0.74999, df = 48, p-value = 0.4569 > 0.05 이므로 차이없다는 Ho 채택 
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  -0.1759976  0.3747441
# sample estimates:
#       cor 
# 0.1076224   

#(분석결과 해석)-----------------------------------------------------------
# p-value = 0.4569 > 0.05 이므로 차이없다는 Ho 채택
# ==> 두 x변수(Illiteracy,Population)간 상관관계가 없고, 독립적이다. 
# ==> 독립변수로 y(Murder)를 설명하는 
#     채택변수로 두 x변수(Illiteracy,Population)는 유효함.
#------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# 다중 공선성 체크 #2: car::vif(fit0)
#-------------------------------------------------------------------------------
colnames(state.x77_df)
state.x77_df_2 = state.x77_df[ , c(1,4,2)]
best_fit<- lm( Murder ~ 0+., data = state.x77_df_2);summary(best_fit)
# Call:
#   lm(formula = Murder ~ 0 + ., data = state.x77_df_2)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.4026 -1.5603  0.2005  2.0052  8.8193 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
#   Population 2.775e-04  7.788e-05   3.563 0.000839 ***
#   Illiteracy 5.034e+00  3.626e-01  13.884  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.561 on 48 degrees of freedom
# Multiple R-squared:  0.9071,   Adjusted R-squared:  0.9032 
0.9032-0.9071  # [1] -0.0039 ==> 차이 미미=다중공선성 이슈 없을 것으로 보여짐

# F-statistic: 234.4 on 2 and 48 DF,  p-value: < 2.2e-16

car::vif(best_fit)
# Population Illiteracy 
# 1.736845   1.736845    # 다중공선성 이슈 없음

car::vif(best_fit)<10 
# Population Illiteracy 
# TRUE       TRUE       # 다중공선성 이슈 없음

best_fit<- lm( Murder ~ 0+Population+Illiteracy, data = state.x77_df_2); summary(best_fit)
summary(best_fit)

# (결론)
#------------------------------------------------------------------------------
# (1) 예측 회귀선 : y^
#     y^= Murder = 0.0002775*Population +  5.0340248*Illiteracy 
#------------------------------------------------------------------------------
# (2)위 예측회귀식이 전체변동성을 약 90% 설명해 준다. 
#          (=Multiple R-squared:  0.9071,   Adjusted R-squared:  0.9032)
#                         ㄴ MSS/TSS = 90%/100%
#                            Model Sum of sq / Total Sum of Sq
# TSS = MSS + ESS = 90% + 10% 
#   Y = ^Y  +  e
#       ㄴMurder = 0.0002775*Population +  5.0340248*Illiteracy  

# SUM(y-^y)^2/(N-1) = TSS
#------------------------------------------------------------------------------
# (3) 위 예측회귀식이 p-value: < 2.2e-16 < 0.05 이므로, 95% 신뢰구간하에서
#     유의한 모형식이다.
#------------------------------------------------------------------------------


#----------------------------------------------------------
# 모형화 방법 #2)
#----------------------------------------------------------
fit=lm(Murder ~ 0+Illiteracy + Population, data=state.x77_df)
summary(fit)->a
# 모형
colnames(state.x77_df) -> n
n[3];n[1]
summary(fit)[1]
a$coefficients->b
b[1:2]
paste0("다중회귀분석 예측모형식 : Murder(미국50개 주별 살인율)",
       b[1],"*",n[3],"(문맹율)"," + ",
       b[2],"*",n[1],"(주별 인구수)")

# 잔차
a$residuals

#--------------------------------------------------
# (2) 다중공선성 문제 없어야  cor(x_i,x_j) ~0 (<0.23)  
# 각각의 독립변수 x가 y를 고유하게 독립적으로 설명 
#--------------------------------------------------


library(car)
car::vif(fit)
# >   car::vif(fit)
# Population Illiteracy 
# 1.736845   1.736845 
car::vif(fit)>10
# >   car::vif(fit)>10
# Population Illiteracy 
# FALSE      FALSE 

R_squared=0.9 # R_squared >= 90% 다중공선성 이슈 발생할 가능성이 높다. 
(vif=1/(1-R_squared)) # [1] 10



# 4개 잔차도 
par(mfrow=c(2,2))
plot(fit)
# 상관도
psych::pairs.panels(state.x77_df)
psych::pairs.panels(state.x77_df_1)


b <- residuals(fit)
shapiro.test(b)
# Shapiro-Wilk normality test
# 
# data:  b
# W = 0.96648, p-value = 0.1658
#=================== 다중 회귀분석 종료 ==========================

#--------------------------------------------------------
# 다중선형 회귀식(확정)
#    Murder = 2.194e-04*Population + 3.717 *Illiteracy
#     (Multiple R-squared:  0.9203,   Adjusted R-squared:  0.9073 )
# 
#    (2) 다중공선성 문제 해결됨
# Population Illiteracy 
# 1.736845   1.736845
#--------------------------------------------------------
# Call:
#   lm(formula = Murder ~ 0 + ., data = state.x77_df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.4282 -1.6601 -0.1824  1.5163  7.7378 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)   
# Population  2.194e-04  9.273e-05   2.366  0.02256 * 
# Income      6.308e-05  8.208e-04   0.077  0.93910   
# Illiteracy  3.717e+00  1.089e+00   3.414  0.00141 **
# `Life Exp`  5.180e-02  8.152e-02   0.635  0.52854   
# `HS Grad`  -4.649e-02  8.052e-02  -0.577  0.56669   
# Frost      -5.702e-04  1.029e-02  -0.055  0.95609   
# Area        9.212e-06  5.415e-06   1.701  0.09611 . 
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.507 on 43 degrees of freedom
# Multiple R-squared:  0.9203,   Adjusted R-squared:  0.9073 
# F-statistic: 70.89 on 7 and 43 DF,  p-value: < 2.2e-16
#--------------------------------------------------


# partition(skim(df)) : 데이터형별 분리  ==> chr형을 일괄적으로 factor형으로 변경 가능하게 함. 
skimr::partition(skim(state.x77))



# $numeric
# -- Variable type: numeric ---------------------------------------------------------------------------------------
#  A tibble: 8 x 11
# skim_variable n_missing complete_rate     mean        sd     p0       p25      p50      p75     p100 hist 
# * <chr>             <int>         <dbl>    <dbl>     <dbl>  <dbl>     <dbl>    <dbl>    <dbl>    <dbl> <chr>
# 1 Population            0             1  4246.    4464.     365    1080.     2838.    4968.    21198   ▇▂▁▁▁
# 2 Income                0             1  4436.     614.    3098    3993.     4519     4814.     6315   ▃▅▇▂▁
# 3 Illiteracy            0             1     1.17     0.610    0.5     0.625     0.95     1.58      2.8 ▇▃▂▂▁
# 4 Life Exp              0             1    70.9      1.34    68.0    70.1      70.7     71.9      73.6 ▃▃▇▅▅
# 5 Murder                0             1     7.38     3.69     1.4     4.35      6.85    10.7      15.1 ▆▇▃▇▂
# 6 HS Grad               0             1    53.1      8.08    37.8    48.0      53.2     59.2      67.3 ▅▂▇▆▃
# 7 Frost                 0             1   104.      52.0      0      66.2     114.     140.      188   ▅▃▅▇▆
# 8 Area                  0             1 70736.   85327.    1049   36985.    54277    81162.   566432   ▇▁▁▁▁


# 데이타구조 변환 : data.frame으로 데이타 구조 변환
state.x77_df <- as.data.frame(state.x77)
str(state.x77_df)

state.x77_typepart <- partition(skim(state.x77_df))
state.x77_typepart$numeric
# $numeric
# 
# -- Variable type: numeric ---------------------------------------------------------------------------------------
#   # A tibble: 8 x 11
#   skim_variable n_missing complete_rate     mean        sd     p0       p25      p50      p75     p100 hist 
# * <chr>             <int>         <dbl>    <dbl>     <dbl>  <dbl>     <dbl>    <dbl>    <dbl>    <dbl> <chr>
# 1 Population            0             1  4246.    4464.     365    1080.     2838.    4968.    21198   ▇▂▁▁▁
# 2 Income                0             1  4436.     614.    3098    3993.     4519     4814.     6315   ▃▅▇▂▁
# 3 Illiteracy            0             1     1.17     0.610    0.5     0.625     0.95     1.58      2.8 ▇▃▂▂▁
# 4 Life Exp              0             1    70.9      1.34    68.0    70.1      70.7     71.9      73.6 ▃▃▇▅▅
# 5 Murder                0             1     7.38     3.69     1.4     4.35      6.85    10.7      15.1 ▆▇▃▇▂
# 6 HS Grad               0             1    53.1      8.08    37.8    48.0      53.2     59.2      67.3 ▅▂▇▆▃
# 7 Frost                 0             1   104.      52.0      0      66.2     114.     140.      188   ▅▃▅▇▆
# 8 Area                  0             1 70736.   85327.    1049   36985.    54277    81162.   566432   ▇▁▁▁▁         


state.x77_typepart$numeric          
state.x77_typepart$charcter  # NULL  <==== character형 변수 없음

#-------------------------------------------------
sum(is.character(state.x77_df))
# 논리연산자 TRUE = 1 , FALSE = 0

# 데이터 구조 변환
class(state.x77) # [1] "matrix" 형/구조/타입
state.x77_df <- as.data.frame(state.x77)
class(state.x77_df) # [1] "data.frame"
state.x77_mat <- as.matrix(state.x77_df); class(state.x77_mat)

library(base)


skimr::skim(state.x77_df)

# 행갯수 확인 : 
nrow(state.x77_df)  # [1] 50

a <- 1:100
a
quantile(a)
boxplot(a)
sample(a)
plot(a)

state.x77_df
s <- sample(1:50) 

set.seed(42) # 난수발생 초기값지정
# 행(번호)추출 50개 중 35개를 비복원추출하라 
sl <- base::sample(1:nrow(state.x77_df),  # 전체 행번호 
                   nrow(state.x77_df)*0.7,      # 추출 횟수  35개를
                   # ================
                   #        50        
                   replace = FALSE)             # 비복원 추출 (중복 배제 )

#     ======> 랜덤추출= random sampling = 난수발생

help(sample)

# 데이터 분리 : 모형개발용/훈련용 train,  검증용 test <= 모형 평가 

train <- state.x77_df[ sl,   ]; nrow(train);dim(train)
test <-  state.x77_df[-sl,   ]; nrow(test);dim(test)

# 행렬    
str(state.x77)    # num [1:50, 1:8]
str(state.x77_df) # 'data.frame':   50 obs. of  8 variables:
# [ 행, 열 ]  행렬, 데이터프레임  : num [1:50, 1:8]
# { }   패키지 = 함수의 집합
ggplot2::geom_boxplot()
# ()   함수

# 열갯수 확인
ncol(state.x77_df)  # [1] 8

str(state.x77_df)
# 'data.frame':   50 obs. of  8 variables:

#-------------------------------------------------------------------------------
# 연속형 변수 --> 그룹형 변수로 파생변수 생성
table(state.x77_df$Income);
quantile(state.x77_df$Income)
#   0%  25%  50%  75% 100% 
# 3098 3993 4519 4814 6315 
state.x77_df$Income_gr <- ifelse(state.x77_df$Income>=4814, "inc_gr_4814+",
                                 ifelse(state.x77_df$Income>=4519, "inc_gr_4519_4814",
                                        ifelse(state.x77_df$Income>=3993, "inc_gr_3993_4519","inc_gr_under3993" )))
#-----------------------------------------------------------------------------------------------------------------
# (참고용)  파이프라인 연산자  %>%  역할  : 객체 승계
#-----------------------------------------------------------------------------------------------------------------
# pipeline operator (파이프연산자 : ) = crtl+shift+m %>% 
# magrittr::`%>%`()

state.x77_df %>% data.frame() %>% head()
#         df %>%  function1()  %>% function1() %>% ......
# 왼쪽에서 지정된 df을 %>% 이후, 각 함수들이 연결해 재사용하게 하는 역할 
#-----------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 결측치(NA) 갯수 체크 : is.na() : 결측치 확인 논리연산자 
#-------------------------------------------------------------------------------
#(1) 전체적 확인 
table(is.na(state.x77))
# FALSE  : 0 으로 인식,   TRUE : 1로 인식
# 400

is.na(state.x77_df) %>% data.frame()
# tibble::tibble()
is.na(state.x77_df) %>% tibble::tibble()

#  # A tibble: 50 x 1
# .[,"Murder"] [,"Population"] [,"Income"]
#   <lgl>        <lgl>           <lgl>      
# 1 FALSE        FALSE           FALSE      
# 2 FALSE        FALSE           FALSE      
# 3 FALSE        FALSE           FALSE      
# 4 FALSE        FALSE           FALSE      
# 5 FALSE        FALSE           FALSE      
# 6 FALSE        FALSE           FALSE      
# 7 FALSE        FALSE           FALSE      
# 8 FALSE        FALSE           FALSE      
# 9 FALSE        FALSE           FALSE      
# 10 FALSE        FALSE           FALSE      
# # ... with 40 more rows

sum(is.na(state.x77_df)) # [1] 0

#-------------------------------------------------------------------------------
#(2) 개별 변수별 확인 
colSums(is.na(state.x77_df))
# Murder Population     Income Illiteracy   Life Exp 
#  0          0          0          0          0 
# HS Grad      Frost       Area  Income_gr 
#  0          0          0          0 

#-------------------------------------------------------------------------------
#(3) 패키지를 활용한 확인

skimr::skim(state.x77_df)
# -- Data Summary ------------------------
# Values      
# Name                       state.x77_df
# Number of rows             50          
# Number of columns          9           
# _______________________                
# Column type frequency:                 
#   character                1           
# numeric                  8           
# ________________________               
# Group variables            None        
# 
# -- Variable type: character --------------------------------------------------------------------------------
#   # A tibble: 1 x 8
#   skim_variable n_missing complete_rate   min   max empty n_unique whitespace
# * <chr>             <int>         <dbl> <int> <int> <int>    <int>      <int>
#   1 Income_gr             0             1    12    16     0        4          0
# 
# -- Variable type: numeric ----------------------------------------------------------------------------------
#   # A tibble: 8 x 11
#   skim_variable n_missing complete_rate     mean        sd     p0       p25      p50      p75     p100 hist 
#                 =========
#                 결측치 수 

# * <chr>             <int>         <dbl>    <dbl>     <dbl>  <dbl>     <dbl>    <dbl>    <dbl>    <dbl> <chr>
# 1 Murder                0             1     7.38     3.69     1.4     4.35      6.85    10.7      15.1 ▆▇▃▇▂
# 2 Population            0             1  4246.    4464.     365    1080.     2838.    4968.    21198   ▇▂▁▁▁
# 3 Income                0             1  4436.     614.    3098    3993.     4519     4814.     6315   ▃▅▇▂▁
# 4 Illiteracy            0             1     1.17     0.610    0.5     0.625     0.95     1.58      2.8 ▇▃▂▂▁
# 5 Life Exp              0             1    70.9      1.34    68.0    70.1      70.7     71.9      73.6 ▃▃▇▅▅
# 6 HS Grad               0             1    53.1      8.08    37.8    48.0      53.2     59.2      67.3 ▅▂▇▆▃
# 7 Frost                 0             1   104.      52.0      0      66.2     114.     140.      188   ▅▃▅▇▆
# 8 Area                  0             1 70736.   85327.    1049   36985.    54277    81162.   566432   ▇▁▁▁▁

pastecs::stat.desc(state.x77_df$Murder)[3]
# nbr.na = # of missing value (결측치 갯수 )
# nbr.na 
#   0 

# 요약
psych::describe(state.x77_df)
#            vars  n     mean       sd   median  trimmed      mad     min      max     range  skew kurtosis       se
# Population    1 50  4246.42  4464.49  2838.50  3384.28  2890.33  365.00  21198.0  20833.00  1.92     3.75   631.37
# Income        2 50  4435.80   614.47  4519.00  4430.08   581.18 3098.00   6315.0   3217.00  0.20     0.24    86.90
# Illiteracy    3 50     1.17     0.61     0.95     1.10     0.52    0.50      2.8      2.30  0.82    -0.47     0.09
# Life Exp      4 50    70.88     1.34    70.67    70.92     1.54   67.96     73.6      5.64 -0.15    -0.67     0.19
# Murder        5 50     7.38     3.69     6.85     7.30     5.19    1.40     15.1     13.70  0.13    -1.21     0.52
# HS Grad       6 50    53.11     8.08    53.25    53.34     8.60   37.80     67.3     29.50 -0.32    -0.88     1.14
# Frost         7 50   104.46    51.98   114.50   106.80    53.37    0.00    188.0    188.00 -0.37    -0.94     7.35
# Area          8 50 70735.88 85327.30 54277.00 56575.72 35144.29 1049.00 566432.0 565383.00  4.10    20.39 12067.10
#                 ㄴ== 관측치의 갯수



# 전체 매트릭스내 변수간 cor 상관관계 계산 가능
# (핵심)-------------------------------------------------------
psych::pairs.panels(state.x77_df)
#--------------------------------------------------------------

colnames(state.x77_df)
state.x77_df= state.x77_df[ , c(5, 1:4, 6:8)]

# (과거 방식)
cor(state.x77_df, method = "pearson")

cor(state.x77_df, use = "everything",
    method = c( "spearman")) # "pearson", "kendall"

help(cor)
cor(state.x77_df)[2:8]
# 아리까리 할 때 ~ cor.test() 통해 상관관계 있는지여부 체크한다. 


#-----------cor(y,x) 개별 상관관계 검증 ----------------------------------------
# Illiteracy 는 y변수인 Murder 상관관계 0.7029752로 강한양의 상관관계는
# 유의한 의미<있>음 (p-value = 1.258e-08 <0.05 (95% ci) 차이없다는 Ho reject(기각))
#-------------------------------------------------------------------------------
cor(state.x77_df$Murder, state.x77_df$Illiteracy) # [1] 0.7029752
cor.test(state.x77_df$Murder, state.x77_df$Illiteracy)
# Pearson's product-moment correlation
# data:  state.x77_df$Murder and state.x77_df$Illiteracy
# t = 6.8479, df = 48, p-value = 0.00000001 <0.05이므로,
#                     95% ci 하에서 상관관계가 없다는 Ho을 기각함.
#                     Murder와 Illiteracy는 상관관계가 있음.

# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  0.5279280 0.8207295
# sample estimates:
#       cor 
# 0.7029752 

#-----------cor(y,x) 개별 상관관계 검증 ----------------------------------------
# Population 는 y변수인 Murder 상관관계 0.7029752로 강한양의 상관관계는
# 유의한 의미<있>음 (p-value = 1.258e-08 <0.05 (95% ci) 차이없다는 Ho reject(기각))
#-------------------------------------------------------------------------------
cor(state.x77_df$Murder, state.x77_df$Population) # [1] 0.3436428
cor.test(state.x77_df$Murder, state.x77_df$Population)
# Pearson's product-moment correlation
# data:  state.x77_df$Murder and state.x77_df$Population
# t = 2.5352, df = 48, p-value = 0.01455 <0.05이므로,
#                     95% ci 하에서 상관관계가 없다는 Ho을 기각함.
#                     Murder와 Population 상관관계가 있음.

# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  0.07220124 0.56768971
# sample estimates:
#       cor 
# 0.3436428 
#-------------------------------------------------------------------------------
#
#-----------cor(y,x) 개별 상관관계 검증 ----------------------------------------
# Area 는 y변수인 Murder 상관관계 0.2283902로 
# p-value = 0.1106 > 0.05 (95% ci) 차이없다는 Ho accept(채택))
# ==> Murder와 Area 두 변수간 상관관계 없다. 
#-------------------------------------------------------------------------------
cor(state.x77_df$Murder, state.x77_df$Area) # [1] 0.2283902
cor.test(state.x77_df$Murder, state.x77_df$Area )
# Pearson's product-moment correlation
# data:  state.x77_df$Murder and state.x77_df$Area
# t = 1.6253, df = 48, p-value = 0.1106 >0.05 이므로,
# (95% ci) 차이없다는 Ho accept(채택))
# ==> Murder와 Area 두 변수간 상관관계 없다. 

# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  -0.47782468  0.05157292
# sample estimates:
#        cor 
# 0.2283902  


help(cor)
class(state.x77_df$Income_gr)

state.x77
state.x77_df <- data.frame(state.x77)

state.x77_df_temp <- state.x77_df
quantile(state.x77_df_temp$Murder)


# 2021.6.15

#--------------------------------------------------------------------------------
# 이진형 변수화를 통한 카이제곱검증과 F검증 비교 
#--------------------------------------------------------------------------------
# (실험 : y변수를 이진형/이산형 변수로 변환 후 비교하기)
#--------------------------------------------------------------------------------
# (I) 사전 셋팅 
#(1) ( 이진형 변수 구분 기준 수립 방안)
#--------------------------------------------------------------------------------
# 평균
m <- mean(state.x77_df_temp$Murder) # [1] 7.378 

#중앙값
q3 <- quantile(state.x77_df_temp$Murder)[3] # 6.85 = median = 중앙값 
q3 <- quantile(state.x77_df_temp$Murder)[] # 6.85 = median = 중앙값 
#--------------------------------------------------------------------------------
# (2) 신규변수 생성 : Murder_gr : y 이진형변수(binary)
class(state.x77_df_temp$Murder) # [1] "numeric"
state.x77_df_temp$Murder
# 새로운 변수 Murder_gr : ifelse 조건문으로 신규변수 생성 
state.x77_df_temp$Murder_gr=  ifelse(state.x77_df_temp$Murder>= m , 1, 0)
#                                                              === 평균기준 y 이분화
#--------------------------------------------------------------------------------
# (참조)분포를 plot으로 그려보기
par(mfrow=c(1,1)) #  plot 배치 위치 
boxplot(state.x77_df_temp$Murder) # 박스플롯 그리기 

table(state.x77_df_temp$Income_gr)
# 1000 3993 4519 4814 
# 13   12   12   13 

#--------------------------------------------------------------------------------
# (II-1)  그룹변수 생성 : X변수 그룹화 예시 
state.x77_df_temp$Income_gr <- ifelse(state.x77_df_temp$Income>=4814 |state.x77_df_temp$Income <3993 ,
                                      "4814+ or under3993",
                                      ifelse(state.x77_df_temp$Income>=4519, "4519_4814","3993_4519" ))
state.x77_df_temp$Income_gr2 <- ifelse(state.x77_df_temp$Income>=4814 |state.x77_df_temp$Income <3993 ,
                                       "4814+ or under3993",
                                       "3993_4814" )

cor.test(state.x77_df_temp$Murder_gr, state.x77_df_temp$Income_gr )

#--------------------------------------------------------------------------------
# (II-2)  결과 분석 
gmodels::CrossTable(state.x77_df_temp$Murder_gr, state.x77_df_temp$Income,
                    prop.r = T, chisq = T)
# Pearson's Chi-squared test 
# ------------------------------------------------------------
# Chi^2 =  50     d.f. =  49     p =  0.4334 > 0.05 (95% ci), Ho 채택

length(state.x77_df_temp$Income)


gmodels::CrossTable(state.x77_df_temp$Murder_gr, state.x77_df_temp$Income_gr,
                    prop.r = T, chisq = T)
gmodels::CrossTable(state.x77_df_temp$Murder_gr, state.x77_df_temp$Income_gr2,
                    prop.r = T, chisq = T)
quantile(state.x77_df_temp$Murder_gr)
# Total Observations in Table:  50 
# 
# 
#                        | state.x77_df_temp$Income_gr2 
# state.x77_df_temp$Murder_gr 
#   |          3993_4814 | 4814+ or under3993 |          Row Total | 
# -----------------------|--------------------|--------------------|--------------------|
# 0 |                 17 |                 10 |                 27 | 
#   |              1.259 |              1.163 |                    | 
#   |              0.630 |              0.370 |              0.540 | 
#   |              0.708 |              0.385 |                    | 
#   |              0.340 |              0.200 |                    | 
#------------------------|--------------------|--------------------|--------------------|
# 1 |                  7 |                 16 |                 23 | 
#   |              1.478 |              1.365 |                    | 
#   |              0.304 |              0.696 |              0.460 | <== 
#   |              0.292 |              0.615 |                    | 
#                 ======   vs           ======

#   |              0.140 |              0.320 |                    | 
# -----------------------|--------------------|--------------------|--------------------|
#   Column Total 
#   |                 24 |                 26 |                 50 | 
#   |              0.480 |              0.520 |                    | 
# -----------------------|--------------------|--------------------|--------------------|
#   
#   
#   Statistics for All Table Factors
# 
#                   ~ N(E(X),sigma(X))  정규성 
#  x1, x2, x3,..... ~ z(0,1)  가정하, 분산차이 검증 Chi-squared test 

# Pearson's Chi-squared test 
# ------------------------------------------------------------
# Chi^2 =  5.265     d.f. =  1     p =  0.02176 <0.05 (95% ci) Ho 기각, 
# Murder_gr에 영향여부에 관한 카이제곱검증에서, Income_gr2의 그룹간 차이 유의 함.
# 채택변수임.
# 
# Pearson's Chi-squared test with Yates' continuity correction 
# ------------------------------------------------------------
# Chi^2 =  4.042     d.f. =  1     p =  0.04437 

#-------------------------------------------------------------------------------
# 2021.6.18~ 시작할 부분~  
mean(state.x77$Population)
# Error in state.x77$Population : $ operator is invalid for atomic vectors

summary(state.x77)
# 전체 매트릭스내 summary 계산 가능
summary(state.x77$Population)
# 전체 매트릭스내 부분 변수간summary 계산 불가

head(state.x77,10)
#             Population Income Illiteracy Life Exp Murder HS Grad Frost   Area
# Alabama           3615   3624        2.1    69.05   15.1    41.3    20  50708
# Alaska             365   6315        1.5    69.31   11.3    66.7   152 566432
# Arizona           2212   4530        1.8    70.55    7.8    58.1    15 113417
# Arkansas          2110   3378        1.9    70.66   10.1    39.9    65  51945
# California       21198   5114        1.1    71.71   10.3    62.6    20 156361
# Colorado          2541   4884        0.7    72.06    6.8    63.9   166 103766
# Connecticut       3100   5348        1.1    72.48    3.1    56.0   139   4862
# Delaware           579   4809        0.9    70.06    6.2    54.6   103   1982
# Florida           8277   4815        1.3    70.66   10.7    52.6    11  54090
# Georgia           4931   4091        2.0    68.54   13.9    40.6    60  58073

#-------------------------------------------------------------------
# Xs 변수명 확인 : colnames()
#-------------------------------------------------------------------

colnames(state.x77)
# Y변수찾기 : [5] Murder
# [1] "Population" "Income"    
# [3] "Illiteracy" "Life Exp"  
# [5] "Murder"     "HS Grad"   
# [7] "Frost"      "Area"

# (1) 기본 matrix 형 state.x77
str(state.x77)
# num [1:50, 1:8] 3615 365 2212 2110 21198 ...


#------------------------------------------------
# (2) 데이터셋 분리 단계
#     추가적으로 data.frame형 생성
#     matrix에서 분석에 더 편리한 data.frame형으로 변환함
#------------------------------------------------


# 완전모형 : 모든 x후보변수들을 다 넣고, 선형회귀분석(lm = Linear Regression) 실행 
colnames(state.x77)
# [1] "Population" "Income"    
# [3] "Illiteracy" "Life Exp"  
# [5] "Murder"     "HS Grad"   
# [7] "Frost"      "Area"

class(state.x77)

state.x77_df <- as.data.frame(state.x77)

psych::pairs.panels(state.x77_cor)


#=======================================================================
# () 밖에서는 <- / = 동일 : 오른쪽을 왼쪽 df/벡터.. 할당/생성시켜라.
# () 안에서는  =  ( <- 변수명이 이상한 것이 나옴)  cf  ==  같다.

a <- c(rep(1,5), rep(10,5),100000);a  #벡터 a  생성하라. 
b<- c(rep(2,5), rep(20,5),19);b

ab_df = data.frame(a,b) ;ab_df

ab_mat= matrix(1:10, 2)
class(ab_mat)

class(ab)
library(dplyr)
library(tidyverse)
ab_df %>% dplyr::filter(a == 10 & b==20) %>% head(2)

#=======================================================================

#열 배열 순서 바꾸기
colnames(state.x77)
# [1] "Population" "Income"     "Illiteracy"
# [4] "Life Exp"  
# [5] "Murder"  <===  y변수   
# [6] "HS Grad"   
# [7] "Frost"      "Area"
state.x77_cor = state.x77[  , c(5,1:4,6:8)]    # matrxit 구조 : 연산/회귀분석 등 불가 
state.x77_cor = state.x77_df[, c(5,1:4,6:8)] # dataframe 구조 : 연산/회귀분석 등 가능 
psych::pairs.panels(state.x77_cor)

# y = 취업자수(^ = 실업자수^ = 직장인+자영업자)
# x = 연령, 학력, 지역, (외생변수=GDP, 이벤트변수=코로나)

# r 함수 : 원칙 소문자.
# 예외

options(scipen = 100)  # options(scipen = 1) 다시 e 지수형으로 전환됨. 
quantile(a)
IQR(a)

# 이상치 (상한치) 있으면 "upper_outlier" 표기하라. 
ifelse(a> quantile(a)[4] + 1.5*IQR(a), "upper_outlier" , "이상치상한 없음")
#          -------------
#                 Q3
# [1] "이상치상한 없음" "이상치상한 없음" "이상치상한 없음"
# [4] "이상치상한 없음" "이상치상한 없음" "이상치상한 없음"
# [7] "이상치상한 없음" "이상치상한 없음" "이상치상한 없음"
# [10] "이상치상한 없음" "upper_outlier"

# 이상치의 해당 위치를 밝혀라 : 해당인덱스 번호 
which(a> quantile(a)[4] + 1.5*IQR(a))  # [1] 11
# ㄴ 해당인덱스 번호 

par(mfrow=c(1,1))
boxplot(state.x77_df)

dev.off()
#----------------------------------------------------------------

par(mfrow = c(2,2))
plot(best_fit)


#-----------------------------------------------------------------------------
# Residuals 가 0이라는 뜻 : Y - ^y = 0 (Y = ^y) : 
# Residuals(검정색 원) 의  Fitted (=오차 예측선(붉은선))에 딱 붙어 있는 것
#-----------------------------------------------------------------------------
# (검토결과)
# (1) 잔차들의 분포를 본 것 : 안정적임 ==> 동분산성 보여짐 
# (단,  Nevada(오차 거의 9정도),  Rhode Islands (오차 거의 -5정도)
# (이상치=특이치=outlier인지 의심이 됨. 이상치 확정 아니나 체크 필요)
# 잔차들 사이 영향이 있는지 패턴이 안보임( 독립성=자기상관관계 없어 보여짐 )

# e값이 (residuals) 독립성유지, 동분산성 가짐, 정규성 가짐, 자기상관성 X
#----------------------------------------------------
# Nevada 가 이상치라는 것 확인하기: car::outlierTest() (검증) 
#                                     ㄴcar 이라는 패키지의 outlierTest 함수 사용

# R. Global Environment 및 Data 아래 fit(회귀모형식명) 아래 클릭시, 
# 회귀계수,회귀분석모형 변수, 잔차(residuals= e^)/ 잔차값 확인 가능함.
# fit[["residuals"]][["Nevada"]]  : 7.62로 이상치/특이치 체크 가능함.

#----------------------------------------------------
#이상(관측)치 (outlier) : 
# 이상치(오류여부 등) 체크 후 이상한 경우 삭제 후 재모형화 하면
# ***(기대효과)
#     "이상한(의미 없는/입력 오류치 등)" 이상치 / 특이치를 제거 후 
#      모형 예측력(model accuracy) 증가함. * 값의 패턴이 혼자 달라도 actual 값이면 살려두어야 함.
#----------------------------------------------------

car::outlierTest(best_fit)
#        rstudent unadjusted p-value Bonferroni p 
# Nevada 3.936675         0.00027177     0.013588 < 0.05 이므로 95%CI하에서, 
# ㄴ Nevada가 이상치가 아니다/차이없다라는 H0기각한다.
#     즉, Nevada는 이상치 이다.

#----------------------------------------------------------------

state_all_matrix <- as.matrix(state.x77_df)
str(state_all_matrix)
class(state_all_matrix)
# Nevada observation(행)확인
# (1) 전체로 찾기
rownames(state_all_matrix) # [28] "Nevada"

# (2) 관측치명으로 바로 위치찾기
which(rownames(state_all_matrix)=="Nevada") # [1] 28

# Nevada observation(행) 삭제
state_matrix_outlier <- state_all_matrix[-28,]  

rownames(state_matrix_outlier)  # Nevada삭제확인

#----------------------------------------------------------------
# (step5):모형 성능(정확도)개선 : 평가결과 모형을 추가적 개선
# best_fit <- lm(Murder ~ 0+Population+Illiteracy, data = state.x77_df)
# 모형개선 : outlier Nevada 삭제 후 재모형화
best_fit_exc_outlier <- lm(Murder ~ 0+Population+Illiteracy,
                           data = state_matrix_outlier)
# Error in model.frame.default(formula = Murder ~ 0 + Population + Illiteracy,  : 
#                     'data' must be a data.frame, not a matrix or an array
# -> 연산가능한 data.frame형태로 변환해야함

# dataframe으로 변경 및 확인
state_matrix_exe_outlier_df <- as.data.frame(state_matrix_outlier)
class(state_matrix_exe_outlier_df) # [1] "data.frame"

# dataframe으로 변경 후 모형 개선 : outlier Nevada 삭제 후 재모형화

best_fit_exc_outlier <- lm(Murder ~ 0+Population+Illiteracy,
                           data = state_matrix_exe_outlier_df)
summary(best_fit_exc_outlier)
# Call:
#   lm(formula = Murder ~ 0 + Population + Illiteracy, data = state_matrix_outlier_df)
# 
# Residuals:
#   Min     1Q Median     3Q    Max
# -4.313 -1.538  0.170  1.892  4.038 
# 
# Coefficients:
#   Estimate Std. Error t value             Pr(>|t|)    
# Population 0.00028510 0.00006828   4.176             0.000128 ***
#   Illiteracy 4.95995533 0.31831622  15.582 < 0.0000000000000002 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.244 on 47 degrees of freedom
# Multiple R-squared:  0.9273,	Adjusted R-squared:  0.9242 
# F-statistic: 299.8 on 2 and 47 DF,  p-value: < 0.00000000000000022
#----------------------------------------------------------------
# (모형 개선 효과) : 모형 성능(정확도) 개선 : 수정결정계수 2.08% 추가상승
#
#   (2) (개선 후) 0.924 - (1) (개선 전) 0.9032 = 0.0208% ( 2.08% 개선)
#----------------------------------------------------------------
#   (1) 이상치 Nevada제거 전:Adjusted R-requared: 0.9032
#   (2) 이상치 Nevada제거 후:Adjusted R-requared: 0.924

#----------------------------------------------------------------
# 이상치 제거 후 best_fit_exc_outlier 모형에 대한 검증
#----------------------------------------------------------------
car::vif(best_fit_exc_outlier)
# Population Illiteracy 
# 1.737906   1.737906   ==> 다중공선성 문제 없음.


summary(best_fit_exc_outlier)[1:10]
summary(best_fit_exc_outlier)$coefficients
summary(best_fit_exc_outlier)$coefficients[1:2]

# install.packages("MASS")
library(MASS)
MASS::stepAIC(best_fit_exc_outlier)
  # Start:  AIC=81.19
  # Murder ~ 0 + Population + Illiteracy
  # 
  # Df Sum of Sq     RSS     AIC
  # <none>                     236.76  81.185
  # - Population  1     87.83  324.59  94.646
  # - Illiteracy  1   1223.05 1459.80 168.318
  # 
  # Call:
  #   lm(formula = Murder ~ 0 + Population + Illiteracy, data = state_matrix_outlier_df)
  # 
  # Coefficients:
  #   Population  Illiteracy  
  # 0.0002851   4.9599553  


#-------------------------------------------------------------------------------
# 모형화 방법 #1)
#-------------------------------------------------------------------------------
summary(best_fit_exc_outlier) -> a
# 모형
colnames(state.x77_df) -> n
n[4];n[2]

a$coefficients->b
b[1:2]
paste0("다중회귀분석 예측모형식 : Murder(미국50개) 주별 살인율 = ",
       b[1],"*",n[4],"(문맹율)"," + ",
       b[2],"*",n[2],"(주별 인구수)")

# 잔차
a$residuals
#-------------------------------------------------------------------------------
# 최종모형회귀식
# Murder= 4.95995532717906 * Illiteracy + 0.000285097520190714 * Population
#-------------------------------------------------------------------------------

# y실제값 = y_actuals
state.x77_df$Murder[1]
head(state.x77_df, 1)[1]
# Murder
# Alabama   15.1


# y^ 예측값 <-- Murder= 4.95995532717906 * Illiteracy + 0.000285097520190714 * Population 로 산출된 예측값
best_fit_exc_outlier$fitted.values[1]
# Alabama 
# 11.44653

# residuals = e^ = y - y^
residuals_Alabama = state.x77_df$Murder[1] - best_fit_exc_outlier$fitted.values[1]
residuals_Alabama
# Alabama 
# 3.653466 

colnames(state.x77_df)

state.x77_df[1,4] # [1] 2.1

state.x77_df[1,2] # [1] 3615

state.x77_df[1,]
#         Murder Population Income Illiteracy Life Exp HS Grad Frost  Area
# Alabama   15.1       3615   3624        2.1    69.05    41.3    20 50708

b=best_fit_exc_outlier$coefficients
b
# Illiteracy   Population 
# 4.9599553272 0.0002850975 

b[1];b[2]
# Illiteracy 
# 4.959955 
# Population 
# 0.0002850975

# y^ 예측값 산출과정
y1_forecasts = b[1]*state.x77_df[1,4] + b[2]*state.x77_df[1,2]
y1_forecasts # 11.44653 

# 잔차값(오차값 예측치1) =   y(실제y1값)               - y^(y1_forecasts)
e1_hat                   = state.x77_df$Murder[1]      - y1_forecasts
e1_hqt  #   3.653466 

residuals(best_fit_exc_outlier)[1]
# Alabama 
# 3.653466 
plot(best_fit_exc_outlier)

# 실제값, 예측값
y_실제값 = state_matrix_exe_outlier_df$Murder
y_예측값 = best_fit_exc_outlier$fitted.values
잔차값_오차예측값 = best_fit_exc_outlier$residuals

statenames <- rownames(state_matrix_exe_outlier_df)

final_outpout <- data.frame(statenames, y_실제값, y_예측값, 잔차값_오차예측값)
setwd("C:/0.0_다중회귀분석")
getwd()
write.csv(final_outpout, file="data/final_outpout.csv", row.names = F)

#-------------------------------------------------------------------------------
# 변수 선택 기법 : step()
#-------------------------------------------------------------------------------
#                   | <- 상수항 부터 ~ 설명 변수 추가 ..... <-  full 모형|
# (1) 전진 선택    (forward selection) ->................................|
# (2) 후진 소급                                   <- (backward selection)
# (3) 단계적 선택                       (stepwise selection)
#-------------------------------------------------------------------------------

#===============================================================================
# 최종모형
state_matrix_exc_outlier_df <- state.x77_df_2[-28,]
dim(state_matrix_exc_outlier_df)    # [1] 49  3
#                                         ㄴ Nevada가 삭제된 후 df
colnames(state_matrix_exc_outlier_df)   # [1] "Murder"     "Illiteracy" "Population"
fit_state_df_exc_outlier <- lm(Murder ~ 0+., data = state_matrix_exc_outlier_df)

summary(fit_state_df_exc_outlier)
summary(fit_state_df_exc_outlier)$coefficients
#===============================================================================
# 와 별개로

# 오차항에 대한 4가지 가정 점검하면 확정모형으로 결정이 남.
colnames(state.x77_df_exc_outlier)
# 이상치인 Nevada는 제외했으나, 모든 x변수를 고려하여 변수 선택기법을 적용하고자 함

state.x77_df_exc_outlier <- state.x77_df[-28, ]

dim(state.x77_df_exc_outlier) # [1] 49  8 <== full모형 = 완전모형에 들어갈 변수를 모두 갖고있음
#                                   ㄴ Nevada가 삭제된 후 df
#---------------------------------------------------------------------------------------------------
# (1) 모형식명 : [fit_full] full 모형 = 완전모형 fit_full
# (2) df : [state.x77_df_exc_outlier] state.x77_df_exc_outlier <- state.x77_df[-28, ]
#---------------------------------------------------------------------------------------------------

# (1단계 : 완전모형 생성)

fit_full <- lm(Murder~ 0+., data=state.x77_df_exc_outlier)
summary(fit_full)=colnames(state.x77_df_exc_outlier)
# [1] "Murder"     "Population" "Income"     "Illiteracy"
# [5] "Life Exp"   "HS Grad"    "Frost"      "Area" 
#-------------------------------------------------------------------------------

# (1-1) 전진선택법 1단계 : 절편항(상수항) 부터 추가
# full모형 = 완전모형
str(state.x77_df_exc_outlier)
fit.intcpt <- lm(Murder ~ 1, data = state.x77_df_exc_outlier)
# ㄴ 절편항(Intercept)만 분석 : +1, 1(0은 불가)만 추가 
# vs 절편항 삭제 : .-1, .+0 추가
# state.x77 df_exc_outlier
# ㄴ원천 데이터셋 - 이상치 Nevada 제외 후
#     데이터셋 state.x77_df_exc_outlier
summary(fit.intcpt)
# Call:
#   lm(formula = Murder ~ 1, data = state.x77_df_exc_outlier)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -5.8939 -2.9939 -0.4939  3.3061  7.8061 
# 
# Coefficients:
#             Estimate    Std. Error   t value              Pr(>|t|)    
# (Intercept)   7.2939        0.5259     13.87   <0.0000000000000002 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 3.681 on 48 degrees of freedom


#(1-2) 전진선택법 2단계 : 설명변수 1개씩 추가

fit.forward <- step(fit_full, # (1) full-완전모형 
                    scope = list(lower = fit.intcpt, # (0) 상수항1개... 변수추가...
                                 upper = fit_full)) # (full-완전모형)              
# Start:  AIC=82.85
# Murder ~ 0 + (Population + Income + Illiteracy + `Life Exp` + 
#                 `HS Grad` + Frost + Area)
# 
#              Df Sum of Sq    RSS    AIC
# - Income      1     0.619 200.35 81.005
# - Frost       1     4.054 203.79 81.838
# <none>                    199.74 82.853
# - `HS Grad`   1     9.865 209.60 83.216
# - `Life Exp`  1    15.194 214.93 84.446
# - Area        1    28.226 227.96 87.330
# - Population  1    39.155 238.89 89.625
# - Illiteracy  1    46.369 246.10 91.083
# 
# Step:  AIC=81.01
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Frost + 
#   Area - 1
# 
#              Df Sum of Sq    RSS    AIC
# - Frost       1     4.100 204.45 79.998
# <none>                    200.35 81.005
# - `HS Grad`   1    12.550 212.90 81.982
# - `Life Exp`  1    15.085 215.44 82.562
# + Income      1     0.619 199.74 82.853  <== + : 제거된 변수 : 기 제거된 변수로 재투입방지 기록
# - Area        1    27.994 228.35 85.413
# - Population  1    40.619 240.97 88.050
# - Illiteracy  1    48.245 248.60 89.577
# 
# Step:  AIC=80
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Area -1
#         ========================   -----------------------------
#              Df Sum of Sq    RSS     AIC
# <none>                    204.45  79.998
# - `HS Grad`   1     9.354 213.81  80.190
# - `Life Exp`  1    11.087 215.54  80.585
# + Frost       1     4.100 200.35  81.005  # 2번째 제거된 변수
# + Income      1     0.665 203.79  81.838  # 1번째 제거된 변수
# - Area        1    24.065 228.52  83.450
# - Population  1    58.391 262.84  90.307
# - Illiteracy  1   122.240 326.69 100.963
# ㄴ 마이너스가 모형식에 반영될 변수임


# (2단계 분석결과 : 전진 선택법 분석 결과 선택된 변수)
# Population + Illiteracy + `Life Exp` + `HS Grad` + Area
fit.forward = lm(Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Area - 1, data=state.x77_df_exc_outlier)
summary(fit.forward)
# Coefficients:
#                Estimate   Std. Error t value   Pr(>|t|)    
#   Population  0.000247358  0.000069779   3.545   0.000945 ***
#   Illiteracy  3.640662671  0.709813651   5.129 0.00000629 ***
#   `Life Exp`  0.084764377  0.054876140   1.545   0.129594    
#   `HS Grad`  -0.090765080  0.063973302  -1.419   0.163004    
#   Area        0.000009811  0.000004311   2.276   0.027784 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.156 on 44 degrees of freedom
# Multiple R-squared:  0.9372,	Adjusted R-squared:  0.9301 
# F-statistic: 131.4 on 5 and 44 DF,  p-value: < 0.00000000000000022

# 다중공선성 확인 -> 한계확인
car::vif(fit.forward)>10
# Population Illiteracy `Life Exp`  `HS Grad`       Area 
#      FALSE      FALSE       TRUE       TRUE      FALSE
car::vif(fit.forward)
# Population Illiteracy `Life Exp`  `HS Grad`       Area 
#   1.967819   9.368274 159.757865 123.277922   2.379520


# (3단계)다중공선성 해결 방안 도출
# `Life Exp`의 다중공선성이 159.757865로 가장 높아 제외 후 재모형화

fit.forward_1 <- lm(Murder ~ Population + Illiteracy + `HS Grad` + Area - 1, 
                    data=state.x77_df_exc_outlier)
summary(fit.forward_1)
# Call:
#   lm(formula = Murder ~ Population + Illiteracy + `HS Grad` + Area - 
#        1, data = state.x77_df_exc_outlier)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -3.996 -1.562  0.075  1.730  4.114 
# 
# Coefficients:
#                 Estimate  Std. Error t value          Pr(>|t|)    
#   Population 0.000257850 0.000070509   3.657          0.000666 ***
#   Illiteracy 4.505536069 0.442935239  10.172 0.000000000000304 ***
#   `HS Grad`  0.006296387 0.012186176   0.517          0.607908    
#   Area       0.000006555 0.000003818   1.717          0.092895 .  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.189 on 45 degrees of freedom
# Multiple R-squared:  0.9338,	Adjusted R-squared:  0.9279 
# F-statistic: 158.8 on 4 and 45 DF,  p-value: < 0.00000000000000022
car::vif(fit.forward_1)
# Population Illiteracy  `HS Grad`       Area 
# 1.949175   3.538975   4.339591   1.810526 

fit.forward_2 <- lm(Murder ~ Population + Illiteracy + Area - 1, 
                    data=state.x77_df_exc_outlier)
summary(fit.forward_2)
# Call:
#   lm(formula = Murder ~ Population + Illiteracy + Area - 1, 
#                data = state.x77_df_exc_outlier)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.9012 -1.6548  0.0973  1.8461  4.0514 
# 
# Coefficients:
#                 Estimate  Std. Error t value             Pr(>|t|)    
#   Population 0.000269143 0.000066500   4.047             0.000196 ***
#   Illiteracy 4.648480300 0.343140551  13.547 < 0.0000000000000002 ***
#   Area       0.000007266 0.000003533   2.057             0.045414 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.171 on 46 degrees of freedom
# Multiple R-squared:  0.9334,	Adjusted R-squared:  0.9291 
# F-statistic:   215 on 3 and 46 DF,  p-value: < 0.00000000000000022
car::vif(fit.forward_2)
# Population Illiteracy       Area 
#   1.761882   2.158329   1.575241 

# (4단계)상관관계확인 cor.test()
#       ㄴ (3단계에서 유효하다고 본Area에 대해 상관관계가 없음이 확인됨)

cor.test(state.x77_df$Murder, state.x77_df$Illiteracy) #      cor 0.7029752 
cor.test(state.x77_df$Murder, state.x77_df$Population) #      cor 0.3436428 
cor.test(state.x77_df$Murder, state.x77_df$Area) #      cor 0.2283902  < 0.23 이므로 상관관계가 없다고판단
# 따라서 Area 제외.

fit.forward_3 <- lm(Murder ~ Population + Illiteracy - 1, 
                    data=state.x77_df_exc_outlier)
summary(fit.forward_3)
# Call:
#   lm(formula = Murder ~ Population + Illiteracy - 1, 
#                data = state.x77_df_exc_outlier)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -4.313 -1.538  0.170  1.892  4.038 
# 
# Coefficients:
#                Estimate Std. Error t value             Pr(>|t|)    
#   Population 0.00028510 0.00006828   4.176             0.000128 ***
#   Illiteracy 4.95995533 0.31831622  15.582 < 0.0000000000000002 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.244 on 47 degrees of freedom
# Multiple R-squared:  0.9273,	Adjusted R-squared:  0.9242 
# F-statistic: 299.8 on 2 and 47 DF,  p-value: < 0.00000000000000022
car::vif(fit.forward_3)
# Population Illiteracy 
#   1.737906   1.737906 


# stat.x77_df_exc_outlier : df for (1) full-완전모형
colnames(state.x77_df_exc_outlier)
# [1] "Murder"     "Population" "Income"     "Illiteracy" "Life Exp"   "HS Grad"   
# [7] "Frost"      "Area"


#---------------------------------------------------------------------------------------------------
# (2) 후진 변수 선택/제거법 (backward selection/elimination)
# 모든 변수가 포함된 모델(full모형)에서
# 기준 통계치에 가장 도움이 되지 않는
# 변수를 1개씩 제거하는 방법
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# (1) 모형식명 : [fit_full] full 모형 = 완전모형 fit_full
# (2) df : [state.x77_df_exc_outlier] state.x77_df_exc_outlier <- state.x77_df[-28, ]
#---------------------------------------------------------------------------------------------------

# (1단계 : 완전모형 생성)

fit_full <- lm(Murder~ 0+., data=state.x77_df_exc_outlier)
summary(fit_full)=colnames(state.x77_df_exc_outlier)
# [1] "Murder"     "Population" "Income"     "Illiteracy"
# [5] "Life Exp"   "HS Grad"    "Frost"      "Area" 


# 2단계
fit.backward <- step(fit_full, # (1) full-완전모형 
                     scope = list(lower = fit.intcpt,
                                  upper = fit_full),
                     direction = "backward") # 후진소거법
# Start:  AIC=82.85
# Murder ~ 0 + (Population + Income + Illiteracy + `Life Exp` + 
#                 `HS Grad` + Frost + Area)
# 
#              Df Sum of Sq    RSS    AIC
# - Income      1     0.619 200.35 81.005
# - Frost       1     4.054 203.79 81.838
# <none>                    199.74 82.853
# - `HS Grad`   1     9.865 209.60 83.216
# - `Life Exp`  1    15.194 214.93 84.446
# - Area        1    28.226 227.96 87.330
# - Population  1    39.155 238.89 89.625
# - Illiteracy  1    46.369 246.10 91.083
# 
# Step:  AIC=81.01
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Frost + 
#   Area - 1
# 
#              Df Sum of Sq    RSS    AIC
# - Frost       1     4.100 204.45 79.998
# <none>                    200.35 81.005
# - `HS Grad`   1    12.550 212.90 81.982
# - `Life Exp`  1    15.085 215.44 82.562
# - Area        1    27.994 228.35 85.413
# - Population  1    40.619 240.97 88.050
# - Illiteracy  1    48.245 248.60 89.577
# 
# Step:  AIC=80
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Area - 
#   1
# 
#              Df Sum of Sq    RSS     AIC
# <none>                    204.45  79.998
# - `HS Grad`   1     9.354 213.81  80.190
# - `Life Exp`  1    11.087 215.54  80.585
# - Area        1    24.065 228.52  83.450
# - Population  1    58.391 262.84  90.307
# - Illiteracy  1   122.240 326.69 100.963
#---------------------------------------------------------------------------------------------------
# (3) 단계적선택기법(stepwise selection)
# 모든 변수가 포함된 모델에서 출발하여
# 기준 통계치에서 도움이 되지 않는 변수를 삭제하거나
# 모델에서 빠져있는 변수 중에서 기존 통계치를 가장 개선 시키는 변수를 추가한다.
# 이러한 변수의 추가 또는 삭제를 반복
# 절편만 포함된 모델에서 출발하여 변수의 추가, 삭제를 반복할 수도 있다.
#---------------------------------------------------------------------------------------------------
# (1) 모형식명 : [fit_full] full 모형 = 완전모형 fit_full
# (2) df : [state.x77_df_exc_outlier] state.x77_df_exc_outlier <- state.x77_df[-28, ]
#---------------------------------------------------------------------------------------------------

# (1단계 : 완전모형 생성)
fit_full <- lm(Murder~ 0+., data=state.x77_df_exc_outlier)
summary(fit_full)=colnames(state.x77_df_exc_outlier)
# [1] "Murder"     "Population" "Income"     "Illiteracy"
# [5] "Life Exp"   "HS Grad"    "Frost"      "Area" 

# 2단계
fit.backward <- step(fit_full, # (1) full-완전모형 
                     scope = list(lower = fit.intcpt,
                                  upper = fit_full),
                     direction = "both")
# Start:  AIC=82.85
# Murder ~ 0 + (Population + Income + Illiteracy + `Life Exp` + 
#                 `HS Grad` + Frost + Area)
# 
#              Df Sum of Sq    RSS    AIC
# - Income      1     0.619 200.35 81.005
# - Frost       1     4.054 203.79 81.838
# <none>                    199.74 82.853
# - `HS Grad`   1     9.865 209.60 83.216
# - `Life Exp`  1    15.194 214.93 84.446
# - Area        1    28.226 227.96 87.330
# - Population  1    39.155 238.89 89.625
# - Illiteracy  1    46.369 246.10 91.083
# 
# Step:  AIC=81.01
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Frost + 
#   Area - 1
# 
# Df Sum of Sq    RSS    AIC
# - Frost       1     4.100 204.45 79.998
# <none>                    200.35 81.005
# - `HS Grad`   1    12.550 212.90 81.982
# - `Life Exp`  1    15.085 215.44 82.562
# + Income      1     0.619 199.74 82.853
# - Area        1    27.994 228.35 85.413
# - Population  1    40.619 240.97 88.050
# - Illiteracy  1    48.245 248.60 89.577
# 
# Step:  AIC=80
# Murder ~ Population + Illiteracy + `Life Exp` + `HS Grad` + Area - 
#   1
# 
# Df Sum of Sq    RSS     AIC
# <none>                    204.45  79.998
# - `HS Grad`   1     9.354 213.81  80.190
# - `Life Exp`  1    11.087 215.54  80.585
# + Frost       1     4.100 200.35  81.005
# + Income      1     0.665 203.79  81.838
# - Area        1    24.065 228.52  83.450
# - Population  1    58.391 262.84  90.307
# - Illiteracy  1   122.240 326.69 100.963
# ---------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------
# II . 오차항에 관한 4가지 가정 체크 
#----------------------------------------------------------------------------------
# (1) 회귀 진단: 회귀 모형화 전 사전 진단 필요 *회귀식 사용이 가능한 문제인지 진단
#----------------------------------------------------------------------------------
#  y  =  y^  + e^
# 100 = 90*  + 10
# 100 = 95** + 5      ==> 모형정확도(Accuracy)      *90% --> **95% (+5%증가)하면, 
#                             오차율(낮을수록 좋음) *10% --> ** 5% (-5%감소)한다.
#===============================================================================
# 회귀진단 : 회귀모형이 과연 적절(유효성 검증) 한가?
#===============================================================================

# <최소(오차)제곱법(OLS)>를 사용하는 선형모형은 가장 오래된 통계방법의 하나이다. 
#  ㄴ min(sum(y-y^)^2)  <오차제곱합최소화>시키는 [β^] 찾기 
#             ----
#              (e^)^2
#
# (다중)회귀모형의 검증이 왜 필요한가? 
# (1) 오차 불규칙성 등으로 인해, 관측값/데이터가 불규칙하거나 
# (2) 반응변수(Y변수)와 예측변수(X변수) 사이의 관계를 잘못 설정한 경우 
# 부정확한 통계 모형을 만들 수 있다. 

# 실제로 반응변수(Y변수)와 예측변수(X변수) 사이에 
# (1) 유효한 설명/예측 관계가 있음에도 불구하고 관계가 없다고 결론내리거나 
# (2) 반응변수(Y변수)와 예측변수(X변수) 사이에 아무런 관계가 없음에도 불구하고 
#     유효한 설명/예측 관계가 있다고 결론을 내려 

# (old dataset) 과거데이타이용해 추정한/예측한 (추정모형식(Y^)을 만들었어도,  
# (new dataset) 새로운 데이타셋 적용(실제 real-world)에 그 모형을 적용해보면 
#               예측이 빗나가 불필요한 심각한 오류를 만들 수 있기에, 이를 검증하기 위해,
#                ㄴ(새로운 데이타셋 적용해서 재검증하여, 모형정확도가 강건하게  유지되도록 
#                  예측/추정 모형식의  유효성을  검증하는 것임.  
#-------------------------------------------------------------------------------
# 오차항에 대한 4가지 가정: e i.i.d~ N( ,Sigma^2) : 
#   e ~i(indenendantly:독립성).i(identically:동분산성).d  Normal(정규성)( ,Sigma^2)
#                           
# (1) 독립성 : i=independtly (오차와 x변수간 상관 관계가 없다는 의미)
# (2) "자기: 상관 관계가 없을 것 (예 e(t-1)이 e(t)의 방향성에 영향을 미치면 안됨). 
#     오차에 패턴이 있으면 안됨.

# (3) 동분산성: 일정한 분산 Sigma^2 (<-분산이 일정하다는 의미)
#                                  i.i.d
# (4) 정규성: 정규 분포를 따름 e^   ~   N( ,Sigma^2)

#---------------------------------------------------------------------- 
#-------------------------------------------------------------
# (0) 잔차(오차의 예측값 = y - y^)의 등분산성 그래프 확인
#-------------------------------------------------------------
par(mfrow = c(1, 1))
plot(fit_state_df_exc_outlier) # Nevada(이상치) 삭제 후 최종모형

# (0) Nevada 삭제 후 2차 잔차의 이상치(outlier) 확인
car::outlierTest(fit_state_df_exc_outlier)
  # No Studentized residuals with Bonferroni p < 0.05
  # Largest |rstudent|:
  #               rstudent unadjusted p-value Bonferroni p
  # Rhode Island -2.011475           0.050155           NA

#-------------------------------------------------------------------------------------------------------------------  
#-------------------------------------------------------------
# (1) 잔차의 독립성 체크
# car::durbinWatsonTest(fit_res)
#-------------------------------------------------------------

fit_res <- residuals(fit_state_df_exc_outlier)

car::durbinWatsonTest(fit_res) # fit_res = 잔차식명 (주의)
# [1] 2.152136
# (결론) DW ~ 2 : 잔차 간 자기 상관관계 없음 = 잔차 독립성 만족

car::durbinWatsonTest(fit_state_df_exc_outlier)
# lag Autocorrelation D-W Statistic p-value
# 1        -0.1350228      2.152136   0.504 > 0.05 이므로 
# 95% 신뢰구간하에서 잔차 간 자기상관관계 없다는 Ho채택함. 
# 즉, 잔차 간 자기 상관관계 없음. 잔차 간 독립성을 만족함
# Alternative hypothesis: rho != 0


# (결론) 잔차 간 자기 상관관계 없음.
#       잔차 독립성 만족 <== 다중회귀모형 성립조건 중 하나

#-------------------------------------------------------------------------------------------
# DW 검증 통계량(Durbin-Watson test Value) 해석
# DW ~ 0 : 잔차 간 양의 자기상관관계
# DW ~ 2 : 잔차 간 자기 상관관계 없음 = 잔차 독립성 만족  
#            ㄴ 다중회귀모형 성립조건 중 하나
# DW ~ 4 : 잔차 간 음의 자기상관관계
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------  
# (2) 잔차의 자기상관성 체크:  e^ = residuals
#     Durbin-Watson test Value와 p-value로 판단
#-------------------------------------------------------------------------------------------------------------------
options(digits = 4)
# install.packages("lmtest")
library(lmtest)
lmtest::dwtest(fit_state_df_exc_outlier)
# Durbin-Watson test
# 
# data:  fit_state_df_exc_outlier
# DW = 2.1521, p-value = 0.7199 > 0.05 (95% 신뢰구간하) 잔차 간 자기상관관계 없다는 Ho채택함.
# (결론) 잔차 간 자기 상관관계 없음.
# alternative hypothesis: true autocorrelation is greater than 0
#-------------------------------------------------------------------------------------------
# DW 검증 통계량(Durbin-Watson test Value) 해석
# DW ~ 0 : 잔차 간 양의 자기상관관계
# DW ~ 2 : 잔차 간 자기 상관관계 없음 = 잔차 독립성 만족  
#            ㄴ 다중회귀모형 성립조건 중 하나
# DW ~ 4 : 잔차 간 음의 자기상관관계
#-------------------------------------------------------------------
# (3) 잔차의 동분산성 체크 : ncvTest() 
#                      : Non-<constant Variance> Test
#                                 동   분산
#-------------------------------------------------------------------
# 
# install.packages("tidyverse")
# install.packages("reshape")
# install.packages("car")
library(tidyverse)
library(reshape)
library(car)

ncvTest(fit_state_df_exc_outlier)
# Non-constant Variance Score Test 
# Variance formula: ~ fitted.values 
# Chisquare = 0.01083335, Df = 1, p = 0. > 0.05 

# p값이 0.05보다 크므로 귀무가설(H0) 채택영역에 들어
#                       -> 동분산성을 가정할 수 있다.
# H0: 동분산성 특이하지 않다(=문제 없다) <== 귀무가설(H0)
# H1: 동분산성 특이하다(=문제 있다 )
# (결론) H0 채택, 동분산성을 만족함.

# 등분산성(Homoscedasticity): 분산이 일정하다는 가정을 만족한다면 
# 그림에서 잔차(Residuals)값이 0을 기준으로 
# 수평선 주위의 random band 형태로 나타나야 함. 
# 해당 그래프에서는 등분산성을 만족하는 것으로 보임. 



#----------------------------------------------------------------------
# (4) 잔차의 정규성 체크 : shapiro.test()
#----------------------------------------------------------------------

fit_res <- residuals(fit_state_df_exc_outlier)
shapiro.test(fit_res)
# Shapiro-Wilk normality test
# 
# data:  fit_res
# W = 0.97565, p-value = 0.7
# p값이 0.05보다 크므로 H0 채택영역에 들어,
#      귀무가설(H0) 채택 -> 정규성을 가정할 수 있다.
# H0: 정규성 특이하지 않다 (정규성 만족한다)
# H1: 정규성 특이하다
# H0 채택





