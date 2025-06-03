library(mgcv)
library(foreign) 
library(lavaan)
library(tidyverse)
library(ggplot2)
library(ggthemes)
library(haven)
library(lavaanPlot)
library(knitr)
library(psych)
library(semPlot)
library(RColorBrewer)
library(semTools)
library(wesanderson)
library(dmetar)


dfo <- read.csv("data_git.csv") %>%
df = dfo
df <- data.frame(scale(df[names(df) %in% c('source', 'presentation',  'NRecorder', 'context_variability_re', 'n_rep', 'modality_re', 'learn_mode', 'feedback_re2_ana', 'explicitness_instruction',   'training_duration', 'trial_number', 'consolidation', 'es' )]))
df <- df %>% rename(modality = modality_re)



# effect size distribution

vis <- data.frame(dfo[names(dfo) %in% c('source', 'presentation',  'NRecorder', 'context_variability_re', 'n_rep', 'modality_re', 'learn_mode', 'feedback_re2_ana', 'explicitness_instruction',   'training_duration', 'trial_number', 'consolidation', 'es' )])

categorical_vars <- c('source', 'presentation', 'context_variability_re', 'n_rep', 'modality_re', 'learn_mode', 'feedback_re2_ana', 'explicitness_instruction')
continuous_vars <- c('NRecorder', 'training_duration', 'trial_number', 'consolidation')
vis[categorical_vars] <- lapply(vis[categorical_vars], as.factor)

p_values_cat <- sapply(categorical_vars, function(var) {
  model <- aov(es ~ get(var), data = vis)
  p_value <- summary(model)[[1]][["Pr(>F)"]][1]  
  return(p_value)
})

p_values_continuous <- sapply(continuous_vars, function(variable) {
  df_t <- na.omit(vis[, c("es", variable)])
  median_value <- median(df_t[[variable]], na.rm = TRUE)
  df_t$category <- ifelse(df_t[[variable]] <= median_value, "Low", "High")
  model <- aov(es ~ category, data = df_t)
  p_value <- summary(model)[[1]][["Pr(>F)"]][1]  
  return(p_value)
})

palette <- brewer.pal(3, "Reds") #change color
darker_palette <- scales::alpha(palette, alpha = 0.8) 
variable_1 <- c( 'explicitness_instruction' # 'learn_task', 'feedback', 'explicitness_instruction, 
                 # context_var, repetition, sound_type, reptition, modality, presentation
)

for (variable in variable_1) {
  df_t <- na.omit(vis[, c("es", variable)])
  plot<-ggplot(df_t, aes(y = factor(get(variable)), x = es, fill = factor(get(variable)))) +
    geom_boxplot(color = "black", outlier.size = 1, width = 0.3, alpha = 0.7) +  
    geom_jitter(aes(color = factor(get(variable))), size = 0.1, width = 0.15, height = 0, alpha = 0.7) +
    scale_fill_manual(values = palette) +
    scale_color_manual(values = darker_palette) + 
    scale_x_continuous(
      limits = c(0, 5),
      breaks = seq(0, 10, by = 2) 
    )+ theme_classic()+
    labs(fill = variable, x = "", y = "")+
    theme(legend.position = "none") 
  ggsave(paste0("/Volumes/Nan/data/phono/figSuppl/figs/svg/es_1_", variable, ".svg"), plot = plot, width = 3.5, height = 1.3, units = "in")         #change label
}


# glm

model_full <- glm(es ~ feedback_re2_ana + learn_mode + explicitness_instruction, data = dfo, family = gaussian())
summary(model_full)
null_deviance <- model_full$null.deviance
residual_deviance <- model_full$deviance
r_squared <- 1 - (residual_deviance / null_deviance)
model_null <- glm(es ~ 1, data = dfo, family = gaussian())
anova(model_null, model_full, test = "Chisq")
predicted_es <- predict(model_full, type = "response")

plot_data <- data.frame(
  Real = dfo$es, 
  Predicted = predicted_es
)
ggplot(plot_data, aes(x = Predicted, y = Real)) +
  geom_jitter(color = "#E69F00", alpha = 0.7, size = 2.5, width = 0.2, height = 0.2) +
  geom_abline(slope = 1, intercept = 0, color = "#D55E00", linetype = "dashed", linewidth = 1) +
  labs(
    x = "Predicted Effect Size",
    y = "Real Effect Size",
    title = "Predicted vs. Real Effect Size"
  ) +
  theme_minimal(base_size = 25) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    panel.grid.major = element_line(color = "gray80", linetype = "dotted"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.8  # **Increase height relative to width**
  )


# sem
df <- dfo %>%
  select(-modality, -context_variability, -feedback)%>%
  rename(
    talker_var = NRecorder,
    context_var = context_variability_re,
    repetition = n_rep,
    #learn_mode = learn_mode,
    feedback = feedback_re2_ana,
    learn_mode = learn_mode,
    instruction = explicitness_instruction,
    duration = ltraining_duration,
    trial = ltrial_number,
    consolidatio = lconsolidation,
    modality = modality_re
  )%>%
  mutate(
    learn_mode = factor(learn_mode, ordered = TRUE),
    source = factor(source, ordered = TRUE),            
    repetition = factor(repetition, ordered = TRUE),      
    modality = factor(modality, ordered = TRUE),          
    presentation = factor(presentation, ordered = TRUE),    
    instruction = factor(instruction, ordered = TRUE),  
  )

model <- "
# Measurement model
intensity =~ duration + trial + consolidatio
variability =~  talker_var + context_var   + source + repetition + modality + presentation
engagement =~  feedback + learn_mode + instruction

# Structural model 
# Regressions
es ~ intensity
"

fit <- sem(model, data = df)
summary(fit, fit.measures = TRUE, standardized = TRUE)
semPaths(fit, 
         what = "col", whatLabels = "par", 
         style = "mx", 
         color = colorlist,
         edge.color = "black",
         #edge.label.cex = 0.5,
         rotation = 1,  layout = "tree3", 
         mar = c(2, 2, 2, 2), 
         nCharNodes = 8,
         shapeMan = "rectangle", 
         sizeMan = 8, sizeMan2 = 5,
         intercepts = FALSE,   # This hides the triangles for intercepts
         exoCov = FALSE
         #residuals = FALSE
)


# effect size

dfes <- dfo %>%
  dplyr::select(c(Articles, org_order, es, se, vi))

df1 <- escalc(measure="SMD",  data=dfes, yi= es, vi = vi,  slab=paste("Study ID:", Articles))#transfer into escalc format
df1 <- df1[order(df1$es), ]

res <- rma(yi,
           vi,
           data = df1,
)
res
predict_primary <- predict(res)
predict_primary


# sensitivity
metainf(res1)

tiff(file="forest_sensitivity.tiff",
     res=800,width = 9000,height = 20000)#save tiff
forest(metainf(res1),
       #fonts = "times",
)
dev.off()
InA <- InfluenceAnalysis(x = res1, random = TRUE) 
plot(InA)
plot(InA, "influence") 
plot(InA, "I2") 
plot(InA, "es")

tiff(file="funnel.tiff",
     res=800,width = 7600,height = 4800)#put infront of code
funnel(res1, 
       fonts = "times",
       xlab = "Hedges' g")
dev.off()
res2 <- trimfill(res1)
res2
funnel(res2, xlab = "Hedges' g")


