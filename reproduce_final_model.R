# Reproduce the final model described in the original 2021 study.
# Run from the repository root: Rscript reproduce_final_model.R
# This script is a repository preparation supplement, not the original script.

data("state", package = "datasets")
states <- as.data.frame(state.x77)
analysis_data <- states[rownames(states) != "Nevada", , drop = FALSE]

# Original final specification: no intercept, Illiteracy and Population.
final_model <- lm(Murder ~ 0 + Illiteracy + Population, data = analysis_data)

dir.create("results", showWarnings = FALSE)
predictions <- data.frame(
  State = rownames(analysis_data),
  Actual = analysis_data$Murder,
  Fitted = as.numeric(predict(final_model, newdata = analysis_data)),
  Residual = as.numeric(residuals(final_model))
)
write.csv(predictions, "results/fitted_values.csv", row.names = FALSE)
capture.output(summary(final_model), file = "results/model_summary.txt")

pdf("results/regression_diagnostics.pdf", width = 9, height = 9)
par(mfrow = c(2, 2))
plot(final_model)
dev.off()

print(summary(final_model))
cat("\nAIC:", AIC(final_model), "\n")
cat("step() criterion:", extractAIC(final_model)[2], "\n")
cat("\nShapiro-Wilk test:\n")
print(shapiro.test(residuals(final_model)))

if (requireNamespace("car", quietly = TRUE)) {
  cat("\nVIF (model has no intercept):\n")
  print(car::vif(final_model))
  cat("\nNon-constant variance test:\n")
  print(car::ncvTest(final_model))
}
if (requireNamespace("lmtest", quietly = TRUE)) {
  cat("\nDurbin-Watson test (current state row order):\n")
  print(lmtest::dwtest(final_model))
}
